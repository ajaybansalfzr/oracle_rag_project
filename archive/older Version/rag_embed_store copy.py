# scripts/rag_embed_store.py (FINAL - CORRECTED DATA HANDLING)

import os
import pickle
import re
import sqlite3
from pathlib import Path

import faiss
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# --- Setup ---
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt", quiet=True)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_FOLDER = PROJECT_ROOT / "output"
EMBEDDINGS_FOLDER = PROJECT_ROOT / "embeddings"
MODEL_NAME = "all-MiniLM-L6-v2"
DB_PATH = EMBEDDINGS_FOLDER / "oracle_metadata.db"
FAISS_PATH = EMBEDDINGS_FOLDER / "oracle_index.faiss"
BM25_PATH = EMBEDDINGS_FOLDER / "oracle_bm25.pkl"
CHUNK_SIZE_TOKENS = 384
CHUNK_OVERLAP_TOKENS = 50

# --- Initialization ---
EMBEDDINGS_FOLDER.mkdir(exist_ok=True)
print(">>> Cleaning up old index files (if any)...")
for path in [DB_PATH, FAISS_PATH, BM25_PATH]:
    if path.exists():
        os.remove(path)
        print(f"  - Removed {path.name}")

print("\n>>> Loading embedding model and tokenizer...")
model_semantic = SentenceTransformer(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{MODEL_NAME}")
print("Models loaded successfully.")

embedding_dim = model_semantic.get_sentence_embedding_dimension()
index_faiss = faiss.IndexFlatL2(embedding_dim)
index_faiss = faiss.IndexIDMap(index_faiss)

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

c.execute(
    """
    CREATE TABLE metadata (
        vector_id INTEGER PRIMARY KEY,
        chunk_id TEXT NOT NULL UNIQUE,
        document_name TEXT,
        page_num INTEGER,
        section_id TEXT,
        chunk_text TEXT,
        header_text TEXT,
        footer_text TEXT,
        hyperlink_text TEXT,
        table_text TEXT
    )
"""
)
conn.commit()


def text_to_token_aware_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text or not isinstance(text, str):
        return []
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return []
    if len(tokens) <= chunk_size:
        return [tokenizer.decode(tokens, skip_special_tokens=True)]
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i : i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text)
    return chunks


def main():
    vector_id_counter = 0
    bm25_corpus = []

    csv_files = list(CSV_FOLDER.glob("extracted_oracle_pdf_final_*.csv"))
    if not csv_files:
        print(f"❌ No 'extracted_oracle_pdf_final_*.csv' files found in '{CSV_FOLDER}'.")
        return

    print(f"\nFound {len(csv_files)} processed CSV files to embed.")

    for csv_file in csv_files:
        doc_name_match = re.search(r"extracted_oracle_pdf_final_(.+)\.csv", csv_file.name)
        if not doc_name_match:
            continue
        doc_name = doc_name_match.group(1)

        print(f"\n▶️ Processing '{doc_name}' for embedding...")
        df = pd.read_csv(csv_file).fillna("")

        for _, row in df.iterrows():
            # Create a clean text body for searching, excluding the noisy section_id.
            searchable_text = ". ".join(filter(None, [row.get("indexed_text", ""), row.get("table_text", "")]))

            text_chunks = text_to_token_aware_chunks(
                searchable_text,
                chunk_size=CHUNK_SIZE_TOKENS,
                overlap=CHUNK_OVERLAP_TOKENS,
            )

            if not text_chunks and searchable_text.strip():
                text_chunks = [searchable_text]

            for chunk in text_chunks:
                chunk_id = f"{doc_name}_pg{row['page_num']}_vec{vector_id_counter}"

                embedding = model_semantic.encode([chunk])
                index_faiss.add_with_ids(np.array(embedding, dtype=np.float32), np.array([vector_id_counter]))

                bm25_corpus.append({"id": chunk_id, "tokens": word_tokenize(chunk.lower())})

                c.execute(
                    """
                    INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        vector_id_counter,
                        chunk_id,
                        doc_name,
                        int(row["page_num"]),
                        row["section_id"],
                        chunk,
                        row["header_text"],
                        row["footer_text"],
                        row["hyperlink_text"],
                        row["table_text"],
                    ),
                )

                vector_id_counter += 1

    print("\n✅ All files processed. Finalizing and saving indexes...")
    conn.commit()
    conn.close()

    faiss.write_index(index_faiss, str(FAISS_PATH))
    print(f"  - Saved FAISS index with {index_faiss.ntotal} vectors to '{FAISS_PATH}'")

    if bm25_corpus:
        bm25 = BM25Okapi([item["tokens"] for item in bm25_corpus])
        bm25_doc_index_to_chunk_id = {i: item["id"] for i, item in enumerate(bm25_corpus)}
        bm25_data = {"index": bm25, "doc_map": bm25_doc_index_to_chunk_id}
        with open(BM25_PATH, "wb") as f:
            pickle.dump(bm25_data, f)
        print(f"  - Saved BM25 index for {len(bm25_corpus)} chunks to '{BM25_PATH}'")

    print(f"\nTotal chunks embedded: {vector_id_counter}")
    print(">> Embedding process complete.")


if __name__ == "__main__":
    main()
