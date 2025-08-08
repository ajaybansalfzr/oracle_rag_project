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

# === Configuration ===
EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]  # Add or adjust model names as needed
CHUNK_SIZE_TOKENS = 384
CHUNK_MIN_TOKENS = 50
CHUNK_OVERLAP_SENTENCES = 1  # Overlap by 1 full sentence

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_FOLDER = PROJECT_ROOT / "output"
EMBEDDINGS_FOLDER = PROJECT_ROOT / "embeddings"

# === Ensure NLTK punkt is available ===
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# # === Robust sentence splitter ===
# def split_sentences(text: str) -> list[str]:
#     if not isinstance(text, str):
#         return []
#     # Split on sentence boundaries (., !, ?) followed by whitespace and uppercase
#     sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
#     return [s.strip() for s in sentences if s.strip()]


# === A more intelligent, sentence-aware chunker ===
def robust_chunk_by_sentence(
    text: str, max_tokens: int, min_tokens: int, overlap_sentences: int, tokenizer
) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    # Use a robust sentence splitter
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0

    for i, sentence in enumerate(sentences):
        sentence_token_count = len(tokenizer.encode(sentence))

        # If adding the next sentence would exceed the max token size, finalize the current chunk
        if current_chunk_tokens > 0 and (current_chunk_tokens + sentence_token_count > max_tokens):
            # Only add the chunk if it meets the minimum size
            if current_chunk_tokens >= min_tokens:
                chunks.append(" ".join(current_chunk_sentences))

            # Start a new chunk, preserving the desired sentence overlap
            # This is the "sliding window" based on sentences
            if overlap_sentences > 0:
                overlap_index = max(0, len(current_chunk_sentences) - overlap_sentences)
                current_chunk_sentences = current_chunk_sentences[overlap_index:]
                # Recalculate token count for the overlapping sentences
                current_chunk_tokens = sum(len(tokenizer.encode(s)) for s in current_chunk_sentences)
            else:
                current_chunk_sentences = []
                current_chunk_tokens = 0

        # Add the current sentence to the chunk
        current_chunk_sentences.append(sentence)
        current_chunk_tokens += sentence_token_count

    # Add the last remaining chunk if it's not empty
    if current_chunk_sentences and current_chunk_tokens >= min_tokens:
        chunks.append(" ".join(current_chunk_sentences))
    # If the last chunk is too small, try to merge it with the previous one
    elif chunks and current_chunk_sentences:
        chunks[-1] += " " + " ".join(current_chunk_sentences)

    return chunks


# # === Robust token-aware chunker ===
# def robust_chunk(text: str, max_tokens: int, min_tokens: int, overlap_tokens: int, tokenizer, token_based: bool) -> list[str]:
#     sentences = split_sentences(text)
#     chunks = []
#     current_chunk = []
#     current_len = 0

#     for sentence in sentences:
#         tokens = tokenizer.tokenize(sentence) if token_based else list(sentence)
#         sent_len = len(tokens)

#         # If adding this sentence exceeds max, finalize the chunk
#         if current_len + sent_len > max_tokens:
#             if current_len >= min_tokens:
#                 chunks.append(" ".join(current_chunk))
#                 # keep overlap
#                 overlap = current_chunk[-overlap_tokens:] if overlap_tokens else []
#                 current_chunk = overlap.copy()
#                 current_len = sum(len(tokenizer.tokenize(s)) if token_based else len(s) for s in current_chunk)
#             else:
#                 # force include small chunk
#                 current_chunk.append(sentence)
#                 chunks.append(" ".join(current_chunk))
#                 current_chunk = []
#                 current_len = 0

#         else:
#             current_chunk.append(sentence)
#             current_len += sent_len

#     if current_chunk:
#         chunks.append(" ".join(current_chunk))

#     # Merge very small trailing chunks
#     final_chunks = []
#     for chunk in chunks:
#         tokens = tokenizer.tokenize(chunk) if token_based else list(chunk)
#         if final_chunks and len(tokens) < min_tokens:
#             final_chunks[-1] += " " + chunk
#         else:
#             final_chunks.append(chunk)

#     return final_chunks


# === Main execution ===
def main():
    # Prepare folders
    EMBEDDINGS_FOLDER.mkdir(exist_ok=True)

    # Cleanup old indexes and databases
    print(">>> Cleaning up old index and DB files...")
    for pattern in ["*.db", "*.faiss", "*.pkl"]:
        for f in EMBEDDINGS_FOLDER.glob(pattern):
            try:
                f.unlink()
                print(f"  - Removed {f.name}")
            except Exception:
                pass

    # Find CSVs
    csv_files = list(CSV_FOLDER.glob("extracted_oracle_pdf_final_*.csv"))
    if not csv_files:
        print(f"❌ No 'extracted_oracle_pdf_final_*.csv' files found in {CSV_FOLDER}")
        return

    # Process each embedding model
    for model_name in EMBEDDING_MODELS:
        print(f"\n>>> Processing model: {model_name}")

        sanitized_model_name = model_name.replace("/", "-")

        # Setup paths for this model
        db_path = EMBEDDINGS_FOLDER / f"oracle_metadata_{sanitized_model_name}.db"
        faiss_path = EMBEDDINGS_FOLDER / f"oracle_index_{sanitized_model_name}.faiss"
        bm25_path = EMBEDDINGS_FOLDER / f"oracle_bm25_{sanitized_model_name}.pkl"

        # Load embedding model & tokenizer
        print(f">>> Loading embedding model and tokenizer for {model_name}...")
        model = SentenceTransformer(model_name)
        tokenizer = model.tokenizer  # Get the tokenizer directly from the loaded model
        # try:
        #     tokenizer = AutoTokenizer.from_pretrained(model_name)
        #     token_based = True
        # except Exception:
        #     tokenizer = None
        #     token_based = False
        embedding_dim = model.get_sentence_embedding_dimension()

        # Initialize FAISS
        index = faiss.IndexFlatL2(embedding_dim)
        index = faiss.IndexIDMap(index)

        # Initialize SQLite DB and metadata table
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS metadata")
        c.execute(
            """
            CREATE TABLE metadata (
                vector_id    INTEGER PRIMARY KEY,
                chunk_id     TEXT    NOT NULL UNIQUE,
                document_name TEXT,
                page_num     INTEGER,
                section_id   TEXT,
                indexed_text TEXT,
                indexed_text_tagged TEXT,
                header_text  TEXT,
                footer_text  TEXT,
                hyperlink_text TEXT,
                table_text   TEXT
            )
        """
        )
        conn.commit()

        vector_id = 0
        bm25_corpus = []

        # Iterate through all CSV files
        for csv_file in csv_files:
            match = re.search(r"extracted_oracle_pdf_final_(.+)\.csv", csv_file.name)
            if not match:
                continue
            doc_name = match.group(1)

            print(f"▶️ Embedding document: {doc_name}")
            df = pd.read_csv(csv_file).fillna("")

            for _, row in df.iterrows():
                text = row.get("indexed_text", "")
                if not text.strip():
                    continue

                # Chunk the text
                chunks = robust_chunk_by_sentence(
                    text,
                    max_tokens=CHUNK_SIZE_TOKENS,
                    min_tokens=CHUNK_MIN_TOKENS,
                    overlap_sentences=CHUNK_OVERLAP_SENTENCES,
                    tokenizer=tokenizer,
                )
                if not chunks:
                    chunks = [text]

                # Embed and store each chunk
                for idx, chunk in enumerate(chunks, start=1):
                    # Prepare IDs and tags
                    chunk_id = f"{doc_name}_pg{row['page_num']}_sec{row['section_id']}_chunk{idx}"
                    metadata_tag = f"[Page:{row['page_num']}][Section:{row['section_id']}][Chunk:{idx}]"
                    indexed_text = chunk.strip()
                    indexed_text_tagged = f"{metadata_tag} {indexed_text}"

                    # Compute embedding
                    embedding = model.encode([indexed_text])
                    index.add_with_ids(
                        np.array(embedding, dtype=np.float32),
                        np.array([vector_id], dtype=np.int64),
                    )

                    # Prepare BM25
                    tokens = word_tokenize(indexed_text.lower())
                    bm25_corpus.append({"id": chunk_id, "tokens": tokens})

                    # Insert metadata
                    c.execute(
                        """
                        INSERT INTO metadata (
                          vector_id, chunk_id, document_name, page_num, section_id,
                          indexed_text, indexed_text_tagged,
                          header_text, footer_text, hyperlink_text, table_text
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            vector_id,
                            chunk_id,
                            doc_name,
                            int(row["page_num"]),
                            row["section_id"],
                            indexed_text,
                            indexed_text_tagged,
                            row.get("header_text", ""),
                            row.get("footer_text", ""),
                            row.get("hyperlink_text", ""),
                            row.get("table_text", ""),
                        ),
                    )
                    vector_id += 1

        # Finalize DB
        conn.commit()
        conn.close()

        # Save FAISS index
        faiss.write_index(index, str(faiss_path))
        print(f"  - Saved FAISS index ({index.ntotal} vectors) to {faiss_path.name}")

        # Save BM25 index
        if bm25_corpus:
            bm25 = BM25Okapi([item["tokens"] for item in bm25_corpus])
            doc_map = {i: item["id"] for i, item in enumerate(bm25_corpus)}
            with open(bm25_path, "wb") as f:
                pickle.dump({"index": bm25, "doc_map": doc_map}, f)
            print(f"  - Saved BM25 index for {len(bm25_corpus)} chunks to {bm25_path.name}")

    print("\n✅ All embedding models processed successfully.")


if __name__ == "__main__":
    main()
