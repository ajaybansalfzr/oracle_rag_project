# # import os
# # import pandas as pd
# import sqlite3 #ok
# import numpy as np #ok
# import faiss #ok
# import pickle #ok
# from pathlib import Path #ok
# from tqdm import tqdm #ok
# from sentence_transformers import SentenceTransformer #ok
# from rank_bm25 import BM25Okapi #ok
# from nltk.tokenize import word_tokenize #ok
# import nltk

# # import re
# # from transformers import AutoTokenizer

# # Import our central database helper
# from helpers.database_helper import get_db_connection

# # === Configuration ===
# EMBEDDING_MODELS = ['all-MiniLM-L6-v2', 'BAAI/bge-small-en-v1.5']  # Add or adjust model names as needed
# # CHUNK_SIZE_TOKENS = 384
# # CHUNK_MIN_TOKENS = 50
# # CHUNK_OVERLAP_SENTENCES = 1 # Overlap by 1 full sentence

# # === Paths ===
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# CSV_FOLDER = PROJECT_ROOT / "output"
# EMBEDDINGS_FOLDER = PROJECT_ROOT / "embeddings"
# EMBEDDINGS_FOLDER.mkdir(exist_ok=True)

# # === Ensure NLTK punkt is available ===
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt', quiet=True)

import pickle
import sqlite3
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Import database helper
from helpers.database_helper import get_db_connection

# --- Configuration (from FSD) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
DB_PATH = PROJECT_ROOT / "output" / "project_data.db"
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# FR-4: Ensemble of models as defined in the FSD
EMBEDDING_MODELS = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]

# As suggested by GEMINI adding a new funciton


def get_model_and_filenames(model_name: str):
    """Generates standardized filenames and column names for a given model."""
    # Creates a filesystem-safe name (e.g., 'BAAI/bge-small-en-v1.5' -> 'BAAI_bge-small-en-v1_5')
    safe_name = model_name.replace("/", "_")
    # Creates a database-safe column name (e.g., 'all-MiniLM-L6-v2' -> 'vector_id_all_minilm_l6_v2')
    # vector_id_col = f"vector_id_{safe_name.replace('-', '_').lower()}"
    vector_id_col = f"vector_id_{safe_name.replace('-', '_').replace('.', '_').lower()}"

    return {
        "model": SentenceTransformer(model_name),
        "faiss_path": EMBEDDINGS_DIR / f"oracle_index_{safe_name}.faiss",
        "metadata_path": EMBEDDINGS_DIR / f"oracle_metadata_{safe_name}.db",
        "bm25_path": EMBEDDINGS_DIR / f"oracle_bm25_{safe_name}.pkl",
        "vector_id_col": vector_id_col,
    }


# # === Robust sentence splitter ===
# def split_sentences(text: str) -> list[str]:
#     if not isinstance(text, str):
#         return []
#     # Split on sentence boundaries (., !, ?) followed by whitespace and uppercase
#     sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
#     return [s.strip() for s in sentences if s.strip()]

# # === A more intelligent, sentence-aware chunker ===
# def robust_chunk_by_sentence(text: str, max_tokens: int, min_tokens: int, overlap_sentences: int, tokenizer) -> list[str]:
#     if not isinstance(text, str) or not text.strip():
#         return []

#     # Use a robust sentence splitter
#     sentences = re.split(r'(?<=[.!?])\s+', text.strip())
#     sentences = [s.strip() for s in sentences if s.strip()]

#     if not sentences:
#         return []

#     chunks = []
#     current_chunk_sentences = []
#     current_chunk_tokens = 0

#     for i, sentence in enumerate(sentences):
#         sentence_token_count = len(tokenizer.encode(sentence))

#         # If adding the next sentence would exceed the max token size, finalize the current chunk
#         if current_chunk_tokens > 0 and (current_chunk_tokens + sentence_token_count > max_tokens):
#             # Only add the chunk if it meets the minimum size
#             if current_chunk_tokens >= min_tokens:
#                 chunks.append(" ".join(current_chunk_sentences))

#             # Start a new chunk, preserving the desired sentence overlap
#             # This is the "sliding window" based on sentences
#             if overlap_sentences > 0:
#                 overlap_index = max(0, len(current_chunk_sentences) - overlap_sentences)
#                 current_chunk_sentences = current_chunk_sentences[overlap_index:]
#                 # Recalculate token count for the overlapping sentences
#                 current_chunk_tokens = sum(len(tokenizer.encode(s)) for s in current_chunk_sentences)
#             else:
#                 current_chunk_sentences = []
#                 current_chunk_tokens = 0

#         # Add the current sentence to the chunk
#         current_chunk_sentences.append(sentence)
#         current_chunk_tokens += sentence_token_count

#     # Add the last remaining chunk if it's not empty
#     if current_chunk_sentences and current_chunk_tokens >= min_tokens:
#         chunks.append(" ".join(current_chunk_sentences))
#     # If the last chunk is too small, try to merge it with the previous one
#     elif chunks and current_chunk_sentences:
#         chunks[-1] += " " + " ".join(current_chunk_sentences)

#     return chunks

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
# def main():
#     """
#     Rebuilds the vector stores from scratch using all summarized content
#     from the central project database. This is a safe, idempotent operation.
#     """
#     project_conn = get_db_connection()

#     for model_name in EMBEDDING_MODELS:
#         sanitized_model_name = model_name.replace("/", "-")
#         print(f"\n>>> Rebuilding vector stores for model: {sanitized_model_name}")

#         vector_id_column = f"vector_id_{'minilm' if 'minilm' in sanitized_model_name else 'bge'}"

#         # --- Delete Old Vector Store Files for a Clean Rebuild ---
#         for pattern in ["*.db", "*.faiss", "*.pkl"]:
#             for f in EMBEDDINGS_FOLDER.glob(f"*{sanitized_model_name}*{pattern}"):
#                 f.unlink()
#         print("  - Removed old vector store files for this model.")

#         # --- 1. Find Work: Identify chunks that need to be embedded for this model ---
#         query = f"""
#             SELECT c.chunk_id, c.summary,
#                 s.section_header, s.page_num, s.header_text, s.footer_text,
#                 s.hyperlink_text, s.table_text,
#                 d.doc_name
#             FROM chunks c
#             JOIN sections s ON c.section_id = s.section_id
#             JOIN documents d ON s.doc_id = d.doc_id
#             WHERE c.summary IS NOT NULL AND c.summary != ''
#         """
#         all_chunks = project_conn.execute(query).fetchall()
#         if not all_chunks:
#             print(f"  - No summarized chunks found. Nothing to embed.")
#             continue
#         print(f"  - Found {len(all_chunks)} total summarized chunks to process.")

#         # rows_to_process = project_conn.execute(query).fetchall()

#         # if not rows_to_process:
#         #     print(f"  -  All content is already embedded for model {sanitized_model_name}.")
#         #     continue

#         # print(f"  - Found {len(rows_to_process)} new chunks to embed.")

#         # --- 2. Load Models and Existing Vector Stores ---
#         model = SentenceTransformer(model_name)
#         embedding_dim = model.get_sentence_embedding_dimension()

#         meta_db_path = EMBEDDINGS_FOLDER / f"oracle_metadata_{sanitized_model_name}.db"
#         meta_conn = sqlite3.connect(meta_db_path)
#         meta_conn.row_factory = sqlite3.Row
#         meta_cursor = meta_conn.cursor()
#         meta_cursor.execute("""
#         CREATE TABLE IF NOT EXISTS metadata (
#             vector_id INTEGER PRIMARY KEY,
#             chunk_id INTEGER NOT NULL UNIQUE,
#             document_name TEXT,
#             page_num INTEGER,
#             section_id TEXT,
#             indexed_text TEXT,
#             header_text TEXT,
#             footer_text TEXT,
#             hyperlink_text TEXT,
#             table_text TEXT
#             );
#         """)

#         faiss_path = EMBEDDINGS_FOLDER / f"oracle_index_{sanitized_model_name}.faiss"
#         if faiss_path.exists():
#             index = faiss.read_index(str(faiss_path))
#         else:
#             index = faiss.IndexFlatL2(embedding_dim)
#             index = faiss.IndexIDMap(index)

#         next_vector_id = index.ntotal

#         # --- 3. Process New Chunks ---
#         summaries_to_embed = [row['summary'] for row in rows_to_process]

#         print("  - Generating embeddings for new chunk summaries...")
#         new_embeddings = model.encode(
#             summaries_to_embed, show_progress_bar=True, batch_size=32
#         )

#         faiss_ids_to_add = [next_vector_id + i for i in range(len(rows_to_process))]

#         metadata_to_add = []
#         project_db_updates = []
#         for i, row in enumerate(rows_to_process):
#             current_vector_id = faiss_ids_to_add[i]
#             metadata_to_add.append((
#                 current_vector_id, row['chunk_id'], row['doc_name'],
#                 row['page_num'], row['section_header'], row['summary'],
#                 row['header_text'], row['footer_text'],
#                 row['hyperlink_text'], row['table_text']
#             ))
#             project_db_updates.append((current_vector_id, row['chunk_id']))

#         # --- 4. Update and Save Vector Stores ---
#         if faiss_ids_to_add:
#             index.add_with_ids(np.array(new_embeddings, dtype=np.float32), np.array(faiss_ids_to_add))
#             faiss.write_index(index, str(faiss_path))
#             print(f"  -  FAISS index updated. Total vectors: {index.ntotal}")

#             meta_cursor.executemany(
#                 """
#                 INSERT INTO metadata (
#                     vector_id, chunk_id, document_name, page_num, section_id,
#                     indexed_text, header_text, footer_text, hyperlink_text, table_text
#                 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#                 """,
#                 metadata_to_add
#             )
#             meta_conn.commit()
#             print(f"  -  Model-specific metadata DB updated.")

#             update_query = f"UPDATE chunks SET {vector_id_column} = ? WHERE chunk_id = ?"
#             project_conn.executemany(update_query, project_db_updates)
#             project_conn.commit()
#             print(f"  -  Central project database updated with new vector IDs.")

#         # --- 5. Rebuild and Save BM25 Index from the full, updated metadata ---
#         print("  - Rebuilding BM25 index for lexical search...")
#         all_meta_rows = meta_conn.execute("SELECT chunk_id, indexed_text FROM metadata ORDER BY vector_id").fetchall()

#         bm25_corpus = [word_tokenize(row['indexed_text'].lower()) for row in all_meta_rows]
#         bm25 = BM25Okapi(bm25_corpus)
#         bm25_doc_map = {i: row['chunk_id'] for i, row in enumerate(all_meta_rows)}

#         bm25_path = EMBEDDINGS_FOLDER / f"oracle_bm25_{sanitized_model_name}.pkl"
#         with open(bm25_path, 'wb') as f:
#             pickle.dump({'index': bm25, 'doc_map': bm25_doc_map}, f)
#         print(f"  -  BM25 index rebuilt and saved with {len(all_meta_rows)} documents.")

#         meta_conn.close()

#     # --- 6. Final Status Update in Project DB ---
#     project_conn.execute("""
#         UPDATE documents SET status = 'embedded' WHERE doc_id IN (
#             SELECT d.doc_id FROM documents d
#             JOIN sections s ON d.doc_id = s.doc_id
#             JOIN chunks c ON s.section_id = c.section_id
#             GROUP BY d.doc_id
#             HAVING MIN(c.vector_id_minilm IS NOT NULL) AND MIN(c.vector_id_bge IS NOT NULL)
#         ) AND status != 'embedded'
#     """)
#     project_conn.commit()
#     project_conn.close()
#     print("\n Embedding process complete for all models.")

# if __name__ == '__main__':
#     main()


def main():
    """
    Main function to generate and store embeddings and indexes for all models.
    """
    print(" Starting embedding and indexing process...")

    for model_name in EMBEDDING_MODELS:
        print(f"\n--- Processing model: {model_name} ---")

        config = get_model_and_filenames(model_name)
        model = config["model"]

        print("  - Deleting existing index files for a clean rebuild...")
        for path in [
            config["faiss_path"],
            config["metadata_path"],
            config["bm25_path"],
        ]:
            path.unlink(missing_ok=True)

        conn_main = get_db_connection()
        cursor_main = conn_main.cursor()

        cursor_main.execute(
            f"""
            SELECT c.chunk_id, c.summary
            FROM chunks c
            JOIN sections s ON c.section_id = s.section_id
            JOIN documents d ON s.doc_id = d.doc_id
            WHERE d.status = 'chunked_and_summarized' AND c.{config["vector_id_col"]} IS NULL
        """
        )
        summaries_to_embed = cursor_main.fetchall()

        if not summaries_to_embed:
            print("  -  No new summaries to embed for this model.")
            conn_main.close()
            continue

        print(f"  - Found {len(summaries_to_embed)} summaries to embed.")
        chunk_ids, summary_texts = zip(*[(r["chunk_id"], r["summary"]) for r in summaries_to_embed])

        print("  - Generating vector embeddings (this may take a while)...")
        embeddings = model.encode(summary_texts, show_progress_bar=True)
        dimension = embeddings.shape[1]

        print(f"  - Building FAISS index (dimension: {dimension})...")
        index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
        index.add_with_ids(np.array(embeddings, dtype=np.float32), np.array(chunk_ids))
        faiss.write_index(index, str(config["faiss_path"]))
        print(f"  -  FAISS index saved to {config['faiss_path']}")

        print("  - Building BM25 index...")
        tokenized_summaries = [doc.split(" ") for doc in summary_texts]
        bm25 = BM25Okapi(tokenized_summaries)
        with open(config["bm25_path"], "wb") as f:
            pickle.dump(bm25, f)
        print(f"  -  BM25 index saved to {config['bm25_path']}")

        print("  - Creating decoupled metadata database...")
        conn_meta = sqlite3.connect(config["metadata_path"])
        cursor_meta = conn_meta.cursor()
        cursor_meta.execute(
            """
            CREATE TABLE metadata (
                chunk_id INTEGER PRIMARY KEY, doc_name TEXT, page_num INTEGER,
                section_header TEXT, summary TEXT, raw_text TEXT
            )
        """
        )

        cursor_main.execute(
            """
            SELECT c.chunk_id, d.doc_name, s.page_num, s.section_header, c.summary, c.chunk_text
            FROM chunks c
            JOIN sections s ON c.section_id = s.section_id
            JOIN documents d ON s.doc_id = d.doc_id
        """
        )
        all_metadata = cursor_main.fetchall()

        cursor_meta.executemany(
            "INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    r["chunk_id"],
                    r["doc_name"],
                    r["page_num"],
                    r["section_header"],
                    r["summary"],
                    r["chunk_text"],
                )
                for r in all_metadata
            ],
        )
        conn_meta.commit()
        conn_meta.close()
        print(f"  -  Metadata DB saved to {config['metadata_path']}")

        print("  - Writing vector IDs back to the main project database...")
        update_data = [(chunk_id, chunk_id) for chunk_id in chunk_ids]
        cursor_main.executemany(
            f"UPDATE chunks SET {config['vector_id_col']} = ? WHERE chunk_id = ?",
            update_data,
        )
        conn_main.commit()
        print(f"  -  Main DB updated for model {model_name}.")
        conn_main.close()

    print("\n--- Finalizing document statuses ---")
    conn_final = get_db_connection()
    cursor_final = conn_final.cursor()

    vector_id_cols = [get_model_and_filenames(m)["vector_id_col"] for m in EMBEDDING_MODELS]

    cursor_final.execute(
        f"""
        UPDATE documents
        SET status = 'embedded'
        WHERE doc_id IN (
            SELECT d.doc_id
            FROM documents d
            JOIN sections s ON d.doc_id = s.doc_id
            JOIN chunks c ON s.section_id = c.section_id
            WHERE d.status = 'chunked_and_summarized'
            GROUP BY d.doc_id
            HAVING COUNT(CASE WHEN {' AND '.join([f'c.{col} IS NOT NULL' for col in vector_id_cols])} THEN 1 END) = COUNT(c.chunk_id)
        )
    """
    )
    updated_count = cursor_final.rowcount
    conn_final.commit()
    conn_final.close()

    if updated_count > 0:
        print(f"  -  Successfully updated status for {updated_count} document(s) to 'embedded'.")
    else:
        print("  - No documents were ready for final status update.")

    print("\n>> All embedding and indexing processes are complete.")


if __name__ == "__main__":
    main()


# # === Main execution ===
# def main():
#     # Prepare folders
#     EMBEDDINGS_FOLDER.mkdir(exist_ok=True)

#     # Cleanup old indexes and databases
#     print(">>> Cleaning up old index and DB files...")
#     for pattern in ["*.db", "*.faiss", "*.pkl"]:
#         for f in EMBEDDINGS_FOLDER.glob(pattern):
#             try:
#                 f.unlink()
#                 print(f"  - Removed {f.name}")
#             except Exception:
#                 pass

#     # Find CSVs
#     csv_files = list(CSV_FOLDER.glob("extracted_oracle_pdf_final_*.csv"))
#     if not csv_files:
#         print(f" No 'extracted_oracle_pdf_final_*.csv' files found in {CSV_FOLDER}")
#         return

#     # Process each embedding model
#     for model_name in EMBEDDING_MODELS:
#         print(f"\n>>> Processing model: {model_name}")

#         sanitized_model_name = model_name.replace("/", "-")

#         # Setup paths for this model
#         db_path = EMBEDDINGS_FOLDER / f"oracle_metadata_{sanitized_model_name}.db"
#         faiss_path = EMBEDDINGS_FOLDER / f"oracle_index_{sanitized_model_name}.faiss"
#         bm25_path = EMBEDDINGS_FOLDER / f"oracle_bm25_{sanitized_model_name}.pkl"

#         # Load embedding model & tokenizer
#         print(f">>> Loading embedding model and tokenizer for {model_name}...")
#         model = SentenceTransformer(model_name)
#         tokenizer = model.tokenizer # Get the tokenizer directly from the loaded model
#         # try:
#         #     tokenizer = AutoTokenizer.from_pretrained(model_name)
#         #     token_based = True
#         # except Exception:
#         #     tokenizer = None
#         #     token_based = False
#         embedding_dim = model.get_sentence_embedding_dimension()

#         # Initialize FAISS
#         index = faiss.IndexFlatL2(embedding_dim)
#         index = faiss.IndexIDMap(index)

#         # Initialize SQLite DB and metadata table
#         conn = sqlite3.connect(db_path)
#         c = conn.cursor()
#         c.execute("DROP TABLE IF EXISTS metadata")
#         c.execute("""
#             CREATE TABLE metadata (
#                 vector_id    INTEGER PRIMARY KEY,
#                 chunk_id     TEXT    NOT NULL UNIQUE,
#                 document_name TEXT,
#                 page_num     INTEGER,
#                 section_id   TEXT,
#                 indexed_text TEXT,
#                 indexed_text_tagged TEXT,
#                 header_text  TEXT,
#                 footer_text  TEXT,
#                 hyperlink_text TEXT,
#                 table_text   TEXT
#             )
#         """)
#         conn.commit()

#         vector_id = 0
#         bm25_corpus = []

#         # Iterate through all CSV files
#         for csv_file in csv_files:
#             match = re.search(r'extracted_oracle_pdf_final_(.+)\.csv', csv_file.name)
#             if not match:
#                 continue
#             doc_name = match.group(1)

#             print(f" Embedding document: {doc_name}")
#             df = pd.read_csv(csv_file).fillna('')

#             for _, row in df.iterrows():
#                 text = row.get('indexed_text', '')
#                 if not text.strip():
#                     continue

#                 # Chunk the text
#                 chunks = robust_chunk_by_sentence(
#                     text,
#                     max_tokens=CHUNK_SIZE_TOKENS,
#                     min_tokens=CHUNK_MIN_TOKENS,
#                     overlap_sentences=CHUNK_OVERLAP_SENTENCES,
#                     tokenizer=tokenizer,
#                 )
#                 if not chunks:
#                     chunks = [text]

#                 # Embed and store each chunk
#                 for idx, chunk in enumerate(chunks, start=1):
#                     # Prepare IDs and tags
#                     chunk_id = f"{doc_name}_pg{row['page_num']}_sec{row['section_id']}_chunk{idx}"
#                     metadata_tag = f"[Page:{row['page_num']}][Section:{row['section_id']}][Chunk:{idx}]"
#                     indexed_text = chunk.strip()
#                     indexed_text_tagged = f"{metadata_tag} {indexed_text}"

#                     # Compute embedding
#                     embedding = model.encode([indexed_text])
#                     index.add_with_ids(
#                         np.array(embedding, dtype=np.float32),
#                         np.array([vector_id], dtype=np.int64)
#                     )

#                     # Prepare BM25
#                     tokens = word_tokenize(indexed_text.lower())
#                     bm25_corpus.append({'id': chunk_id, 'tokens': tokens})

#                     # Insert metadata
#                     c.execute(
#                         """
#                         INSERT INTO metadata (
#                           vector_id, chunk_id, document_name, page_num, section_id,
#                           indexed_text, indexed_text_tagged,
#                           header_text, footer_text, hyperlink_text, table_text
#                         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#                         """,
#                         (
#                             vector_id,
#                             chunk_id,
#                             doc_name,
#                             int(row['page_num']),
#                             row['section_id'],
#                             indexed_text,
#                             indexed_text_tagged,
#                             row.get('header_text', ''),
#                             row.get('footer_text', ''),
#                             row.get('hyperlink_text', ''),
#                             row.get('table_text', '')
#                         )
#                     )
#                     vector_id += 1

#         # Finalize DB
#         conn.commit()
#         conn.close()

#         # Save FAISS index
#         faiss.write_index(index, str(faiss_path))
#         print(f"  - Saved FAISS index ({index.ntotal} vectors) to {faiss_path.name}")

#         # Save BM25 index
#         if bm25_corpus:
#             bm25 = BM25Okapi([item['tokens'] for item in bm25_corpus])
#             doc_map = {i: item['id'] for i, item in enumerate(bm25_corpus)}
#             with open(bm25_path, 'wb') as f:
#                 pickle.dump({'index': bm25, 'doc_map': doc_map}, f)
#             print(f"  - Saved BM25 index for {len(bm25_corpus)} chunks to {bm25_path.name}")

#     print("\n All embedding models processed successfully.")

# if __name__ == '__main__':
#     main()
