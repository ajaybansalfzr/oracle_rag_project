import pickle
import sqlite3

import faiss
import nltk
import numpy as np
from rank_bm25 import BM25Okapi

# from pathlib import Path
from sentence_transformers import SentenceTransformer

# Refactored imports to use centralized config and utils
from scripts import config
from scripts.utils.database_utils import get_db_connection
from scripts.utils.utils import get_logger, get_safe_model_name

# # UPDATED: Import new utilities
# from scripts.utils.database_utils import get_db_connection
# from scripts.utils.utils import get_logger


# # Import database helper
# from helpers.database_helper import get_db_connection

# # --- Configuration (from FSD) ---
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
# DB_PATH = PROJECT_ROOT / "output" / "project_data.db"
# EMBEDDINGS_DIR.mkdir(exist_ok=True)

# # FR-4: Ensemble of models as defined in the FSD
# EMBEDDING_MODELS = ['all-MiniLM-L6-v2', 'BAAI/bge-small-en-v1.5']


# NEW: Initialize logger
logger = get_logger(__name__)

# NEW: Ensure NLTK punkt is available for BM25
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    logger.info("Downloading 'punkt' model for NLTK.")
    nltk.download("punkt", quiet=True)


# def get_model_config(model_name: str, index_type: str):
#     """Generates standardized filenames for a given model and index type."""
#     safe_name = model_name.replace('/', '_')
#     vector_id_col = f"vector_id_{safe_name.replace('-', '_').replace('.', '_').lower()}"

#     # Filenames are now distinguished by type (chunk vs section)
#     faiss_path = EMBEDDINGS_DIR / f"{index_type}_index_{safe_name}.faiss"
#     meta_path = EMBEDDINGS_DIR / f"{index_type}_metadata_{safe_name}.db"
#     bm25_path = EMBEDDINGS_DIR / f"{index_type}_bm25_{safe_name}.pkl"

#     return {
#         "model": SentenceTransformer(model_name),
#         "faiss_path": faiss_path,
#         "metadata_path": meta_path,
#         "bm25_path": bm25_path,
#         "vector_id_col": vector_id_col
#     }


def create_indexes_for_tier(data: list, model, model_name: str, tier: str):
    """
    Creates FAISS, BM25, and metadata DB for a given tier ('chunk' or 'section').
    Implements FR-14 (Dual Index) and FR-15 (Decoupled Metadata).
    """
    if not data:
        logger.info(f"No new items to process for tier '{tier}'. Skipping.")
        return

    id_col = f"{tier}_id"
    summary_col = "summary" if tier == "chunk" else "section_summary"

    ids = [row[id_col] for row in data]
    summaries = [row[summary_col] for row in data]

    # --- Artifact Path Generation (using safe names) ---
    safe_name = get_safe_model_name(model_name)
    faiss_path = config.EMBEDDINGS_DIR / f"{tier}_index_{safe_name}.faiss"
    meta_path = config.EMBEDDINGS_DIR / f"{tier}_metadata_{safe_name}.db"
    bm25_path = config.EMBEDDINGS_DIR / f"{tier}_bm25_{safe_name}.pkl"

    # Per FSD, section indexes are rebuilt, chunk indexes are additive.
    if tier == "section":
        logger.warning(f"Rebuilding all '{tier}' artifacts for model '{model_name}' as per FSD strategy.")
        faiss_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        bm25_path.unlink(missing_ok=True)

    # --- 1. FAISS Index (Semantic) ---
    logger.info(f"Generating {len(summaries)} embeddings for tier '{tier}'...")
    embeddings = model.encode(summaries, show_progress_bar=True, batch_size=128)
    dimension = embeddings.shape[1]

    if faiss_path.exists():
        logger.info(f"Loading existing FAISS index from {faiss_path}")
        index = faiss.read_index(str(faiss_path))
    else:
        logger.info("Creating new FAISS index.")
        index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))

    index.add_with_ids(np.array(embeddings, dtype=np.float32), np.array(ids))
    faiss.write_index(index, str(faiss_path))
    logger.info(f"FAISS index for tier '{tier}' saved. Total vectors: {index.ntotal}")

    # --- 2. Metadata DB (for fast lookups) ---
    is_new_meta_db = not meta_path.exists()
    conn_meta = sqlite3.connect(meta_path)
    if is_new_meta_db:
        conn_meta.execute(
            f"""CREATE TABLE metadata (
                {id_col} INTEGER PRIMARY KEY, doc_name TEXT, page_num INTEGER,
                section_header TEXT, summary TEXT, raw_text TEXT)"""
        )

    meta_data_to_add = [
        (
            row[id_col],
            row["doc_name"],
            row["page_num"],
            row["section_header"],
            row[summary_col],
            row["raw_text"],
        )
        for row in data
    ]
    conn_meta.executemany("INSERT OR REPLACE INTO metadata VALUES (?, ?, ?, ?, ?, ?)", meta_data_to_add)
    conn_meta.commit()
    conn_meta.close()
    logger.info(f"Metadata DB for tier '{tier}' updated.")

    # --- 3. BM25 Index (Lexical) ---
    all_meta_conn = sqlite3.connect(meta_path)
    all_meta_conn.row_factory = sqlite3.Row
    all_items = all_meta_conn.execute(f"SELECT summary, {id_col} FROM metadata").fetchall()

    tokenized_corpus = [nltk.word_tokenize(doc["summary"].lower()) for doc in all_items]
    bm25_index = BM25Okapi(tokenized_corpus)
    bm25_doc_map = {i: item[id_col] for i, item in enumerate(all_items)}
    with open(bm25_path, "wb") as f:
        pickle.dump({"index": bm25_index, "doc_map": bm25_doc_map}, f)
    logger.info(f"BM25 index for tier '{tier}' rebuilt and saved with {len(all_items)} documents.")
    all_meta_conn.close()


# def process_embeddings_for_model(model_name: str):
#     """Processes both chunk and section embeddings for a single model."""
#     logger.info(f"--- Processing model: {model_name} ---")

#     # --- Tier 1: Process Chunk Embeddings ---
#     logger.info("Processing Tier 1 (Chunk) embeddings...")
#     config_chunk = get_model_config(model_name, "chunk")

#     conn = get_db_connection()
#     # Find new chunks to embed for this model
#     query_chunks = f"""
#         SELECT c.chunk_id, c.summary, d.doc_name, s.page_num, s.section_header, c.chunk_text
#         FROM chunks c
#         JOIN sections s ON c.section_id = s.section_id
#         JOIN documents d ON s.doc_id = d.doc_id
#         WHERE d.status = 'chunked_and_summarized' AND c.{config_chunk['vector_id_col']} IS NULL
#     """
#     chunks_to_process = conn.execute(query_chunks).fetchall()

#     if chunks_to_process:
#         update_vector_store(chunks_to_process, config_chunk, 'chunk_id', 'summary', 'chunk_text', conn)
#     else:
#         logger.info("No new chunks to embed for this model.")

#     # --- Tier 2: Process Section Embeddings (CR-002) ---
#     logger.info("Processing Tier 2 (Section) embeddings...")
#     config_section = get_model_config(model_name, "section")

#     # Find new sections to embed. We use a placeholder check for now.
#     # A more robust system might add a vector_id to the sections table.
#     query_sections = """
#         SELECT s.section_id, s.section_summary, d.doc_name, s.page_num, s.section_header, s.raw_text
#         FROM sections s
#         JOIN documents d ON s.doc_id = d.doc_id
#         WHERE d.status = 'chunked_and_summarized' AND s.section_summary IS NOT NULL
#     """
#     sections_to_process = conn.execute(query_sections).fetchall()

#     if sections_to_process:
#         # Note: We are rebuilding the section index each time for simplicity.
#         # FR-24 logic could be extended here for granular section updates.
#         config_section['faiss_path'].unlink(missing_ok=True)
#         config_section['metadata_path'].unlink(missing_ok=True)
#         config_section['bm25_path'].unlink(missing_ok=True)
#         update_vector_store(sections_to_process, config_section, 'section_id', 'section_summary', 'raw_text', conn, is_chunk=False)
#     else:
#         logger.info("No new sections to embed for this model.")

#     conn.close()


def process_embeddings_for_model(model_name: str):
    logger.info(f"--- Processing model: {model_name} ---")

    # Efficiently load the model only ONCE.
    model = SentenceTransformer(model_name)
    conn = get_db_connection()

    vector_id_col = f"vector_id_{get_safe_model_name(model_name)}"

    # --- Process Tier 1 (Chunks) ---
    chunk_query = f"""
        SELECT c.chunk_id, c.summary, d.doc_name, s.page_num, s.section_header, c.chunk_text as raw_text
        FROM chunks c JOIN sections s ON c.section_id = s.section_id JOIN documents d ON s.doc_id = d.doc_id
        WHERE d.status = 'chunked_and_summarized' AND c.{vector_id_col} IS NULL
    """
    chunks_to_process = conn.execute(chunk_query).fetchall()
    create_indexes_for_tier(chunks_to_process, model, model_name, "chunk")

    # FR-15: Write vector IDs back to the main DB for linkage
    if chunks_to_process:
        chunk_ids = [row["chunk_id"] for row in chunks_to_process]
        update_data = [(chunk_id, chunk_id) for chunk_id in chunk_ids]
        conn.executemany(f"UPDATE chunks SET {vector_id_col} = ? WHERE chunk_id = ?", update_data)
        conn.commit()
        logger.info(f"Updated main DB with {len(chunk_ids)} vector IDs for model '{model_name}'.")

    # --- Process Tier 2 (Sections) ---
    section_query = """
        SELECT s.section_id, s.section_summary, d.doc_name, s.page_num, s.section_header, s.raw_text
        FROM sections s JOIN documents d ON s.doc_id = d.doc_id
        WHERE d.status = 'chunked_and_summarized' AND s.section_summary IS NOT NULL
    """
    sections_to_process = conn.execute(section_query).fetchall()
    create_indexes_for_tier(sections_to_process, model, model_name, "section")

    conn.close()


# def update_vector_store(items_to_process: list, config: dict, id_col: str, summary_col: str, text_col: str, main_db_conn, is_chunk: bool = True):
#     """Adds new items to existing FAISS, BM25, and metadata stores."""

#     ids, summaries, raw_texts = zip(*[(r[id_col], r[summary_col], r[text_col]) for r in items_to_process])

#     logger.info(f"Generating {len(summaries)} new embeddings...")
#     new_embeddings = config['model'].encode(summaries, show_progress_bar=True)
#     dimension = new_embeddings.shape[1]

#     # --- FAISS Index (FR-24: Additive Update) ---
#     if config['faiss_path'].exists():
#         logger.info(f"Loading existing FAISS index from {config['faiss_path']}")
#         index = faiss.read_index(str(config['faiss_path']))
#     else:
#         logger.info("Creating new FAISS index.")
#         index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))

#     index.add_with_ids(np.array(new_embeddings, dtype=np.float32), np.array(ids))
#     faiss.write_index(index, str(config['faiss_path']))
#     logger.info(f"FAISS index updated. Total vectors: {index.ntotal}")

#     # --- Metadata DB (FR-24: Additive Update) ---
#     is_new_meta_db = not config['metadata_path'].exists()
#     conn_meta = sqlite3.connect(config['metadata_path'])
#     if is_new_meta_db:
#         logger.info("Creating new metadata DB.")
#         conn_meta.execute(f"""
#             CREATE TABLE metadata (
#                 {id_col} INTEGER PRIMARY KEY, doc_name TEXT, page_num INTEGER,
#                 section_header TEXT, summary TEXT, raw_text TEXT
#             )
#         """)

#     meta_data_to_add = [
#         (r[id_col], r['doc_name'], r['page_num'], r['section_header'], r[summary_col], r[text_col])
#         for r in items_to_process
#     ]
#     conn_meta.executemany(f"INSERT OR REPLACE INTO metadata VALUES (?, ?, ?, ?, ?, ?)", meta_data_to_add)
#     conn_meta.commit()
#     conn_meta.close()
#     logger.info("Metadata DB updated.")

#     # --- BM25 Index (Rebuilt on each update for simplicity) ---
#     logger.info("Rebuilding BM25 index...")
#     all_meta_conn = sqlite3.connect(config['metadata_path'])
#     all_meta_conn.row_factory = sqlite3.Row
#     all_items = all_meta_conn.execute("SELECT summary, " + id_col + " FROM metadata").fetchall()

#     tokenized_corpus = [nltk.word_tokenize(doc['summary'].lower()) for doc in all_items]
#     bm25_index = BM25Okapi(tokenized_corpus)

#     # The doc map is crucial for BM25 to map its internal index to our chunk/section IDs
#     bm25_doc_map = {i: item[id_col] for i, item in enumerate(all_items)}

#     with open(config["bm25_path"], 'wb') as f:
#         pickle.dump({'index': bm25_index, 'doc_map': bm25_doc_map}, f)
#     logger.info(f"BM25 index rebuilt and saved with {len(all_items)} documents.")
#     all_meta_conn.close()

#     # --- Update Main DB (Only for chunks) ---
#     if is_chunk:
#         logger.info("Writing vector IDs back to the main project database...")
#         update_data = [(chunk_id, chunk_id) for chunk_id in ids]
#         main_db_conn.executemany(f"UPDATE chunks SET {config['vector_id_col']} = ? WHERE chunk_id = ?", update_data)
#         main_db_conn.commit()


def main():
    logger.info("--- Starting Embedding and Indexing Pipeline ---")
    config.EMBEDDINGS_DIR.mkdir(exist_ok=True)

    # CRITICAL FIX: Loop over the model list from the central config file.
    for model_name in config.EMBEDDING_MODELS_LIST:
        process_embeddings_for_model(model_name)

    # FR-16: Finalize document statuses
    logger.info("--- Finalizing document statuses ---")
    conn_final = get_db_connection()

    vector_id_cols = [f"vector_id_{get_safe_model_name(m)}" for m in config.EMBEDDING_MODELS_LIST]
    all_models_embedded_check = " AND ".join([f"c.{col} IS NOT NULL" for col in vector_id_cols])

    update_query = f"""
        UPDATE documents SET status = 'embedded' WHERE doc_id IN (
            SELECT d.doc_id FROM documents d
            JOIN sections s ON d.doc_id = s.doc_id
            JOIN chunks c ON s.section_id = c.section_id
            WHERE d.status = 'chunked_and_summarized'
            GROUP BY d.doc_id
            HAVING COUNT(c.chunk_id) = COUNT(CASE WHEN {all_models_embedded_check} THEN 1 END)
        )
    """
    cursor_final = conn_final.cursor()
    cursor_final.execute(update_query)
    updated_count = cursor_final.rowcount
    conn_final.commit()
    conn_final.close()

    if updated_count > 0:
        logger.info(f"Successfully updated status for {updated_count} document(s) to 'embedded'.")
    else:
        logger.info("No new documents were ready for final 'embedded' status update.")

    logger.info("--- All embedding and indexing processes are complete. ---")


if __name__ == "__main__":
    main()


# def main():
#     """Main function to generate and store embeddings for all models."""
#     for model_name in EMBEDDING_MODELS:
#         process_embeddings_for_model(model_name)

#     logger.info("--- Finalizing document statuses ---")
#     conn_final = get_db_connection()

#     # Construct the HAVING clause dynamically based on configured models
#     vector_id_cols = [get_model_config(m, "chunk")["vector_id_col"] for m in EMBEDDING_MODELS]
#     all_models_embedded_check = " AND ".join([f'c.{col} IS NOT NULL' for col in vector_id_cols])

#     # FR-16: Holistic Document Status Update
#     update_query = f"""
#         UPDATE documents
#         SET status = 'embedded'
#         WHERE doc_id IN (
#             SELECT d.doc_id
#             FROM documents d
#             JOIN sections s ON d.doc_id = s.doc_id
#             JOIN chunks c ON s.section_id = c.section_id
#             WHERE d.status = 'chunked_and_summarized'
#             GROUP BY d.doc_id
#             HAVING COUNT(c.chunk_id) = COUNT(CASE WHEN {all_models_embedded_check} THEN 1 END)
#         )
#     """
#     cursor_final = conn_final.cursor()
#     cursor_final.execute(update_query)
#     updated_count = cursor_final.rowcount
#     conn_final.commit()
#     conn_final.close()

#     if updated_count > 0:
#         logger.info(f"Successfully updated status for {updated_count} document(s) to 'embedded'.")
#     else:
#         logger.info("No new documents were ready for final 'embedded' status update.")

#     logger.info("All embedding and indexing processes are complete.")

# if __name__ == "__main__":
#     main()
