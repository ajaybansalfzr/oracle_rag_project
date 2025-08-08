import pickle
import sqlite3
import sys

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

# At the top of create_vector_store.py
from scripts.utils.vector_store_utils import cleanup_archived_vectors

# NEW: Initialize logger
logger = get_logger(__name__)

# NEW: Ensure NLTK punkt is available for BM25
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    logger.info("Downloading 'punkt' model for NLTK.")
    nltk.download("punkt", quiet=True)


def create_indexes_for_tier(data: list, model, model_name: str, tier: str):
    """
    Creates FAISS, BM25, and metadata DB for a given tier ('chunk' or 'section').
    Implements FR-14 (Dual Index) and FR-15 (Decoupled Metadata).
    """
    if not data:
        logger.info(f"No new items to process for tier '{tier}'. Skipping.")
        return

    id_col = f"{tier}_id"
    # summary_col = "summary" if tier == "chunk" else "section_summary"

    # ids = [row[id_col] for row in data]
    # summaries = [row[summary_col] for row in data]
    # Always embed the actual text, never the summary:
    embed_col = "chunk_text" if tier == "chunk" else "raw_text"
    embeddings_input = [
        row[embed_col] for row in data if row[embed_col] and isinstance(row[embed_col], str) and row[embed_col].strip()
    ]
    ids = [
        row[f"{tier}_id"]
        for row in data
        if row[embed_col] and isinstance(row[embed_col], str) and row[embed_col].strip()
    ]

    if not embeddings_input:
        logger.warning(f"No valid {embed_col} entries to process for tier '{tier}'. Skipping.")
        return

    # --- Artifact Path Generation (using safe names) ---
    safe_name = get_safe_model_name(model_name)
    faiss_path = config.EMBEDDINGS_DIR / f"{tier}_index_{safe_name}.faiss"
    meta_path = config.EMBEDDINGS_DIR / f"{tier}_metadata_{safe_name}.db"
    # bm25_path = config.EMBEDDINGS_DIR / f"{tier}_bm25_{safe_name}.pkl"

    # Per FSD, section indexes are rebuilt, chunk indexes are additive.
    # if tier == 'section':
    #     logger.warning(f"Rebuilding all '{tier}' artifacts for model '{model_name}' as per FSD strategy.")
    #     faiss_path.unlink(missing_ok=True)
    #     meta_path.unlink(missing_ok=True)
    #     # bm25_path.unlink(missing_ok=True)

    # --- 1. FAISS Index (Semantic) ---
    # logger.info(f"Generating {len(summaries)} embeddings for tier '{tier}'...")
    # embeddings = model.encode(summaries, show_progress_bar=True, batch_size=128)
    # dimension = embeddings.shape[1]
    logger.info(f"Generating {len(embeddings_input)} embeddings for tier '{tier}'...")
    embeddings = model.encode(embeddings_input, show_progress_bar=True, batch_size=128)
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

    # --- 2. Metadata DB (for fast lookups and two-way linkage) ---
    conn_meta = sqlite3.connect(meta_path)
    # CORRECTED SCHEMA: Define a single, robust schema that works for both tiers.
    # The 'chunk_id_link' column explicitly fulfills FR-15 for two-way linkage.
    conn_meta.execute(
        f"""CREATE TABLE IF NOT EXISTS metadata (
            {id_col} INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            chunk_id_link INTEGER,
            doc_name TEXT, 
            page_num INTEGER,
            section_header TEXT, 
            summary TEXT, 
            raw_text TEXT)"""
    )
    meta_data_to_add = []
    # raw_text_col = "raw_text"
    raw_text_col = "chunk_text" if tier == "chunk" else "raw_text"

    # raw_text_col = "chunk_text" if tier == "chunk" else "raw_text"
    # Both chunk and section queries return a column named 'raw_text'.

    for row in data:
        # If it's a chunk, its ID is the link. If it's a section, there's no chunk link.
        # doc_id_val = row['doc_id']  # ✅ new line
        chunk_link = row["chunk_id"] if tier == "chunk" else None
        # meta_data_to_add.append((
        #     row[id_col],
        #     row['doc_id']
        #     chunk_link,
        #     # doc_id_val,  # ✅ new line
        #     row['doc_name'],
        #     row['page_num'],
        #     row['section_header'],
        #     row[summary_col],
        #     row[raw_text_col]
        # ))
        meta_data_to_add.append(
            (
                row[id_col],
                row["doc_id"],
                chunk_link,
                row["doc_name"],
                row["page_num"],
                row["section_header"],
                "",  # row[summary_col],
                row[raw_text_col],
            )
        )

    conn_meta.executemany(
        "INSERT OR REPLACE INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        meta_data_to_add,
    )
    conn_meta.commit()
    conn_meta.close()
    logger.info(f"Metadata DB for tier '{tier}' updated.")

    # # --- 3. BM25 Index (Lexical) ---
    # all_meta_conn = sqlite3.connect(meta_path)
    # all_meta_conn.row_factory = sqlite3.Row
    # all_items = all_meta_conn.execute(f"SELECT summary, {id_col}, doc_id  FROM metadata").fetchall()
    # all_meta_conn.close()

    # tokenized_corpus = [nltk.word_tokenize(doc['summary'].lower()) for doc in all_items]
    # bm25_index = BM25Okapi(tokenized_corpus)

    # # The doc map is crucial for BM25 to map its internal index to our chunk/section IDs
    # bm25_doc_map = {i: {"id": row[id_col], "doc_id": row["doc_id"]} for i, item in enumerate(all_items)}

    # # THE CRITICAL CHANGE: Save the tokenized_corpus along with the index and map.
    # with open(bm25_path, 'wb') as f:
    #     pickle.dump({
    #         'index': bm25_index,
    #         'doc_map': bm25_doc_map,
    #         'corpus': tokenized_corpus  # Save the tokenized text
    #     }, f)
    # logger.info(f"BM25 index for tier '{tier}' rebuilt and saved with {len(all_items)} documents.")


def rebuild_bm25_for_tier(model_name: str, tier: str):
    safe_name = get_safe_model_name(model_name)
    meta_path = config.EMBEDDINGS_DIR / f"{tier}_metadata_{safe_name}.db"
    bm25_path = config.EMBEDDINGS_DIR / f"{tier}_bm25_{safe_name}.pkl"

    if not meta_path.exists():
        logger.warning(f"No metadata found for model '{model_name}' tier '{tier}'. Skipping BM25.")
        return

    logger.info(f"Rebuilding BM25 index for model: {model_name}, tier: {tier}")
    conn = sqlite3.connect(meta_path)
    conn.row_factory = sqlite3.Row
    id_col = f"{tier}_id"
    # text_col = "chunk_text" if tier == "chunk" else "raw_text"
    text_col = "raw_text"

    all_rows = conn.execute(f"SELECT {text_col}, {id_col}, doc_id FROM metadata").fetchall()
    tokenized_corpus = [nltk.word_tokenize((doc[text_col] or "").lower()) for doc in all_rows if doc[text_col]]

    # all_rows = conn.execute(f"SELECT summary, {id_col}, doc_id FROM metadata").fetchall()
    # conn.close()

    # # tokenized_corpus = [nltk.word_tokenize(doc['summary'].lower()) for doc in all_rows]
    # # bm25_index = BM25Okapi(tokenized_corpus)
    # # bm25_doc_map = {i: {"id": row[id_col], "doc_id": row["doc_id"]} for i, row in enumerate(all_rows)}

    # # logger.info(f"Saving BM25 index to {bm25_path} ...")
    # # with open(bm25_path, 'wb') as f:
    # #     pickle.dump({
    # #         'index': bm25_index,
    # #         'doc_map': bm25_doc_map,
    # #         'corpus': tokenized_corpus
    # #     }, f)
    # # logger.info(f"BM25 index for {tier} rebuilt. Entries: {len(all_rows)}")
    # tokenized_corpus = [nltk.word_tokenize((doc['summary'] or '').lower()) for doc in all_rows if doc['summary']]
    if not tokenized_corpus:
        logger.warning(f"BM25 index for {tier} not created for model '{model_name}' — no data found.")
        return

    bm25_index = BM25Okapi(tokenized_corpus)
    bm25_doc_map = {i: {"id": row[id_col], "doc_id": row["doc_id"]} for i, row in enumerate(all_rows)}

    logger.info(f"Saving BM25 index to {bm25_path} ...")
    with open(bm25_path, "wb") as f:
        pickle.dump(
            {"index": bm25_index, "doc_map": bm25_doc_map, "corpus": tokenized_corpus},
            f,
        )
    logger.info(f"BM25 index for {tier} rebuilt. Entries: {len(all_rows)}")


# def process_embeddings_for_model(model_name: str):
#     logger.info(f"--- Processing model: {model_name} ---")

#     # Efficiently load the model only ONCE.
#     model = SentenceTransformer(model_name)
#     conn = get_db_connection()

#     vector_id_col = f"vector_id_{get_safe_model_name(model_name)}"

#     # --- Process Tier 1 (Chunks) ---
#     chunk_query = f"""
#         SELECT c.chunk_id, c.summary, d.doc_id, d.doc_name, s.page_num, s.section_header, c.chunk_text as raw_text
#         FROM chunks c JOIN sections s ON c.section_id = s.section_id JOIN documents d ON s.doc_id = d.doc_id
#         WHERE d.processing_status = 'chunked_and_summarized' AND d.lifecycle_status = 'active' AND c.{vector_id_col} IS NULL
#     """
#     chunks_to_process = conn.execute(chunk_query).fetchall()
#     create_indexes_for_tier(chunks_to_process, model, model_name, 'chunk')

#     # FR-15: Write vector IDs back to the main DB for linkage
#     if chunks_to_process:
#         chunk_ids = [row['chunk_id'] for row in chunks_to_process]
#         update_data = [(chunk_id, chunk_id) for chunk_id in chunk_ids]
#         conn.executemany(f"UPDATE chunks SET {vector_id_col} = ? WHERE chunk_id = ?", update_data)
#         conn.commit()
#         logger.info(f"Updated main DB with {len(chunk_ids)} vector IDs for model '{model_name}'.")

#     # --- Process Tier 2 (Sections) ---
#     section_query = """
#         SELECT s.section_id, s.section_summary, d.doc_id, d.doc_name, s.page_num, s.section_header, s.raw_text
#         FROM sections s JOIN documents d ON s.doc_id = d.doc_id
#         WHERE d.processing_status = 'chunked_and_summarized' AND d.lifecycle_status = 'active' AND s.section_summary IS NOT NULL
#     """
#     sections_to_process = conn.execute(section_query).fetchall()
#     create_indexes_for_tier(sections_to_process, model, model_name, 'section')

#     conn.close()

# In create_vector_store.py
# --- REPLACE the entire process_embeddings_for_model function ---

# def process_embeddings_for_model(model_name: str):
#     logger.info(f"--- Processing model: {model_name} ---")

#     model = SentenceTransformer(model_name)
#     conn = get_db_connection()
#     vector_id_col = f"vector_id_{get_safe_model_name(model_name)}"

#     # # --- Identify the specific documents we will be working on in this run ---
#     # docs_to_update_query = """
#     #     SELECT DISTINCT d.doc_id
#     #     FROM documents d
#     #     JOIN sections s ON d.doc_id = s.doc_id
#     #     LEFT JOIN chunks c ON s.section_id = c.section_id
#     #     WHERE d.processing_status = 'chunked_and_summarized' AND d.lifecycle_status = 'active'
#     # """
#     # Get the NAME and ID of all active documents to be processed. We need both.
#     docs_to_process_query = """
#         SELECT doc_id, doc_name FROM documents
#         WHERE processing_status = 'chunked_and_summarized' AND lifecycle_status = 'active'
#     """
#     docs_to_process = conn.execute(docs_to_process_query).fetchall()
#     # docs_to_update_ids = [row['doc_id'] for row in conn.execute(docs_to_update_query).fetchall()]
#     if not docs_to_process:
#         logger.info(f"No new documents to embed for model '{model_name}'. Skipping.")
#         conn.close()
#         return

#     doc_ids_to_process = [doc['doc_id'] for doc in docs_to_process]
#     doc_names_to_process = [doc['doc_name'] for doc in docs_to_process]
#     logger.info(f"Found {len(doc_ids_to_process)} documents to process for model '{model_name}'.")

#     # if not docs_to_update_ids:
#     #     logger.info(f"No new documents to embed for model '{model_name}'. Skipping.")
#     #     conn.close()
#     #     return

#     # logger.info(f"Found {len(docs_to_update_ids)} documents to process for model '{model_name}'.")

#     # --- GRANULAR UPDATE LOGIC ---
#     # Step 1: Cleanup old section vectors for these specific documents.
#     # This handles the case where a document was updated. We remove the old sections
#     # before adding the new ones. The remove_document_vectors function from your
#     # utils can be adapted for this, or we can write a specific one. Let's adapt.
#     # Note: We will create a more targeted version of this.
#     # cleanup_old_section_vectors(docs_to_update_ids, model_name)
#     cleanup_archived_vectors(doc_names_to_process, model_name, 'chunk')
#     cleanup_archived_vectors(doc_names_to_process, model_name, 'section')


#     # --- Step 2: Process TIER 1 (Chunks) - This logic remains additive and is already correct ---
#     chunk_query = f"""
#         SELECT c.chunk_id, c.summary, d.doc_id, d.doc_name, s.page_num, s.section_header, c.chunk_text as raw_text
#         FROM chunks c JOIN sections s ON c.section_id = s.section_id JOIN documents d ON s.doc_id = d.doc_id
#         WHERE d.processing_status = 'chunked_and_summarized' AND d.lifecycle_status = 'active' AND c.{vector_id_col} IS NULL
#     """
#     chunks_to_process = conn.execute(chunk_query).fetchall()
#     create_indexes_for_tier(chunks_to_process, model, model_name, 'chunk') # 'chunk' is already additive

#     if chunks_to_process:
#         chunk_ids = [row['chunk_id'] for row in chunks_to_process]
#         update_data = [(chunk_id, chunk_id) for chunk_id in chunk_ids]
#         conn.executemany(f"UPDATE chunks SET {vector_id_col} = ? WHERE chunk_id = ?", update_data)
#         conn.commit()
#         logger.info(f"Updated main DB with {len(chunk_ids)} vector IDs for model '{model_name}'.")


#     # --- Step 3: Process TIER 2 (Sections) - Additive Processing ---
#     # This query now ONLY fetches sections for the documents we are currently processing.
#     section_query = f"""
#         SELECT s.section_id, s.section_summary, d.doc_id, d.doc_name, s.page_num, s.section_header, s.raw_text
#         FROM sections s JOIN documents d ON s.doc_id = d.doc_id
#         WHERE d.doc_id IN ({','.join(['?'] * len(docs_to_update_ids))})
#         AND s.section_summary IS NOT NULL
#     """
#     sections_to_process = conn.execute(section_query, docs_to_update_ids).fetchall()
#     create_indexes_for_tier(sections_to_process, model, model_name, 'section') # 'section' will now be additive

#     conn.close()


def process_embeddings_for_model(model_name: str):
    logger.info(f"--- Processing model: {model_name} ---")

    model = SentenceTransformer(model_name)
    conn = get_db_connection()
    vector_id_col = f"vector_id_{get_safe_model_name(model_name)}"

    # Get the NAME and ID of all active documents to be processed. We need both.
    docs_to_process_query = """
        SELECT doc_id, doc_name FROM documents
        WHERE processing_status = 'chunked_and_summarized' AND lifecycle_status = 'active'
    """
    docs_to_process = conn.execute(docs_to_process_query).fetchall()

    if not docs_to_process:
        logger.info(f"No new documents to embed for model '{model_name}'. Skipping.")
        conn.close()
        return

    doc_ids_to_process = [doc["doc_id"] for doc in docs_to_process]
    doc_names_to_process = [doc["doc_name"] for doc in docs_to_process]
    logger.info(f"Found {len(doc_ids_to_process)} documents to process for model '{model_name}'.")

    # --- CORRECTED GRANULAR UPDATE LOGIC ---
    # Step 1: Clean up vectors from any PRIOR, ARCHIVED versions of these documents.
    # We clean up both chunk and section vectors to be thorough.
    cleanup_archived_vectors(doc_names_to_process, model_name, "chunk")
    cleanup_archived_vectors(doc_names_to_process, model_name, "section")

    # Step 2: Add new CHUNK vectors for the CURRENT, ACTIVE documents.
    chunk_query = f"""
        SELECT c.chunk_id, c.summary, d.doc_id, d.doc_name, s.page_num, s.section_header, c.chunk_text
        FROM chunks c JOIN sections s ON c.section_id = s.section_id JOIN documents d ON s.doc_id = d.doc_id
        WHERE d.doc_id IN ({','.join(['?'] * len(doc_ids_to_process))}) AND c.{vector_id_col} IS NULL
    """
    chunks_to_process = conn.execute(chunk_query, doc_ids_to_process).fetchall()
    create_indexes_for_tier(chunks_to_process, model, model_name, "chunk")

    if chunks_to_process:
        chunk_ids = [row["chunk_id"] for row in chunks_to_process]
        update_data = [(chunk_id, chunk_id) for chunk_id in chunk_ids]
        conn.executemany(f"UPDATE chunks SET {vector_id_col} = ? WHERE chunk_id = ?", update_data)
        conn.commit()
        logger.info(f"Updated main DB with {len(chunk_ids)} vector IDs for model '{model_name}'.")

    # Step 3: Add new SECTION vectors for the CURRENT, ACTIVE documents.
    section_query = f"""
        SELECT s.section_id, s.section_summary, d.doc_id, d.doc_name, s.page_num, s.section_header, s.raw_text
        FROM sections s JOIN documents d ON s.doc_id = d.doc_id
        WHERE d.doc_id IN ({','.join(['?'] * len(doc_ids_to_process))})
    """
    sections_to_process = conn.execute(section_query, doc_ids_to_process).fetchall()
    create_indexes_for_tier(sections_to_process, model, model_name, "section")

    conn.close()


def main():
    logger.info("--- Starting Embedding and Indexing Pipeline ---")
    config.EMBEDDINGS_DIR.mkdir(exist_ok=True)

    # CRITICAL FIX: Loop over the model list from the central config file.
    for model_name in config.EMBEDDING_MODELS_LIST:
        process_embeddings_for_model(model_name)

        # ✅ Step 2: BM25 (Chunk + Section)
        rebuild_bm25_for_tier(model_name, "chunk")
        rebuild_bm25_for_tier(model_name, "section")

    # FR-16: Finalize document statuses
    logger.info("--- Finalizing document statuses ---")
    conn_final = get_db_connection()

    vector_id_cols = [f"vector_id_{get_safe_model_name(m)}" for m in config.EMBEDDING_MODELS_LIST]
    all_models_embedded_check = " AND ".join([f"c.{col} IS NOT NULL" for col in vector_id_cols])

    update_query = f"""
        UPDATE documents SET processing_status = 'embedded' WHERE doc_id IN (
            SELECT d.doc_id FROM documents d
            JOIN sections s ON d.doc_id = s.doc_id
            JOIN chunks c ON s.section_id = c.section_id
            WHERE d.processing_status = 'chunked_and_summarized' AND d.lifecycle_status = 'active'
            GROUP BY d.doc_id
            HAVING COUNT(c.chunk_id) > 0 AND COUNT(c.chunk_id) = COUNT(CASE WHEN {all_models_embedded_check} THEN 1 END)
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


if len(sys.argv) > 1 and sys.argv[1].endswith(".csv"):
    from scripts.extractor_for_pdf import evaluate_against_golden

    evaluate_against_golden(sys.argv[1], get_db_connection())


if __name__ == "__main__":
    main()
