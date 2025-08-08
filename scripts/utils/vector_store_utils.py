# scripts/utils/vector_store_utils.py

# import pickle
import sqlite3

import faiss
import numpy as np

from .. import config
from .utils import get_logger, get_safe_model_name

logger = get_logger(__name__)
# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
# EMBEDDING_MODELS = ['all-MiniLM-L6-v2', 'BAAI/bge-small-en-v1.5'] # Should be moved to config later


def _get_artifact_paths(model_name: str, index_type: str) -> dict:
    """Internal helper to get artifact paths using the robust safe name."""
    # Use the new, robust safe name generator
    safe_name = get_safe_model_name(model_name)
    # Use the EMBEDDINGS_DIR path from the central config file
    faiss_path = config.EMBEDDINGS_DIR / f"{index_type}_index_{safe_name}.faiss"
    meta_path = config.EMBEDDINGS_DIR / f"{index_type}_metadata_{safe_name}.db"
    bm25_path = config.EMBEDDINGS_DIR / f"{index_type}_bm25_{safe_name}.pkl"
    return {"faiss_path": faiss_path, "metadata_path": meta_path, "bm25": bm25_path}


# In scripts/utils/vector_store_utils.py, add this new function

# def cleanup_old_section_vectors(doc_ids: list, model_name: str):
#     """
#     Removes all section vectors and metadata for a specific list of document IDs
#     and a specific model. This is used to clean up old data before adding updated
#     sections for a document. This is NOT the same as permanent archival.
#     """
#     if not doc_ids:
#         return

#     from .database_utils import get_db_connection
#     logger.warning(f"Cleaning up old section vectors for {len(doc_ids)} documents for model '{model_name}'.")

#     conn = get_db_connection()
#     id_placeholders = ','.join(['?'] * len(doc_ids))

#     # Find all 'section_id's associated with the documents being updated.
#     # This includes sections from both 'active' and 'archived' versions of the document name,
#     # ensuring we clean up everything related to these docs.
#     section_ids_to_remove = [
#         row['section_id'] for row in conn.execute(
#             f"SELECT section_id FROM sections WHERE doc_id IN ({id_placeholders})", doc_ids
#         ).fetchall()
#     ]
#     conn.close()

#     if not section_ids_to_remove:
#         logger.info("No pre-existing sections found for these documents. No cleanup needed.")
#         return

#     artifacts = _get_artifact_paths(model_name, "section")

#     # --- Perform Cleanup ---
#     # 1. Remove from metadata DB first for safety
#     if artifacts["metadata_path"].exists():
#         with sqlite3.connect(artifacts["metadata_path"]) as conn_meta:
#             id_placeholders = ','.join(['?'] * len(section_ids_to_remove))
#             deleted_count = conn_meta.execute(
#                 f"DELETE FROM metadata WHERE section_id IN ({id_placeholders})",
#                 section_ids_to_remove
#             ).rowcount
#             conn_meta.commit()
#             if deleted_count > 0:
#                 logger.info(f"Removed {deleted_count} old section metadata entries.")

#     # 2. Remove from FAISS index
#     if artifacts["faiss_path"].exists():
#         index = faiss.read_index(str(artifacts["faiss_path"]))
#         num_removed = index.remove_ids(np.array(section_ids_to_remove, dtype=np.int64))
#         if num_removed > 0:
#             faiss.write_index(index, str(artifacts["faiss_path"]))
#             logger.info(f"Removed {num_removed} old section vectors from FAISS index.")

# In scripts/utils/vector_store_utils.py
# --- REPLACE the entire cleanup_old_section_vectors function ---

# def cleanup_old_section_vectors(doc_ids_to_clean: list, model_name: str):
#     """
#     Removes all section vectors and metadata for a specific list of document IDs
#     and a specific model by operating ONLY on the model's metadata DB.
#     This is highly efficient as it avoids querying the main project database.
#     """
#     if not doc_ids_to_clean:
#         return

#     logger.warning(f"Cleaning up old section vectors for {len(doc_ids_to_clean)} documents for model '{model_name}'.")

#     artifacts = _get_artifact_paths(model_name, "section")
#     meta_path = artifacts["metadata_path"]
#     faiss_path = artifacts["faiss_path"]

#     if not meta_path.exists():
#         logger.info("No metadata DB found. No cleanup needed.")
#         return

#     # --- Perform Cleanup using only the Metadata DB ---
#     with sqlite3.connect(meta_path) as conn_meta:
#         id_placeholders = ','.join(['?'] * len(doc_ids_to_clean))

#         # 1. Find the specific vector IDs (section_ids) to remove.
#         cursor = conn_meta.cursor()
#         cursor.execute(f"SELECT section_id FROM metadata WHERE doc_id IN ({id_placeholders})", doc_ids_to_clean)
#         section_ids_to_remove = [row[0] for row in cursor.fetchall()]

#         if not section_ids_to_remove:
#             logger.info("No pre-existing sections found for these documents in the metadata. No cleanup needed.")
#             return

#         # 2. Delete the corresponding rows from the metadata table.
#         deleted_count = conn_meta.execute(
#             f"DELETE FROM metadata WHERE section_id IN ({','.join(['?'] * len(section_ids_to_remove))})",
#             section_ids_to_remove
#         ).rowcount
#         conn_meta.commit()
#         if deleted_count > 0:
#             logger.info(f"Removed {deleted_count} old section metadata entries.")

#     # 3. Remove the identified vector IDs from the FAISS index.
#     if faiss_path.exists() and section_ids_to_remove:
#         index = faiss.read_index(str(faiss_path))
#         num_removed = index.remove_ids(np.array(section_ids_to_remove, dtype=np.int64))
#         if num_removed > 0:
#             faiss.write_index(index, str(faiss_path))
#             logger.info(f"Removed {num_removed} old section vectors from FAISS index.")


def cleanup_archived_vectors(doc_names_to_update: list, model_name: str, tier: str):
    """
    Finds all ARCHIVED documents matching a list of document names and removes
    their corresponding vectors and metadata for a specific model and tier.
    This is the definitive, corrected cleanup logic.
    """
    if not doc_names_to_update:
        return

    from .database_utils import get_db_connection

    logger.warning(
        f"Cleaning up {tier} vectors for {len(doc_names_to_update)} archived document names for model '{model_name}'."
    )

    conn = get_db_connection()
    name_placeholders = ",".join(["?"] * len(doc_names_to_update))

    # Find the doc_ids of all ARCHIVED documents that match the names of the docs we are updating.
    archived_doc_ids_query = f"""
        SELECT doc_id FROM documents
        WHERE doc_name IN ({name_placeholders}) AND lifecycle_status = 'archived'
    """
    archived_doc_ids = [row["doc_id"] for row in conn.execute(archived_doc_ids_query, doc_names_to_update).fetchall()]

    if not archived_doc_ids:
        logger.info("No archived document versions found for the current batch. No vector cleanup needed.")
        conn.close()
        return

    # Now, find all vector IDs (chunk_id or section_id) that belong to these archived documents.
    id_col = f"{tier}_id"
    table = f"{tier}s"  # sections or chunks
    id_placeholders = ",".join(["?"] * len(archived_doc_ids))

    vector_ids_to_remove_query = f"SELECT {id_col} FROM {table} WHERE doc_id IN ({id_placeholders})"
    vector_ids_to_remove = [
        row[id_col] for row in conn.execute(vector_ids_to_remove_query, archived_doc_ids).fetchall()
    ]
    conn.close()

    if not vector_ids_to_remove:
        logger.info(f"Found archived documents but they have no associated '{tier}' vectors to clean up.")
        return

    # --- Perform the actual file cleanup ---
    artifacts = _get_artifact_paths(model_name, tier)
    meta_path = artifacts["metadata_path"]
    faiss_path = artifacts["faiss_path"]
    metadata_id_col = "chunk_id_link" if tier == "chunk" else "section_id"

    # 1. Remove from metadata DB
    if meta_path.exists():
        with sqlite3.connect(meta_path) as conn_meta:
            id_placeholders = ",".join(["?"] * len(vector_ids_to_remove))
            deleted_count = conn_meta.execute(
                f"DELETE FROM metadata WHERE {metadata_id_col} IN ({id_placeholders})",
                vector_ids_to_remove,
            ).rowcount
            if deleted_count > 0:
                logger.info(f"Removed {deleted_count} old {tier} metadata entries.")

    # 2. Remove from FAISS index
    if faiss_path.exists():
        index = faiss.read_index(str(faiss_path))
        num_removed = index.remove_ids(np.array(vector_ids_to_remove, dtype=np.int64))
        if num_removed > 0:
            faiss.write_index(index, str(faiss_path))
            logger.info(f"Removed {num_removed} old {tier} vectors from FAISS index.")


# def remove_document_vectors(doc_id: int):
#     """
#     Finds all chunk and section IDs for a given doc_id and removes their
#     corresponding vectors from all artifact files (FAISS indexes and metadata DBs).
#     This is the definitive implementation of the FR-24 cleanup requirement.
#     """
#     from .database_utils import get_db_connection
#     logger.warning(f"Executing comprehensive vector deletion for doc_id: {doc_id}")

#     main_conn = get_db_connection()
#     # Get IDs for BOTH tiers that need to be deleted.
#     ids_to_delete = main_conn.execute(
#         """
#         SELECT s.section_id, c.chunk_id FROM sections s
#         LEFT JOIN chunks c ON s.section_id = c.section_id
#         WHERE s.doc_id = ?
#         """, (doc_id,)
#     ).fetchall()
#     main_conn.close()

#     if not ids_to_delete:
#         logger.info(f"No existing sections or chunks found for doc_id {doc_id}. No vectors to delete.")
#         return

#     section_ids = list(set(row['section_id'] for row in ids_to_delete if row['section_id'] is not None))
#     chunk_ids = list(set(row['chunk_id'] for row in ids_to_delete if row['chunk_id'] is not None))

#     for model_name in config.EMBEDDING_MODELS_LIST:
#         logger.info(f"--- Cleaning artifacts for model: {model_name} ---")

#         # --- Tier 1 (Chunk) Cleanup ---
#         if chunk_ids:
#             chunk_artifacts = _get_artifact_paths(model_name, "chunk")
#             # 1a. Delete from FAISS index
#             if chunk_artifacts["faiss_path"].exists():
#                 index = faiss.read_index(str(chunk_artifacts["faiss_path"]))
#                 index.remove_ids(np.array(chunk_ids, dtype=np.int64))
#                 faiss.write_index(index, str(chunk_artifacts["faiss_path"]))
#                 logger.info(f"Removed {len(chunk_ids)} chunk vectors for '{model_name}'.")
#             # 1b. Delete from Metadata DB
#             if chunk_artifacts["metadata_path"].exists():
#                 meta_conn = sqlite3.connect(chunk_artifacts["metadata_path"])
#                 meta_conn.execute(f"DELETE FROM metadata WHERE chunk_id_link IN ({','.join('?' for _ in chunk_ids)})", chunk_ids)
#                 meta_conn.commit()
#                 meta_conn.close()
#                 logger.info(f"Removed {len(chunk_ids)} chunk metadata entries for '{model_name}'.")

#         # --- Tier 2 (Section) Cleanup ---
#         if section_ids:
#             section_artifacts = _get_artifact_paths(model_name, "section")
#             # 2a. Delete from FAISS index
#             if section_artifacts["faiss_path"].exists():
#                 index = faiss.read_index(str(section_artifacts["faiss_path"]))
#                 index.remove_ids(np.array(section_ids, dtype=np.int64))
#                 faiss.write_index(index, str(section_artifacts["faiss_path"]))
#                 logger.info(f"Removed {len(section_ids)} section vectors for '{model_name}'.")
#             # 2b. Delete from Metadata DB
#             if section_artifacts["metadata_path"].exists():
#                 meta_conn = sqlite3.connect(section_artifacts["metadata_path"])
#                 meta_conn.execute(f"DELETE FROM metadata WHERE section_id IN ({','.join('?' for _ in section_ids)})", section_ids)
#                 meta_conn.commit()
#                 meta_conn.close()
#                 logger.info(f"Removed {len(section_ids)} section metadata entries for '{model_name}'.")

#     logger.info(f"Vector deletion process completed for doc_id: {doc_id}")


def _archive_and_delete_metadata(conn: sqlite3.Connection, table_name: str, id_column: str, ids_to_process: list):
    """
    Archives rows from a metadata table to a corresponding '_archived' table
    before deleting them from the active table. This is an atomic operation.
    """
    if not ids_to_process:
        return 0

    archived_table_name = f"{table_name}_archived"
    cursor = conn.cursor()

    # Ensure the archive table exists
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {archived_table_name} AS SELECT * FROM {table_name} WHERE 0=1;")
    # Add an archive timestamp for audit purposes
    try:
        cursor.execute(f"ALTER TABLE {archived_table_name} ADD COLUMN archived_at TEXT;")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # 1. Copy data to the archive table
    id_placeholders = ",".join(["?"] * len(ids_to_process))
    cursor.execute(
        f"""
        INSERT INTO {archived_table_name}
        SELECT *, datetime('now') FROM {table_name}
        WHERE {id_column} IN ({id_placeholders})
    """,
        ids_to_process,
    )

    # 2. Delete data from the active table
    cursor.execute(
        f"DELETE FROM {table_name} WHERE {id_column} IN ({id_placeholders})",
        ids_to_process,
    )

    deleted_count = cursor.rowcount
    logger.info(f"Archived and deleted {deleted_count} rows from '{table_name}' metadata.")
    return deleted_count


def remove_document_vectors(doc_id: int):
    """
    Permanently removes all vector artifacts for a document.
    Implements a robust "archive-then-delete" strategy for all metadata before
    removing vectors from the FAISS index itself. This is a destructive operation
    intended for permanent data removal, not for routine updates.
    """
    from .database_utils import get_db_connection

    logger.warning(f"🚨 EXECUTING PERMANENT DELETION for all vector artifacts related to doc_id: {doc_id}")

    conn = get_db_connection()
    ids_to_delete = conn.execute(
        """
        SELECT s.section_id, c.chunk_id FROM sections s
        LEFT JOIN chunks c ON s.section_id = c.section_id
        WHERE s.doc_id = ?
    """,
        (doc_id,),
    ).fetchall()
    conn.close()

    if not ids_to_delete:
        logger.info(f"No sections or chunks found for doc_id: {doc_id}. No vector artifacts to remove.")
        return

    section_ids = sorted(list(set(row["section_id"] for row in ids_to_delete if row["section_id"] is not None)))
    chunk_ids = sorted(list(set(row["chunk_id"] for row in ids_to_delete if row["chunk_id"] is not None)))

    for model_name in config.EMBEDDING_MODELS_LIST:
        logger.info(f"🧹 Processing permanent deletion for model: {model_name}")

        # --- Process each tier (Chunks and Sections) ---
        for tier, ids, id_col_name in [
            ("chunk", chunk_ids, "chunk_id_link"),
            ("section", section_ids, "section_id"),
        ]:
            if not ids:
                continue

            artifacts = _get_artifact_paths(model_name, tier)
            try:
                # 1. Archive and delete from the metadata database first.
                if artifacts["metadata_path"].exists():
                    with sqlite3.connect(artifacts["metadata_path"]) as conn_meta:
                        _archive_and_delete_metadata(conn_meta, "metadata", id_col_name, ids)
                        conn_meta.commit()

                # 2. Only after successful metadata handling, remove from FAISS.
                if artifacts["faiss_path"].exists():
                    index = faiss.read_index(str(artifacts["faiss_path"]))
                    num_removed = index.remove_ids(np.array(ids, dtype=np.int64))
                    if num_removed > 0:
                        faiss.write_index(index, str(artifacts["faiss_path"]))
                        logger.info(
                            f"✅ Permanently removed {num_removed} {tier} vectors from FAISS index for model '{model_name}'."
                        )

            except Exception as e:
                logger.critical(
                    f"❌ FATAL ERROR during {tier.upper()} artifact deletion for model '{model_name}'. "
                    f"The system may be in an inconsistent state. Manual inspection required. Error: {e}",
                    exc_info=True,
                )

    logger.info(f"✅ Permanent vector artifact deletion process completed for doc_id: {doc_id}")


# def remove_document_vectors(doc_id: int):
#     """
#     Safely removes all vector entries (FAISS + metadata) associated with a document ID.
#     Handles both CHUNK and SECTION tiers for all configured embedding models.
#     """
#     from .database_utils import get_db_connection
#     logger.warning(f"🚨 Starting vector deletion for doc_id: {doc_id}")

#     conn = get_db_connection()
#     ids_to_delete = conn.execute("""
#         SELECT s.section_id, c.chunk_id FROM sections s
#         LEFT JOIN chunks c ON s.section_id = c.section_id
#         WHERE s.doc_id = ?
#     """, (doc_id,)).fetchall()
#     conn.close()

#     if not ids_to_delete:
#         logger.info(f"No sections or chunks found for doc_id: {doc_id}. Nothing to delete.")
#         return

#     section_ids = sorted(set(row['section_id'] for row in ids_to_delete if row['section_id']))
#     chunk_ids = sorted(set(row['chunk_id'] for row in ids_to_delete if row['chunk_id']))

#     for model_name in config.EMBEDDING_MODELS_LIST:
#         logger.info(f" Cleaning model: {model_name}")

#         # ---------------- CHUNKS ----------------
#         # if chunk_ids:
#         #     artifacts = _get_artifact_paths(model_name, "chunk")
#         #     try:
#         #         if artifacts["faiss_path"].exists():
#         #             index = faiss.read_index(str(artifacts["faiss_path"]))
#         #             index.remove_ids(np.array(chunk_ids, dtype=np.int64))
#         #             faiss.write_index(index, str(artifacts["faiss_path"]))
#         #             logger.info(f"✅ Removed {len(chunk_ids)} chunk vectors from FAISS")
#         #     except Exception as e:
#         #         logger.error(f"❌ Failed chunk FAISS cleanup: {e}")

#         #     if artifacts["metadata_path"].exists():
#         #         with sqlite3.connect(artifacts["metadata_path"]) as conn_meta:
#         #             deleted = conn_meta.execute(
#         #                 f"DELETE FROM metadata WHERE chunk_id_link IN ({','.join(['?'] * len(chunk_ids))})",
#         #                 chunk_ids
#         #             ).rowcount
#         #             conn_meta.commit()
#         #             logger.info(f"✅ Removed {deleted} chunk metadata rows")
#         if chunk_ids:
#             artifacts = _get_artifact_paths(model_name, "chunk")
#             try:
#                 # Step 1: Delete from the metadata database first.
#                 if artifacts["metadata_path"].exists():
#                     with sqlite3.connect(artifacts["metadata_path"]) as conn_meta:
#                         deleted_count = conn_meta.execute(
#                             f"DELETE FROM metadata WHERE chunk_id_link IN ({','.join(['?'] * len(chunk_ids))})",
#                             chunk_ids
#                         ).rowcount
#                         conn_meta.commit()
#                         logger.info(f"✅ Removed {deleted_count} chunk metadata rows for model '{model_name}'.")

#                 # Step 2: Only after metadata is gone, delete from the FAISS index.
#                 if artifacts["faiss_path"].exists():
#                     index = faiss.read_index(str(artifacts["faiss_path"]))
#                     num_removed = index.remove_ids(np.array(chunk_ids, dtype=np.int64))
#                     if num_removed > 0:
#                         faiss.write_index(index, str(artifacts["faiss_path"]))
#                         logger.info(f"✅ Removed {num_removed} chunk vectors from FAISS for model '{model_name}'.")
#                     else:
#                         logger.warning(f"⚠️ Vector removal from FAISS index for model '{model_name}' completed, but no matching chunk IDs were found to remove.")

#             except Exception as e:
#                 logger.error(
#                     f"❌ An error occurred during CHUNK artifact cleanup for model '{model_name}'. "
#                     f"Manual inspection of artifacts may be required. Error: {e}", exc_info=True
#                 )

#         # ---------------- SECTIONS ----------------
#         # if section_ids:
#         #     artifacts = _get_artifact_paths(model_name, "section")
#         #     try:
#         #         if artifacts["faiss_path"].exists():
#         #             index = faiss.read_index(str(artifacts["faiss_path"]))
#         #             index.remove_ids(np.array(section_ids, dtype=np.int64))
#         #             faiss.write_index(index, str(artifacts["faiss_path"]))
#         #             logger.info(f"✅ Removed {len(section_ids)} section vectors from FAISS")
#         #     except Exception as e:
#         #         logger.error(f"❌ Failed section FAISS cleanup: {e}")

#         #     if artifacts["metadata_path"].exists():
#         #         with sqlite3.connect(artifacts["metadata_path"]) as conn_meta:
#         #             deleted = conn_meta.execute(
#         #                 f"DELETE FROM metadata WHERE section_id IN ({','.join(['?'] * len(section_ids))})",
#         #                 section_ids
#         #             ).rowcount
#         #             conn_meta.commit()
#         #             logger.info(f"✅ Removed {deleted} section metadata rows")
#         if section_ids:
#             artifacts = _get_artifact_paths(model_name, "section")
#             try:
#                 # Step 1: Delete from the metadata database first.
#                 if artifacts["metadata_path"].exists():
#                     with sqlite3.connect(artifacts["metadata_path"]) as conn_meta:
#                         deleted_count = conn_meta.execute(
#                             f"DELETE FROM metadata WHERE section_id IN ({','.join(['?'] * len(section_ids))})",
#                             section_ids
#                         ).rowcount
#                         conn_meta.commit()
#                         logger.info(f"✅ Removed {deleted_count} section metadata rows for model '{model_name}'.")

#                 # Step 2: Only after metadata is gone, delete from the FAISS index.
#                 if artifacts["faiss_path"].exists():
#                     index = faiss.read_index(str(artifacts["faiss_path"]))
#                     num_removed = index.remove_ids(np.array(section_ids, dtype=np.int64))
#                     if num_removed > 0:
#                         faiss.write_index(index, str(artifacts["faiss_path"]))
#                         logger.info(f"✅ Removed {num_removed} section vectors from FAISS for model '{model_name}'.")
#                     else:
#                         logger.warning(f"⚠️ Vector removal from FAISS index for model '{model_name}' completed, but no matching section IDs were found to remove.")

#             except Exception as e:
#                 logger.error(
#                     f"❌ An error occurred during SECTION artifact cleanup for model '{model_name}'. "
#                     f"Manual inspection of artifacts may be required. Error: {e}", exc_info=True
#                 )

#     logger.info(f"✅ Vector deletion completed for doc_id: {doc_id}")
