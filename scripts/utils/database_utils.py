# scripts/helpers/database_helper.py

import hashlib
import sqlite3
from pathlib import Path

from .. import config
from .utils import get_logger, get_safe_model_name

# # --- Configuration ---
# PROJECT_ROOT = Path(__file__).resolve().parents[2] # Go up three levels from utils/
# OUTPUT_DIR = PROJECT_ROOT / "output"
# DB_PATH = OUTPUT_DIR / "project_data.db"
# OUTPUT_DIR.mkdir(exist_ok=True)

# NEW: Initialize the logger for this module
logger = get_logger(__name__)


# --- Schema and Connection ---
def initialize_database():
    """Initializes the SQLite database and creates the complete schema as per FSD v1.9.
    This function is idempotent and safe to run multiple times."""
    # conn = sqlite3.connect(config.DB_PATH)
    # with sqlite3.connect(config.DB_PATH) as conn:
    # # ✅ Enable foreign key support
    #     conn.execute("PRAGMA foreign_keys = ON;")
    #     cursor = conn.cursor()
    try:
        # The 'with' statement MUST encompass all database operations.
        with sqlite3.connect(config.DB_PATH) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            cursor = conn.cursor()

        # # Table to track processed PDF documents and their pipeline status
        # cursor.execute("""
        # CREATE TABLE IF NOT EXISTS documents (
        #     doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
        #     doc_name TEXT NOT NULL UNIQUE,
        #     file_hash TEXT NOT NULL UNIQUE,
        #     status TEXT NOT NULL CHECK(status IN ('extracted', 'chunked_and_summarized', 'embedded'))
        # );
        # """)

        # # --- REPLACE THIS ---
        # # Table to track processed PDF documents and their pipeline status
        # cursor.execute("""
        # CREATE TABLE IF NOT EXISTS documents (
        #     doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
        #     doc_name TEXT NOT NULL UNIQUE,
        #     file_hash TEXT NOT NULL UNIQUE,
        #     status TEXT NOT NULL CHECK(status IN ('extracted', 'chunked_and_summarized', 'embedded'))
        # );
        # """)

        # --- WITH THIS ---
        # Table to track documents, their pipeline status, and lifecycle (active/archived)
        logger.debug("Verifying 'documents' table schema for archival support...")
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_name TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            processing_status TEXT NOT NULL CHECK(processing_status IN ('extracted', 'chunked_and_summarized', 'embedded')),
            lifecycle_status TEXT NOT NULL DEFAULT 'active' CHECK(lifecycle_status IN ('active', 'archived'))
        );
        """
        )

        # Add a unique constraint to ensure only one "active" document with a given name can exist.
        cursor.execute(
            """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_active_document
        ON documents (doc_name, lifecycle_status)
        WHERE lifecycle_status = 'active';
        """
        )

        # Simple migration logic for existing databases.
        # It tries to rename the old 'status' column. If it fails, it's likely already done.
        try:
            cursor.execute("ALTER TABLE documents RENAME COLUMN status TO processing_status;")
            logger.info("Migrated 'documents' table: Renamed 'status' column to 'processing_status'.")
        # except sqlite3.OperationalError as e:
        #     # This is expected if the column is already renamed or never existed.
        #     if "duplicate column name" in str(e) or "no such column" in str(e):
        #         pass
        #     else:
        #         raise
        except sqlite3.OperationalError:
            pass

        # # Table 2: Stores the large, raw text blocks from PDF extraction.
        # Updated to include section_summary for FR-9
        # cursor.execute("""
        # CREATE TABLE IF NOT EXISTS sections (
        #     section_id INTEGER PRIMARY KEY AUTOINCREMENT,
        #     doc_id INTEGER NOT NULL,
        #     page_num INTEGER NOT NULL,
        #     section_header TEXT,
        #     section_summary TEXT, -- Added for FR-9 (Tier 2 Summary)
        #     raw_text TEXT NOT NULL,
        #     header_text TEXT, footer_text TEXT, hyperlink_text TEXT, table_text TEXT,
        #     FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
        # );
        # """)
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS sections (
            section_id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            page_num INTEGER NOT NULL,
            section_header TEXT,
            section_summary TEXT, -- Optional: for future summary
            raw_text TEXT NOT NULL,
            header_text TEXT, 
            footer_text TEXT, 
            hyperlink_text TEXT, 
            table_text TEXT,
            breadcrumb_path TEXT,
            procedure_title TEXT,
            table_summary TEXT,
            image_summary TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents (doc_id) 
        );
        """
        )

        # Table 3: Stores the smaller, token-aware chunks derived from sections.
        # THIS IS THE CORRECTED SCHEMA. This table will hold the summaries and the vector IDs,
        # as each chunk will get its own embedding.
        # cursor.execute("""
        # CREATE TABLE IF NOT EXISTS chunks (
        #     chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
        #     section_id INTEGER NOT NULL,
        #     chunk_text TEXT NOT NULL,
        #     summary TEXT,
        #     vector_id_minilm INTEGER DEFAULT NULL,
        #     vector_id_bge INTEGER DEFAULT NULL,
        #     FOREIGN KEY (section_id) REFERENCES sections (section_id)
        # );
        # """)
        # Table 3: Stores the smaller, token-aware chunks derived from sections.
        # Updated to have specific vector_id columns per model for FR-15 and FR-16

        # cursor.execute("""
        # CREATE TABLE IF NOT EXISTS chunks (
        #     chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
        #     section_id INTEGER NOT NULL,
        #     chunk_text TEXT NOT NULL,
        #     summary TEXT,
        #     vector_id_all_minilm_l6_v2 INTEGER DEFAULT NULL, -- For all-MiniLM-L6-v2
        #     vector_id_baai_bge_small_en_v1_5 INTEGER DEFAULT NULL, -- For BAAI/bge-small-en-v1.5
        #     FOREIGN KEY (section_id) REFERENCES sections (section_id) ON DELETE CASCADE
        # );
        # """)
        # Dynamically build the 'chunks' table schema (FR-15, FR-16)
        chunk_columns = [
            "chunk_id INTEGER PRIMARY KEY AUTOINCREMENT",
            "section_id INTEGER NOT NULL",
            "doc_id INTEGER NOT NULL",
            "chunk_text TEXT NOT NULL",
            "summary TEXT",
        ]
        # Use the new utility to generate a safe column name for each model in the config
        for model_name in config.EMBEDDING_MODELS_LIST:
            safe_col_name = f"vector_id_{get_safe_model_name(model_name)}"
            chunk_columns.append(f"{safe_col_name} INTEGER DEFAULT NULL")

        chunk_columns.append("FOREIGN KEY (section_id) REFERENCES sections (section_id) ")
        chunk_columns.append("FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ")

        create_chunks_table_sql = f"CREATE TABLE IF NOT EXISTS chunks ({', '.join(chunk_columns)});"

        logger.debug(f"Dynamically creating chunks table with SQL: {create_chunks_table_sql}")
        cursor.execute(create_chunks_table_sql)

        # # Table to store the large text blocks extracted from the PDF.
        # # This is the "raw" data before chunking for embedding.
        # cursor.execute("""
        # CREATE TABLE IF NOT EXISTS sections (
        #     section_id INTEGER PRIMARY KEY AUTOINCREMENT,
        #     doc_id INTEGER NOT NULL,
        #     page_num INTEGER NOT NULL,
        #     section_header TEXT,
        #     raw_text TEXT NOT NULL,
        #     summary TEXT,
        #     -- CORRECTED: Added columns to track vector IDs for each model
        #     vector_id_minilm INTEGER DEFAULT NULL,
        #     vector_id_bge INTEGER DEFAULT NULL,
        #     FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
        # );
        # """)

        # In initialize_database(), after all CREATE TABLE statements and before conn.commit()

        logger.debug("Applying performance indexes to database schema...")
        # Index for faster lookups when finding all sections or chunks for a specific document
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sections_doc_id ON sections (doc_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id);")

        # Index for faster lookups when joining chunks back to their parent section
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_section_id ON chunks (section_id);")
        logger.info("Performance indexes have been verified and applied.")

        # conn.commit()
        # conn.close()
        # print(f" Database initialized and ready at: {DB_PATH}")
        # UPDATED: Replaced print with logger
        logger.info(f"Database initialized and schema verified at: {config.DB_PATH}")
    except sqlite3.Error as e:
        logger.critical(f"FATAL: Failed to initialize database schema. Error: {e}", exc_info=True)
        # Re-raise the exception to halt the program, as it cannot continue without a DB.
        raise


def get_db_connection():
    """Returns a connection object to the project database."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row  # Allows accessing columns by name (e.g., row['doc_id'])
    return conn


def calculate_file_hash(file_path: Path) -> str:
    """Calculates the SHA256 hash of a file to detect duplicates."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()
