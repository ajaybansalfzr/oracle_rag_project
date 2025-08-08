# scripts/helpers/database_helper.py

import hashlib
import sqlite3
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Go up two levels from helpers/
OUTPUT_DIR = PROJECT_ROOT / "output"
DB_PATH = OUTPUT_DIR / "project_data.db"
OUTPUT_DIR.mkdir(exist_ok=True)


# --- Schema and Connection ---
def initialize_database():
    """Initializes the SQLite database and creates the complete schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table to track processed PDF documents and their pipeline status
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS documents (
        doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_name TEXT NOT NULL UNIQUE,
        file_hash TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL CHECK(status IN ('extracted', 'chunked_and_summarized', 'embedded'))
    );
    """
    )

    # Table 2: Stores the large, raw text blocks from PDF extraction.
    # cursor.execute("""
    # CREATE TABLE IF NOT EXISTS sections (
    #     section_id INTEGER PRIMARY KEY AUTOINCREMENT,
    #     doc_id INTEGER NOT NULL,
    #     page_num INTEGER NOT NULL,
    #     section_header TEXT,
    #     raw_text TEXT NOT NULL,
    #     header_text TEXT, footer_text TEXT, hyperlink_text TEXT, table_text TEXT,
    #     FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
    # );
    # """)

    # # Table 2: Stores the large, raw text blocks from PDF extraction.
    # Updated to include section_summary for FR-9
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS sections (
        section_id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id INTEGER NOT NULL,
        page_num INTEGER NOT NULL,
        section_header TEXT,
        section_summary TEXT, -- Added for FR-9 (Tier 2 Summary)
        raw_text TEXT NOT NULL,
        header_text TEXT, footer_text TEXT, hyperlink_text TEXT, table_text TEXT,
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
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
        section_id INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        summary TEXT,
        vector_id_all_minilm_l6_v2 INTEGER DEFAULT NULL, -- For all-MiniLM-L6-v2
        vector_id_baai_bge_small_en_v1_5 INTEGER DEFAULT NULL, -- For BAAI/bge-small-en-v1.5
        FOREIGN KEY (section_id) REFERENCES sections (section_id)
    );
    """
    )

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

    conn.commit()
    conn.close()
    print(f" Database initialized and ready at: {DB_PATH}")


def get_db_connection():
    """Returns a connection object to the project database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Allows accessing columns by name (e.g., row['doc_id'])
    return conn


def calculate_file_hash(file_path: Path) -> str:
    """Calculates the SHA256 hash of a file to detect duplicates."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()
