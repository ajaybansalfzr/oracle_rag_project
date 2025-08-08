import sqlite3
from pathlib import Path
from scripts.utils.database_utils import initialize_database, get_db_connection

def verify_schema():
    """Initializes and then verifies the 'sections' table schema."""
    print("Running database initialization...")
    initialize_database()
    
    print("\nConnecting to database to verify schema...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(sections);")
    columns = [row['name'] for row in cursor.fetchall()]
    
    print(f"\nColumns found in 'sections' table: {columns}")
    
    if 'section_summary' in columns:
        print("\n[SUCCESS] 'section_summary' column found in the sections table.")
    else:
        print("\n[FAILURE] 'section_summary' column is MISSING from the sections table.")
    
    conn.close()

if __name__ == "__main__":
    verify_schema()