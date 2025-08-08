# rag_query_cli.py

import argparse
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os # <-- Import os for path handling

# === CONFIG ===
EMBEDDINGS_FOLDER = r"C:\Users\Ajay\Desktop\AI Driven\oracle_rag_project\embeddings"
# Use os.path.join for robust path creation
DB_PATH = os.path.join(EMBEDDINGS_FOLDER, "oracle_metadata.db")
FAISS_PATH = os.path.join(EMBEDDINGS_FOLDER, "oracle_index.faiss")
MODEL_NAME = 'all-MiniLM-L6-v2'

# === Load Model ===
print("Loading model...")
model = SentenceTransformer(MODEL_NAME)
embedding_size = model.get_sentence_embedding_dimension()
index = faiss.read_index(FAISS_PATH)
print("Model and FAISS index loaded.")

# === Load SQLite DB ===
# Connect to the database
conn = sqlite3.connect(DB_PATH)
# <<< IMPROVEMENT: Use sqlite3.Row factory to access columns by name
# This makes the code more readable and robust (e.g., row['Document_Name'] vs. row[2])
conn.row_factory = sqlite3.Row
c = conn.cursor()

# === Search Function ===
def search(query, mode="primary", top_k=5):
    """
    Searches the FAISS index for the query, then filters results based on the mode
    by retrieving metadata from the SQLite database.
    """
    query_vec = model.encode(query)

    # Search FAISS for a larger number of results to ensure we find enough of the desired mode
    # FAISS index doesn't know about 'primary' or 'secondary', so we filter after the search
    D, I = index.search(np.array([query_vec]), top_k * 5) # Retrieve more to filter from

    results = []
    seen_ids = set() # Keep track of retrieved vector_ids to avoid duplicates

    for i in I[0]:
        # FAISS can return -1 for empty slots
        if i == -1 or i in seen_ids:
            continue

        # Fetch the metadata for the given vector_id from SQLite
        c.execute("SELECT * FROM embeddings WHERE vector_id = ?", (int(i),)) # Use int() for safety
        row = c.fetchone()

        # Check if the row exists and its source_type matches the desired mode
        if row and row['source_type'] == mode:
            # The 'D' matrix contains the L2 distance (lower is better)
            distance = D[0][list(I[0]).index(i)]
            results.append((distance, row))
            seen_ids.add(i)

        # Stop once we have collected enough results
        if len(results) >= top_k:
            break

    # Sort final results by distance (score)
    results.sort(key=lambda x: x[0])
    return results

# === CLI Setup ===
def main():
    parser = argparse.ArgumentParser(description="Search the Oracle RAG database from the command line.")
    parser.add_argument("--query", type=str, required=True, help="Your search query")
    parser.add_argument("--mode", type=str, choices=["primary", "secondary"], default="primary", help="Embedding mode to search ('primary' for content, 'secondary' for metadata)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to display")
    args = parser.parse_args()

    matches = search(args.query, mode=args.mode, top_k=args.top_k)

    print(f"\n🔍 Top {len(matches)} results (mode = {args.mode}):\n")
    if not matches:
        print("No results found.")
        return

    for idx, (dist, row) in enumerate(matches, 1):
        # <<< IMPROVEMENT: Accessing columns by name is much cleaner
        doc_name = row['Document_Name']
        chapter = row['Chapter']
        section = row['Section']
        subsection = row['SubSection']
        synopsis = row['Paragraph_Synopsis']
        indexed_text = row['Indexed_Text']

        print(f"[{idx}] Score: {dist:.4f}")
        print(f"   Document: {doc_name} | Chapter: {chapter} | Section: {section} | SubSection: {subsection}")
        if synopsis:
            print(f"   Synopsis: {synopsis}")
        print(f"   Text: {indexed_text[:500]}{'...' if len(indexed_text) > 500 else ''}\n")

if __name__ == "__main__":
    main()
    # Close the database connection when the script is done
    conn.close()