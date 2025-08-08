# rag_embed_store.py

import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3

# === CONFIG ===
CSV_FOLDER = r"C:\Users\Ajay\Desktop\AI Driven\oracle_rag_project\output"
EMBEDDINGS_FOLDER = r"C:\Users\Ajay\Desktop\AI Driven\oracle_rag_project\embeddings"
MODEL_NAME = 'all-MiniLM-L6-v2'
DB_PATH = os.path.join(EMBEDDINGS_FOLDER, "oracle_metadata.db")
FAISS_PATH = os.path.join(EMBEDDINGS_FOLDER, "oracle_index.faiss")

# === Clean slate ===
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
if os.path.exists(FAISS_PATH):
    os.remove(FAISS_PATH)

os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)

# === Load Model ===
model = SentenceTransformer(MODEL_NAME)
embedding_size = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_size)

# === Setup SQLite ===
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
# Note: The Embedding_Vector column was missing from the INSERT statement values.
c.execute("""
    CREATE TABLE embeddings (
        vector_id INTEGER PRIMARY KEY,
        source_type TEXT,
        Document_Name TEXT,
        Chapter TEXT,
        Section TEXT,
        SubSection TEXT,
        Paragraph_Synopsis TEXT,
        Sentence_Count INTEGER,
        Additional_Context TEXT,
        Header TEXT,
        Footer TEXT,
        Indexed_Text TEXT,
        Table_Delimited TEXT,
        Hyperlinks TEXT,
        Embedding_Vector BLOB
    )
""")
conn.commit()

# === Process CSVs ===
vector_id = 0
for csv_file in os.listdir(CSV_FOLDER):
    if not csv_file.startswith("Oracle_Data_") or not csv_file.endswith(".csv"):
        continue

    # It's good practice to wrap file reading in a try-except block
    try:
        df = pd.read_csv(os.path.join(CSV_FOLDER, csv_file))
    except Exception as e:
        print(f"Could not read {csv_file}. Error: {e}")
        continue


    for _, row in df.iterrows():
        row_data = row.to_dict()

        # === Primary Embedding ===
        primary_text = str(row_data.get("Additional_Context", "")) + "\n" + str(row_data.get("Indexed_Text", ""))
        if not primary_text.strip():
            continue

        primary_vec = model.encode(primary_text)
        index.add(np.array([primary_vec]))
        c.execute("""
            INSERT INTO embeddings (
                vector_id, source_type, Document_Name, Chapter, Section, SubSection,
                Paragraph_Synopsis, Sentence_Count, Additional_Context,
                Header, Footer, Indexed_Text, Table_Delimited, Hyperlinks, Embedding_Vector
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            vector_id, "primary",
            row_data.get("Document_Name", ""), row_data.get("Chapter", ""), row_data.get("Section", ""),
            row_data.get("SubSection", ""), row_data.get("Paragraph_Synopsis", ""), row_data.get("Sentence_Count", 0),
            row_data.get("Additional_Context", ""), row_data.get("Header", ""), row_data.get("Footer", ""),
            row_data.get("Indexed_Text", ""), row_data.get("Table_Delimited", ""), row_data.get("Hyperlinks", ""),
            primary_vec.tobytes() # <<< FIXED: Added the embedding vector blob
        ))
        vector_id += 1

        # === Secondary Embedding ===
        meta_parts = [
            str(row_data.get(k, "")) for k in ["Chapter", "Section", "SubSection", "Paragraph_Synopsis", "Header", "Footer", "Hyperlinks"]
        ]
        secondary_text = "\n".join(filter(None, meta_parts)) # filter(None, ...) to avoid empty strings creating extra newlines
        
        # Only create a secondary embedding if there is text content
        if not secondary_text.strip():
            continue

        secondary_vec = model.encode(secondary_text)
        index.add(np.array([secondary_vec]))
        c.execute("""
            INSERT INTO embeddings (
                vector_id, source_type, Document_Name, Chapter, Section, SubSection,
                Paragraph_Synopsis, Sentence_Count, Additional_Context,
                Header, Footer, Indexed_Text, Table_Delimited, Hyperlinks, Embedding_Vector
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            vector_id, "secondary",
            row_data.get("Document_Name", ""), row_data.get("Chapter", ""), row_data.get("Section", ""),
            row_data.get("SubSection", ""), row_data.get("Paragraph_Synopsis", ""), row_data.get("Sentence_Count", 0),
            row_data.get("Additional_Context", ""), row_data.get("Header", ""), row_data.get("Footer", ""),
            row_data.get("Indexed_Text", ""), row_data.get("Table_Delimited", ""), row_data.get("Hyperlinks", ""),
            secondary_vec.tobytes() # <<< FIXED: Added the embedding vector blob
        ))
        vector_id += 1

conn.commit()
conn.close()

# === Save FAISS Index ===
faiss.write_index(index, FAISS_PATH)
print(f"✅ Stored {vector_id} total embeddings (primary + secondary)")