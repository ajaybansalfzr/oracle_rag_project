# scripts/rag_embed_store.py

import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
from pathlib import Path
import pickle
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

try: nltk.data.find('tokenizers/punkt')
except: nltk.download('punkt', quiet=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_FOLDER = PROJECT_ROOT / "output"
EMBEDDINGS_FOLDER = PROJECT_ROOT / "embeddings"
MODEL_NAME = 'all-MiniLM-L6-v2'
DB_PATH = EMBEDDINGS_FOLDER / "oracle_metadata.db"
FAISS_PATH = EMBEDDINGS_FOLDER / "oracle_index.faiss"
BM25_PATH = EMBEDDINGS_FOLDER / "oracle_bm25.pkl"

for path in [DB_PATH, FAISS_PATH, BM25_PATH]:
    if os.path.exists(path): os.remove(path)
os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)

print("Loading models...")
model_semantic = SentenceTransformer(MODEL_NAME)
index_faiss = faiss.IndexFlatL2(model_semantic.get_sentence_embedding_dimension())
print("Models loaded.")

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# <<< MODIFIED: New schema for semantically chunked data
c.execute("""
    CREATE TABLE embeddings (
        vector_id INTEGER PRIMARY KEY,
        chunk_id TEXT NOT NULL UNIQUE,
        Document_Name TEXT,
        Chapter TEXT,
        Section TEXT,
        SubSection TEXT,
        Content_Markdown TEXT,
        Hyperlinks TEXT,
        Image_Text TEXT
    )
""")
conn.commit()

vector_id_counter = 0
corpus_for_bm25 = []
chunk_id_map_for_bm25 = []

csv_files = list(CSV_FOLDER.glob("Oracle_Data_*.csv"))
if not csv_files:
    print("No processed CSV files found in 'output' folder.")
else:
    for csv_file in csv_files:
        print(f"Processing {csv_file.name} for embedding...")
        df = pd.read_csv(csv_file).fillna('')

        for _, row in df.iterrows():
            # <<< MODIFIED: Create a rich text block for embedding from all data
            text_to_embed = f"Chapter: {row['Chapter']}\nSection: {row['Section']}\n{row['SubSection']}\n\n{row['Content_Markdown']}"
            if not text_to_embed.strip(): continue

            # --- FAISS Index ---
            embedding = model_semantic.encode(text_to_embed)
            index_faiss.add(np.array([embedding]))

            # --- BM25 Corpus ---
            corpus_for_bm25.append(word_tokenize(text_to_embed.lower()))
            chunk_id_map_for_bm25.append(row['chunk_id'])

            # --- SQLite Store ---
            c.execute("""
                INSERT INTO embeddings (
                    vector_id, chunk_id, Document_Name, Chapter, Section, SubSection,
                    Content_Markdown, Hyperlinks, Image_Text
                ) VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                vector_id_counter, row['chunk_id'], row['Document_Name'],
                row['Chapter'], row['Section'], row['SubSection'],
                row['Content_Markdown'], row['Hyperlinks'], row['Image_Text']
            ))
            vector_id_counter += 1
            
    # --- Finalize and Save Indexes ---
    conn.commit()
    conn.close()

    faiss.write_index(index_faiss, str(FAISS_PATH))
    print(f"\nOK: Saved FAISS index with {index_faiss.ntotal} vectors.")

    if corpus_for_bm25:
        bm25 = BM25Okapi(corpus_for_bm25)
        bm25_data = {"index": bm25, "chunk_id_map": chunk_id_map_for_bm25}
        with open(BM25_PATH, "wb") as f: pickle.dump(bm25_data, f)
        print("OK: Saved BM25 index.")

    print(f"\nTotal semantic sections embedded: {vector_id_counter}")