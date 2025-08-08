# rag_query_cli.py

import argparse
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import requests  # NEW: For Ollama LLaMA3
import textwrap

# === CONFIG ===
EMBEDDINGS_FOLDER = r"C:\Users\Ajay\Desktop\AI Driven\oracle_rag_project\embeddings"
DB_PATH = os.path.join(EMBEDDINGS_FOLDER, "oracle_metadata.db")
FAISS_PATH = os.path.join(EMBEDDINGS_FOLDER, "oracle_index.faiss")
MODEL_NAME = 'all-MiniLM-L6-v2'
OLLAMA_URL = "http://localhost:11434/api/generate"

# === Load Model ===
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
embedding_size = model.get_sentence_embedding_dimension()
index = faiss.read_index(FAISS_PATH)
print("Model and FAISS index loaded.")

# === Load SQLite DB ===
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
c = conn.cursor()

# === Search Function ===
def search(query, mode="primary", top_k=5):
    query_vec = model.encode(query)
    D, I = index.search(np.array([query_vec]), top_k * 5)
    results = []
    seen_ids = set()

    for i in I[0]:
        if i == -1 or i in seen_ids:
            continue
        c.execute("SELECT * FROM embeddings WHERE vector_id = ?", (int(i),))
        row = c.fetchone()
        if row and row["source_type"] == mode:
            distance = D[0][list(I[0]).index(i)]
            results.append((distance, row))
            seen_ids.add(i)
        if len(results) >= top_k:
            break

    results.sort(key=lambda x: x[0])
    return results

# === LLaMA 3 via Ollama ===
def ask_llama3(context_blocks, question):
    prompt = (
        "Answer the user's question using only the following context blocks. "
        "Be specific and include source references.\n\n"
        + "\n\n".join(context_blocks)
        + f"\n\nQuestion: {question}"
    )
    try:
        res = requests.post(OLLAMA_URL, json={"model": "llama3", "prompt": prompt, "stream": False})
        res.raise_for_status()
        return res.json()["response"].strip()
    except Exception as e:
        return f"[❌ Error contacting LLaMA 3: {e}]"

# === CLI Setup ===
def main():
    parser = argparse.ArgumentParser(description="Oracle RAG CLI with LLaMA 3 Answering")
    parser.add_argument("--query", type=str, required=True, help="Your search query")
    parser.add_argument("--mode", type=str, choices=["primary", "secondary"], default="primary", help="Which embedding type to use")
    parser.add_argument("--top_k", type=int, default=5, help="Top K results to retrieve")
    parser.add_argument("--raw_output", action="store_true", help="Only print final LLM answer (no context shown)")
    args = parser.parse_args()

    matches = search(args.query, mode=args.mode, top_k=args.top_k)

    if not matches:
        print("No results found.")
        return

    context_blocks = []
    for idx, (dist, row) in enumerate(matches, 1):
        doc_name = row['Document_Name']
        chapter = row['Chapter']
        section = row['Section']
        subsection = row['SubSection']
        indexed_text = row['Indexed_Text']

        context_block = f"Context {idx} (Doc: {doc_name}, Section: {section}):\n{indexed_text}"
        context_blocks.append(context_block)

        if not args.raw_output:
            print(f"\n[{idx}] Score: {dist:.4f}")
            print(f"   Document: {doc_name} | Chapter: {chapter} | Section: {section} | SubSection: {subsection}")
            print(textwrap.fill(indexed_text.strip(), width=100))

    # === LLM Response ===
    llm_answer = ask_llama3(context_blocks, args.query)

    print("\n" + "=" * 60)
    print("🧠 LLaMA 3 Answer:\n")
    print(textwrap.fill(llm_answer, width=100))
    print("=" * 60)

if __name__ == "__main__":
    main()
    conn.close()
