# rag_query_cli_thinking.py

import argparse
import os
import sqlite3
import textwrap

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# === CONFIG ===
EMBEDDINGS_FOLDER = r"C:\Users\Ajay\Desktop\AI Driven\oracle_rag_project\embeddings"
DB_PATH = os.path.join(EMBEDDINGS_FOLDER, "oracle_metadata.db")
FAISS_PATH = os.path.join(EMBEDDINGS_FOLDER, "oracle_index.faiss")
MODEL_NAME = "all-MiniLM-L6-v2"
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


# === Dual-Mode Search Function ===
def search_dual_mode(query, top_k=5):
    query_vec = model.encode(query)
    D, I = index.search(np.array([query_vec]), top_k * 5)
    results_primary, results_secondary = [], []
    seen_ids = set()

    for i in I[0]:
        if i == -1 or i in seen_ids:
            continue
        c.execute("SELECT * FROM embeddings WHERE vector_id = ?", (int(i),))
        row = c.fetchone()
        if not row:
            continue

        distance = D[0][list(I[0]).index(i)]
        if row["source_type"] == "primary" and len(results_primary) < top_k:
            results_primary.append((distance, row))
        elif row["source_type"] == "secondary" and len(results_secondary) < top_k:
            results_secondary.append((distance, row))

        seen_ids.add(i)
        if len(results_primary) >= top_k and len(results_secondary) >= top_k:
            break

    return results_primary, results_secondary


# === LLaMA 3 via Ollama ===
def ask_llama3_thinking(primary, secondary, question):
    primary_blocks = [f"Primary {i+1}:\n{r['Indexed_Text']}" for i, (_, r) in enumerate(primary)]
    secondary_blocks = [
        f"Secondary {i+1}:\n{r['Paragraph_Synopsis'] or r['Header'] or r['Footer']}"
        for i, (_, r) in enumerate(secondary)
    ]

    prompt = f"""You are an intelligent assistant that reasons using two perspectives:
- Primary Contexts: rich full-text instructional content
- Secondary Contexts: metadata, tables, hyperlinks, and high-level summaries

Step 1: Carefully read and think through the Primary and Secondary Contexts.
Step 2: Cross-check insights from both to find accurate, nuanced conclusions.
Step 3: Respond clearly, citing which context helped support each part of your answer.

=== PRIMARY CONTEXTS ===
{chr(10).join(primary_blocks)}

=== SECONDARY CONTEXTS ===
{chr(10).join(secondary_blocks)}

Question: {question}
"""

    try:
        res = requests.post(OLLAMA_URL, json={"model": "llama3", "prompt": prompt, "stream": False})
        res.raise_for_status()
        return res.json()["response"].strip()
    except Exception as e:
        return f"[❌ Error contacting LLaMA 3: {e}]"


# === CLI Setup ===
def main():
    parser = argparse.ArgumentParser(description="RAG with Dual-Mode Thinking (Primary + Secondary)")
    parser.add_argument("--query", type=str, required=True, help="Your search query")
    parser.add_argument("--top_k", type=int, default=5, help="Top K from each mode")
    parser.add_argument("--raw_output", action="store_true", help="Only print LLM answer (no context)")
    args = parser.parse_args()

    primary, secondary = search_dual_mode(args.query, top_k=args.top_k)

    if not primary and not secondary:
        print("No results found.")
        return

    if not args.raw_output:
        print("\n🔹 PRIMARY CONTEXTS:")
        for i, (_, row) in enumerate(primary, 1):
            print(f"\n[{i}] Document: {row['Document_Name']} | Section: {row['Section']}")
            print(textwrap.fill(row["Indexed_Text"].strip(), width=100))

        print("\n🔹 SECONDARY CONTEXTS:")
        for i, (_, row) in enumerate(secondary, 1):
            context = row["Paragraph_Synopsis"] or row["Header"] or row["Footer"]
            print(f"\n[{i}] {context}")

    llm_answer = ask_llama3_thinking(primary, secondary, args.query)

    print("\n" + "=" * 60)
    print("🧠 LLaMA 3 Answer (Dual Mode):\n")
    print(textwrap.fill(llm_answer, width=100))
    print("=" * 60)


if __name__ == "__main__":
    main()
    conn.close()
