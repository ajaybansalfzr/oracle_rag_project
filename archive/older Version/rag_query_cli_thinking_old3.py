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
def apply_score_threshold(results, threshold=0.6):
    return [(score, row) for score, row in results if score < threshold]


# === Query Expansion ===
def expand_query(query):
    synonyms = {
        "BI tools": [
            "business intelligence tools",
            "analysis tools",
            "report types",
            "reporting tools",
        ],
        "EPM": ["Enterprise Performance Management", "Narrative Reporting"],
        "Accounting Hub": ["AHCS", "AH Reporting"],
    }
    expanded = [query]
    for key, values in synonyms.items():
        if key.lower() in query.lower():
            expanded += [query.replace(key, val) for val in values]
    return list(set(expanded))  # Deduplicate


def search_dual_mode(query, top_k=5):
    queries = expand_query(query)
    query_vec = model.encode(queries).mean(axis=0)
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

    return apply_score_threshold(results_primary), apply_score_threshold(results_secondary)


# === LLaMA 3 via Ollama ===
def ask_llama3_thinking(primary, secondary, question):
    primary_blocks = [f"Primary {i+1}:\n{r['Indexed_Text']}" for i, (_, r) in enumerate(primary)]
    secondary_blocks = [
        f"Secondary {i+1}:\n{r['Paragraph_Synopsis'] or r['Header'] or r['Footer']}"
        for i, (_, r) in enumerate(secondary)
    ]

    prompt = f"""You are an Oracle Cloud expert assistant.

You have access to two types of information:
- Primary Contexts: detailed instructional content and full-text from documentation.
- Secondary Contexts: structured data like tables, metadata, links, and summaries.

Using both sources, analyze and answer the following question with clarity and precision. 
Keep the response focused, fact-based, and point-by-point. 
Cite supporting context when relevant (e.g., “Source: Primary 2” or “From Secondary 3”) without describing the process.

=== PRIMARY CONTEXTS ===
{chr(10).join(primary_blocks)}

=== SECONDARY CONTEXTS ===
{chr(10).join(secondary_blocks)}

Format:
- Use bullet points for lists.
- Use concise statements.
- Cite sources like: (Source: Primary 2) or (Secondary 1).

Question: {question}
"""

    try:
        res = requests.post(OLLAMA_URL, json={"model": "llama3", "prompt": prompt, "stream": False})
        res.raise_for_status()
        raw = res.json()["response"].strip()
        return clean_answer(raw)
    except Exception as e:
        return f"[❌ Error contacting LLaMA 3: {e}]"


# === Utility Functions ===


def clean_answer(text):
    lines = text.split("\n")
    seen = set()
    result = []
    for line in lines:
        l = line.strip()
        if l and l not in seen:
            seen.add(l)
            result.append(l)
    return "\n".join(result)


import re


def preprocess_text(text):
    if not text:
        return ""
    # Simple formatting: bullets, numbered lists
    text = re.sub(r"(\n)([-*]\s)", r"\1• ", text)
    text = re.sub(r"(\n)(\d+\.\s)", r"\1\2", text)
    return text.strip()


# === CLI Setup ===
def main():
    parser = argparse.ArgumentParser(description="RAG with Dual-Mode Thinking (Primary + Secondary)")
    parser.add_argument("--query", type=str, required=True, help="Your search query")
    parser.add_argument("--top_k", type=int, default=5, help="Top K from each mode")
    parser.add_argument("--raw_output", action="store_true", help="Only print LLM answer (no context)")
    parser.add_argument(
        "--show_chunks",
        action="store_true",
        help="Print top-K raw chunks for debugging",
    )
    parser.add_argument("--gold_answer", type=str, help="Evaluate model answer vs gold")
    args = parser.parse_args()

    primary, secondary = search_dual_mode(args.query, top_k=args.top_k)

    if not primary and not secondary:
        print("No results found.")
        return

    if args.show_chunks:
        print("\n📦 Raw Top-K Chunks:")
        for i, (_, row) in enumerate(primary + secondary, 1):
            print(f'\n--- Chunk {i} ---\n{row["Indexed_Text"][:800]}...')
    elif not args.raw_output:
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
    print("📚 Sources:")
    if args.gold_answer:
        print("\n🔬 Eval Mode:")
        match = sum(1 for w in args.gold_answer.lower().split() if w in llm_answer.lower())
        print(f"Match Coverage: {match} / {len(args.gold_answer.split())} words matched")
    for _, row in primary + secondary:
        print(f" - {row['Document_Name']} > {row['Chapter']} > {row['Section']}")


if __name__ == "__main__":
    main()
    conn.close()
