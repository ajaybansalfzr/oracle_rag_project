import argparse
import os
import sqlite3
import textwrap
from pathlib import Path

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
PROMPT_FOLDER = r"C:\Users\Ajay\Desktop\AI Driven\oracle_rag_project\prompts"  # set to match extracted path

# === Load Embedding Model and FAISS
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(FAISS_PATH)
print("Model and FAISS index loaded.")

# === Load SQLite
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
c = conn.cursor()


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


def read_prompt_template(name):
    path = Path(PROMPT_FOLDER) / f"{name}.txt"
    return path.read_text(encoding="utf-8")  # FIXED


def llama3_call(prompt, system="You are a helpful assistant."):
    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": "llama3",
                "prompt": prompt,
                "system": system,
                "stream": False,
            },
        )
        res.raise_for_status()
        return clean_answer(res.json()["response"].strip())
    except Exception as e:
        return f"[❌ Error: {e}]"


def search_dual_mode(query, top_k=5):
    query_vec = model.encode([query])[0]
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


def ask_llama3_multiphase(primary, secondary, question, persona="consultant", review=False):
    primary_blocks = [f"Primary {i+1}:\n{r['Indexed_Text']}" for i, (_, r) in enumerate(primary)]
    secondary_blocks = [
        f"Secondary {i+1}:\n{r['Paragraph_Synopsis'] or r['Header'] or r['Footer']}"
        for i, (_, r) in enumerate(secondary)
    ]

    # Step 1: Reasoning
    thought_template = read_prompt_template("thought_process")
    thought_prompt = thought_template.format(
        primary_context="\n".join(primary_blocks),
        secondary_context="\n".join(secondary_blocks),
        question=question,
    )
    thought_response = llama3_call(thought_prompt, system="You are a thoughtful reasoning agent.")

    # Step 2: Answer based on Persona
    persona_template_map = {
        "consultant": "consultant_answer",
        "developer": "developer_answer",
        "user": "user_answer",
    }
    template_name = persona_template_map.get(persona.lower(), "consultant_answer")
    answer_prompt = read_prompt_template(template_name).format(thoughts=thought_response, question=question)
    final_answer = llama3_call(answer_prompt)

    # Step 3: Review if enabled
    if review:
        review_template = read_prompt_template("reviewer")
        review_prompt = review_template.format(context="\n".join(primary_blocks[:2]), answer=final_answer)
        review_response = llama3_call(review_prompt)

        # OPTIONAL: Use this line below to log or save the review to a file instead of printing
        print("\n---\n🔍 Review (Debug Mode):\n")
        print(textwrap.fill(review_response, width=100))

    # Always return only the final answer for clean UX
    return final_answer


def main():
    parser = argparse.ArgumentParser(description="Oracle RAG Reasoning Assistant v3")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--persona", type=str, default="consultant", help="consultant, developer, user")
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    primary, secondary = search_dual_mode(args.query, top_k=args.top_k)
    if not primary and not secondary:
        print("No results found.")
        return

    print("=" * 60)
    print("🧠 Oracle RAG v3 Answer:\n")
    result = ask_llama3_multiphase(primary, secondary, args.query, persona=args.persona, review=args.review)
    print(textwrap.fill(result, width=100))
    print("=" * 60)
    print("📚 Sources:")
    for _, row in primary + secondary:
        print(f" - {row['Document_Name']} > {row['Chapter']} > {row['Section']}")
    conn.close()


if __name__ == "__main__":
    main()
