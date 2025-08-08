# rag_query_cli_thinking.py (Updated with 3-pass logic and persona system)

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
    return list(set(expanded))


def apply_score_threshold(results, threshold=0.6):
    return [(score, row) for score, row in results if score < threshold]


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


def ask_llama3_thinking(primary, secondary, question, persona="consultant", review=False):
    primary_blocks = [f"Primary {i+1}:\n{r['Indexed_Text']}" for i, (_, r) in enumerate(primary)]
    secondary_blocks = [
        f"Secondary {i+1}:\n{r['Paragraph_Synopsis'] or r['Header'] or r['Footer']}"
        for i, (_, r) in enumerate(secondary)
    ]

    # STEP 1: Thought Process
    thought_prompt = f"""You are analyzing Oracle Cloud documentation.

Your job is to *think step-by-step* using the context below — extract important points, conflicts, patterns, or missing data, but don’t answer the question yet.

Context:
=== PRIMARY ===
{chr(10).join(primary_blocks)}
=== SECONDARY ===
{chr(10).join(secondary_blocks)}

Question: {question}

🧠 Thought Process:"""
    thought_response = llama3_call(thought_prompt, system="You are a thoughtful reasoning agent.")

    # STEP 2: Persona-based Answer
    role_map = {
        "consultant": "You are an Oracle Cloud Implementation Consultant.",
        "developer": "You are a technical Oracle Cloud developer.",
        "user": "You are a business end-user using Oracle Cloud apps.",
    }
    persona_role = role_map.get(persona.lower(), role_map["consultant"])
    final_prompt = f"""{persona_role}

Based on the earlier reasoning and your role, answer the question below clearly and use bullet points.

🧠 Thought Process:
{thought_response}

Question: {question}

✅ Final Answer (with source if relevant):"""
    final_answer = llama3_call(final_prompt)

    # STEP 3: Optional Review
    if review:
        review_prompt = f"""You are a QA reviewer. Critically review the answer below for clarity, accuracy, or missing context.

Context:
{chr(10).join(primary_blocks[:2])}

Answer:
{final_answer}

🧪 Review Comments:"""
        review_notes = llama3_call(review_prompt)
        return f"{final_answer}\n\n---\n🔍 Review:\n{review_notes}"

    return final_answer


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


def main():
    parser = argparse.ArgumentParser(description="RAG with 3-pass Reasoning + Persona Control")
    parser.add_argument("--query", type=str, required=True, help="Your search query")
    parser.add_argument("--top_k", type=int, default=5, help="Top K from each mode")
    parser.add_argument(
        "--persona",
        type=str,
        default="consultant",
        help="Persona: consultant, developer, or user",
    )
    parser.add_argument("--review", action="store_true", help="Run review pass after answering")
    args = parser.parse_args()

    primary, secondary = search_dual_mode(args.query, top_k=args.top_k)
    if not primary and not secondary:
        print("No results found.")
        return

    print("============================================================")
    print("🧠 LLaMA 3 Answer (Dual Mode + Reasoning):\n")
    llm_answer = ask_llama3_thinking(primary, secondary, args.query, persona=args.persona, review=args.review)
    print(textwrap.fill(llm_answer, width=100))
    print("============================================================")
    print("📚 Sources:")
    for _, row in primary + secondary:
        print(f" - {row['Document_Name']} > {row['Chapter']} > {row['Section']}")


if __name__ == "__main__":
    main()
    conn.close()
