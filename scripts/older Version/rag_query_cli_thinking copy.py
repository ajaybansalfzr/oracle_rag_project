# scripts/rag_query_cli_thinking.py

import argparse
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import requests
from pathlib import Path
import pickle
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import sys

# --- Setup ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_FOLDER = PROJECT_ROOT / "embeddings"
PROMPT_FOLDER = PROJECT_ROOT / "prompts"
DB_PATH = EMBEDDINGS_FOLDER / "oracle_metadata.db"
FAISS_PATH = EMBEDDINGS_FOLDER / "oracle_index.faiss"
BM25_PATH = EMBEDDINGS_FOLDER / "oracle_bm25.pkl"
MODEL_NAME_RETRIEVER = 'all-MiniLM-L6-v2'
MODEL_NAME_RERANKER = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
OLLAMA_URL = "http://localhost:11434/api/generate"

# --- Load Models and Indexes ---
try:
    print("Loading models and indexes...")
    model_retriever = SentenceTransformer(MODEL_NAME_RETRIEVER)
    model_reranker = CrossEncoder(MODEL_NAME_RERANKER)
    index_faiss = faiss.read_index(str(FAISS_PATH))
    with open(BM25_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    index_bm25, bm25_doc_map = bm25_data["index"], bm25_data["doc_map"]
    
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    print("Models and indexes loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load models or database. Please ensure 'rag_embed_store.py' has been run. Details: {e}", file=sys.stderr)
    sys.exit(1)

# --- Helper Functions ---
def clean_answer(text: str) -> str: return text.strip()
def read_prompt_template(name: str) -> str:
    path = PROMPT_FOLDER / f"{name}.txt"
    try: return path.read_text(encoding='utf-8')
    except: return "Based ONLY on the context, answer the question.\n\nContext:\n{context}\n\nQuestion:\n{question}"

def llama3_call(prompt: str, system: str) -> str:
    try:
        print("\nGenerating answer with Llama 3...")
        res = requests.post(OLLAMA_URL, json={"model": "llama3", "prompt": prompt, "system": system, "stream": False}, timeout=60)
        res.raise_for_status()
        return clean_answer(res.json().get("response", "[LLM Error: No response found]"))
    except requests.exceptions.RequestException as e: return f"[LLM Error: {e}]"

# --- RAG Pipeline ---
def hybrid_search(query: str, top_n: int = 20) -> list[int]:
    print(f"\nStep 1: Performing Hybrid Search for query: '{query}'")
    
    # 1. Semantic Search (FAISS)
    query_vec = model_retriever.encode([query])
    _, vector_ids_faiss = index_faiss.search(query_vec, top_n)
    vector_ids_faiss = vector_ids_faiss[0]

    # 2. Keyword Search (BM25)
    tokenized_query = word_tokenize(query.lower())
    bm25_doc_scores = index_bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_doc_scores)[::-1][:top_n]
    # Use the robust map to get chunk_ids
    chunk_ids_bm25 = [bm25_doc_map[i] for i in top_bm25_indices]
    
    vector_ids_bm25 = []
    if chunk_ids_bm25:
        placeholders = ','.join('?' for _ in chunk_ids_bm25)
        c.execute(f"SELECT chunk_id, vector_id FROM metadata WHERE chunk_id IN ({placeholders})", chunk_ids_bm25)
        chunk_to_vec_map = {row['chunk_id']: row['vector_id'] for row in c.fetchall()}
        vector_ids_bm25 = [chunk_to_vec_map[cid] for cid in chunk_ids_bm25 if cid in chunk_to_vec_map]

    # 3. Reciprocal Rank Fusion (RRF)
    k=60; fused_scores = {}
    for r, v_id in enumerate(vector_ids_faiss):
        if v_id != -1: fused_scores[v_id] = fused_scores.get(v_id, 0) + (0.3 * (1 / (k + r + 1)))
    for r, v_id in enumerate(vector_ids_bm25):
        fused_scores[v_id] = fused_scores.get(v_id, 0) + (0.7 * (1 / (k + r + 1)))
            
    sorted_vector_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    print(f"Hybrid search identified {len(fused_scores)} unique candidate chunks.")
    return sorted_vector_ids

def rerank_and_format_context(query: str, vector_ids: list, top_k: int) -> tuple[str, list]:
    print(f"\nStep 2: Re-ranking {len(vector_ids)} candidates to select top {top_k}...")
    if not vector_ids: return "", []

    placeholders = ','.join('?' for _ in vector_ids)
    c.execute(f"SELECT chunk_id, chunk_text FROM metadata WHERE vector_id IN ({placeholders})", vector_ids)
    rows = c.fetchall()
    if not rows: return "", []

    chunk_data = [{'id': row['chunk_id'], 'text': row['chunk_text']} for row in rows]
    query_pairs = [[query, chunk['text']] for chunk in chunk_data]
    scores = model_reranker.predict(query_pairs)

    for i, chunk in enumerate(chunk_data): chunk['score'] = scores[i]
    
    reranked = sorted(chunk_data, key=lambda x: x['score'], reverse=True)
    top_chunk_ids = [chunk['id'] for chunk in reranked[:top_k]]
    if not top_chunk_ids: return "Re-ranking failed.", []

    print(f"Re-ranking complete. Formatting context for top {len(top_chunk_ids)} chunks.")
    placeholders = ','.join('?' for _ in top_chunk_ids)
    order_clause = ' '.join(f'WHEN ? THEN {i}' for i in range(len(top_chunk_ids)))
    
    c.execute(f"SELECT * FROM metadata WHERE chunk_id IN ({placeholders}) ORDER BY CASE chunk_id {order_clause} END", top_chunk_ids + top_chunk_ids)
    
    final_context, citations = [], []
    for row in c.fetchall():
        context_part = (f"📄 Doc: {row['document_name']}\n📑 Section: {row['section_id']}\n📄 Page: {row['page_num']}\n\n"
                        f"🔹 Text:\n---\n{row['chunk_text']}\n---")
        if row['hyperlink_text']: context_part += f"\n\n🔗 Links:\n{row['hyperlink_text']}"
        if row['table_text']: context_part += f"\n\n📊 Table:\n{row['table_text']}"
        if row['header_text'] or row['footer_text']: context_part += f"\n\n🧾 Header/Footer:\n{row['header_text']} / {row['footer_text']}"
        final_context.append(context_part)
        citations.append(f"Doc: {row['document_name']} | Page: {row['page_num']} | Section: {row['section_id']}")

    return "\n\n---\n\n".join(final_context), citations

def main():
    parser = argparse.ArgumentParser(description="Oracle RAG v6 - Production Pipeline")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--show_chunks", action="store_true")
    args = parser.parse_args()

    candidate_ids = hybrid_search(args.query, top_n=20)
    if not candidate_ids:
        print("No candidate documents found from hybrid search.")
        return

    context, sources = rerank_and_format_context(args.query, candidate_ids, top_k=args.top_k)
    if not context:
        print("\nNo relevant context could be found after re-ranking.")
        return

    if args.show_chunks:
        print("\n" + "="*60 + "\nDETAILED CONTEXT CHUNKS (FED TO LLM)\n" + "="*60)
        print(context)
        print("="*60 + "\n")

    system_prompt = read_prompt_template("consultant_answer")
    user_prompt = f"Based ONLY on the context below, answer the question.\n\n**Question:** {args.query}\n\n**Context:**\n---\n{context}\n---"
    result = llama3_call(prompt=user_prompt, system=system_prompt)

    print("\n" + "="*20 + " Oracle RAG Final Answer " + "="*20)
    print(result)
    print("=" * 66 + "\nSOURCES:")
    for source in sources: print(f" - {source}")
    print("=" * 66)

    conn.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)