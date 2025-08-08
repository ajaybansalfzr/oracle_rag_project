import argparse
import pickle
import sqlite3
import sys
from pathlib import Path

import faiss
import nltk
import numpy as np
import requests
from nltk.tokenize import word_tokenize
from sentence_transformers import CrossEncoder, SentenceTransformer

# --- Setup NLTK ---
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt", quiet=True)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_FOLDER = PROJECT_ROOT / "embeddings"
PROMPT_FOLDER = PROJECT_ROOT / "prompts"
# Model 1 resources
DB_PATH_1 = EMBEDDINGS_FOLDER / "oracle_metadata_all-MiniLM-L6-v2.db"
FAISS_PATH_1 = EMBEDDINGS_FOLDER / "oracle_index_all-MiniLM-L6-v2.faiss"
BM25_PATH_1 = EMBEDDINGS_FOLDER / "oracle_bm25_all-MiniLM-L6-v2.pkl"
# Model 2 resources
DB_PATH_2 = EMBEDDINGS_FOLDER / "oracle_metadata_BAAI-bge-small-en-v1.5.db"
FAISS_PATH_2 = EMBEDDINGS_FOLDER / "oracle_index_BAAI-bge-small-en-v1.5.faiss"
BM25_PATH_2 = EMBEDDINGS_FOLDER / "oracle_bm25_BAAI-bge-small-en-v1.5.pkl"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME_RETRIEVER = "all-MiniLM-L6-v2"
MODEL_NAME_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Load Models and Indexes ---
try:
    print("Loading models and indexes...")
    model_retriever = SentenceTransformer(MODEL_NAME_RETRIEVER)
    model_reranker = CrossEncoder(MODEL_NAME_RERANKER)
    # Model 1
    index_faiss_1 = faiss.read_index(str(FAISS_PATH_1))
    with open(BM25_PATH_1, "rb") as f:
        bm25_data_1 = pickle.load(f)
    index_bm25_1, bm25_doc_map_1 = bm25_data_1["index"], bm25_data_1["doc_map"]
    conn1 = sqlite3.connect(f"file:{DB_PATH_1}?mode=ro", uri=True)
    conn1.row_factory = sqlite3.Row
    c1 = conn1.cursor()
    # Model 2
    index_faiss_2 = faiss.read_index(str(FAISS_PATH_2))
    with open(BM25_PATH_2, "rb") as f:
        bm25_data_2 = pickle.load(f)
    index_bm25_2, bm25_doc_map_2 = bm25_data_2["index"], bm25_data_2["doc_map"]
    conn2 = sqlite3.connect(f"file:{DB_PATH_2}?mode=ro", uri=True)
    conn2.row_factory = sqlite3.Row
    c2 = conn2.cursor()
    print("Models and indexes loaded successfully.")
except Exception as e:
    print(
        f"FATAL ERROR: Could not load models or database. Please ensure 'rag_embed_store.py' has been run. Details: {e}",
        file=sys.stderr,
    )
    sys.exit(1)


# --- Helper Functions ---
def clean_answer(text: str) -> str:
    return text.strip()


def read_prompt_template(name: str) -> str:
    path = PROMPT_FOLDER / f"{name}.txt"
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def llama3_call(prompt: str, system: str) -> str:
    try:
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": "llama3",
                "prompt": prompt,
                "system": system,
                "stream": False,
            },
            timeout=60,
        )
        res.raise_for_status()
        return clean_answer(res.json().get("response", ""))
    except Exception as e:
        return f"[LLM Error: {e}]"


# --- RAG Retrieval Functions ---
def hybrid_search(query: str, top_n: int = 20, ensemble: bool = False) -> list[str]:
    print(f"\nStep 1: Performing {'Ensemble' if ensemble else 'Hybrid'} Search for query: '{query}'")
    # Encode query
    query_vec = model_retriever.encode([query])
    tokens = word_tokenize(query.lower())
    fused = {}

    # --- Model 1 (all-MiniLM-L6-v2) Search ---
    faiss_ids_1 = index_faiss_1.search(query_vec, top_n)[1][0]
    bm25_scores_1 = index_bm25_1.get_scores(tokens)
    top_bm25_idx_1 = np.argsort(bm25_scores_1)[::-1][:top_n]
    bm25_ids_1 = []
    if top_bm25_idx_1.size:
        chunk_ids = [bm25_doc_map_1[i] for i in top_bm25_idx_1]
        placeholders = ",".join("?" for _ in chunk_ids)
        c1.execute(
            f"SELECT vector_id FROM metadata WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        bm25_ids_1 = [row["vector_id"] for row in c1.fetchall()]

    print(f"FAISS/BM25 scores from Model 1: FAISS={len([v for v in faiss_ids_1 if v != -1])}, BM25={len(bm25_ids_1)}")

    # --- Logic for Fusing Results ---
    k = 60  # RRF constant

    if not ensemble:
        # --- SINGLE MODEL (HYBRID) FUSION ---
        # Fuse results from Model 1 only
        for r, vid in enumerate(faiss_ids_1):
            if vid != -1:
                fused[str(vid)] = fused.get(str(vid), 0) + 0.3 * (1 / (k + r + 1))
        for r, vid in enumerate(bm25_ids_1):
            fused[str(vid)] = fused.get(str(vid), 0) + 0.7 * (1 / (k + r + 1))

        sorted_ids = sorted(fused, key=fused.get, reverse=True)
        print(f"Total fused chunks: {len(sorted_ids)}")
        return sorted_ids

    else:
        # --- ENSEMBLE (TWO MODEL) FUSION ---
        # Add Model 1 results to the fusion list with model prefix
        for r, vid in enumerate(faiss_ids_1):
            if vid != -1:
                key = f"m1_{vid}"
                fused[key] = fused.get(key, 0) + 0.3 * (1 / (k + r + 1))
        for r, vid in enumerate(bm25_ids_1):
            key = f"m1_{vid}"
            fused[key] = fused.get(key, 0) + 0.7 * (1 / (k + r + 1))

        # --- Model 2 (BAAI-bge-small-en-v1.5) Search ---
        faiss_ids_2 = index_faiss_2.search(query_vec, top_n)[1][0]
        bm25_scores_2 = index_bm25_2.get_scores(tokens)
        top_bm25_idx_2 = np.argsort(bm25_scores_2)[::-1][:top_n]
        bm25_ids_2 = []
        if top_bm25_idx_2.size:
            chunk_ids2 = [bm25_doc_map_2[i] for i in top_bm25_idx_2]
            placeholders2 = ",".join("?" for _ in chunk_ids2)
            c2.execute(
                f"SELECT vector_id FROM metadata WHERE chunk_id IN ({placeholders2})",
                chunk_ids2,
            )
            bm25_ids_2 = [row["vector_id"] for row in c2.fetchall()]

        print(
            f"FAISS/BM25 scores from Model 2: FAISS={len([v for v in faiss_ids_2 if v != -1])}, BM25={len(bm25_ids_2)}"
        )

        # Add Model 2 results to the fusion list
        for r, vid in enumerate(faiss_ids_2):
            if vid != -1:
                key = f"m2_{vid}"
                fused[key] = fused.get(key, 0) + 0.3 * (1 / (k + r + 1))
        for r, vid in enumerate(bm25_ids_2):
            key = f"m2_{vid}"
            fused[key] = fused.get(key, 0) + 0.7 * (1 / (k + r + 1))

        sorted_ids = sorted(fused, key=fused.get, reverse=True)

        # Now we de-duplicate the final list to avoid sending redundant chunks for reranking
        print(f"Total fused chunks before de-duplication: {len(sorted_ids)}")
        seen_core_ids = set()
        unique_sorted_ids = []
        for id_val in sorted_ids:
            # Normalize by getting the core vector ID without the model prefix
            core_id = id_val.split("_", 1)[1]
            if core_id not in seen_core_ids:
                seen_core_ids.add(core_id)
                unique_sorted_ids.append(id_val)

        print(f"Total unique fused chunks after de-duplication: {len(unique_sorted_ids)}")
        return unique_sorted_ids


def rerank_and_format_context(query: str, vector_ids: list, top_k: int, ensemble: bool = False) -> (str, list[str]):
    print(f"\nStep 2: Re-ranking {len(vector_ids)} candidates to select top {top_k}...")
    if not vector_ids:
        return "", []

    # Split ids by model for fetching
    ids1, ids2 = [], []
    if ensemble:
        for vid in vector_ids:
            if vid.startswith("m1_"):
                ids1.append(int(vid.split("_")[1]))
            elif vid.startswith("m2_"):
                ids2.append(int(vid.split("_")[1]))
    else:
        ids1 = [int(v) for v in vector_ids]

    # Fetch rows for reranking, ensuring no duplicates from different models
    rows_for_reranking = []
    if ids1:
        ph1 = ",".join("?" for _ in ids1)
        c1.execute(
            f"SELECT chunk_id, indexed_text FROM metadata WHERE vector_id IN ({ph1})",
            ids1,
        )
        rows_for_reranking.extend(c1.fetchall())
    if ensemble and ids2:
        ph2 = ",".join("?" for _ in ids2)
        c2.execute(
            f"SELECT chunk_id, indexed_text FROM metadata WHERE vector_id IN ({ph2})",
            ids2,
        )
        existing_chunk_ids = {r["chunk_id"] for r in rows_for_reranking}
        rows_for_reranking.extend([r for r in c2.fetchall() if r["chunk_id"] not in existing_chunk_ids])

    if not rows_for_reranking:
        return "", []

    # Rerank using CrossEncoder
    chunks = [{"id": row["chunk_id"], "text": row["indexed_text"]} for row in rows_for_reranking]
    pairs = [[query, ch["text"]] for ch in chunks]
    scores = model_reranker.predict(pairs, show_progress_bar=False)
    for ch, score in zip(chunks, scores):
        ch["score"] = score

    ranked = sorted(chunks, key=lambda x: x["score"], reverse=True)

    # === NEW LOGIC: DEDUPLICATION AND SELECTION ===
    # This ensures we only select unique chunks based on the highest reranked score.
    seen_chunk_ids = set()
    unique_top_chunks = []
    for ch in ranked:
        if ch["id"] not in seen_chunk_ids:
            seen_chunk_ids.add(ch["id"])
            unique_top_chunks.append(ch)
            if len(unique_top_chunks) == top_k:
                break

    if not unique_top_chunks:
        return "", []

    selected_ids = [ch["id"] for ch in unique_top_chunks]

    # Retrieve full metadata for the final context
    ph_sel = ",".join("?" for _ in selected_ids)

    final_rows = []
    # Query model1's DB
    c1.execute(f"SELECT * FROM metadata WHERE chunk_id IN ({ph_sel})", selected_ids)
    final_rows.extend(c1.fetchall())

    # Query model2's DB if in ensemble mode for any remaining chunks
    if ensemble:
        retrieved_chunk_ids = {r["chunk_id"] for r in final_rows}
        ids_to_fetch_from_c2 = [sid for sid in selected_ids if sid not in retrieved_chunk_ids]
        if ids_to_fetch_from_c2:
            ph_sel_c2 = ",".join("?" for _ in ids_to_fetch_from_c2)
            c2.execute(
                f"SELECT * FROM metadata WHERE chunk_id IN ({ph_sel_c2})",
                ids_to_fetch_from_c2,
            )
            final_rows.extend(c2.fetchall())

    # === NEW LOGIC: DOCUMENT ORDER SORTING ===
    # Sort the final chunks by their ID to ensure they are in logical document order.
    # This is critical for the LLM to understand procedural instructions.
    final_rows.sort(key=lambda r: r["chunk_id"])

    # Build context and citations
    final_context, citations = [], []
    for row in final_rows:
        part = (
            f"📄 Doc: {row['document_name']}\n"
            f"📑 Section: {row['section_id']}\n"
            f"📄 Page: {row['page_num']}\n\n"
            f"🔹 Text:\n---\n{row['indexed_text']}\n---"
        )
        if row["hyperlink_text"]:
            part += f"\n\n🔗 Links:\n{row['hyperlink_text']}"
        if row["table_text"]:
            part += f"\n\n📊 Table:\n{row['table_text']}"
        if row["header_text"] or row["footer_text"]:
            part += f"\n\n🧾 Header/Footer:\n{row['header_text']} / {row['footer_text']}"
        final_context.append(part)
        citations.append(f"Doc: {row['document_name']} | Page: {row['page_num']} | Section: {row['section_id']}")

    return "\n\n---\n\n".join(final_context), citations


# --- CLI Entrypoint ---
def main():
    parser = argparse.ArgumentParser(description="Oracle RAG v6 - Thinking Pipeline")
    parser.add_argument("--query", type=str, required=True, help="The user question to answer.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top chunks to use for context.")
    parser.add_argument("--show_chunks", action="store_true", help="Display retrieved chunks.")
    parser.add_argument(
        "--persona",
        choices=["consultant_answer", "developer_answer", "user_answer"],
        default="consultant_answer",
        help="Which persona answer to generate.",
    )
    parser.add_argument("--review", action="store_true", help="Perform optional review pass.")
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Enable multi-model retrieval (across all-MiniLM and BGE models)",
    )
    args = parser.parse_args()

    # Retrieval
    candidate_ids = hybrid_search(args.query, top_n=20, ensemble=args.ensemble)
    if not candidate_ids:
        print("No candidate documents found from hybrid search.")
        return
    context, sources = rerank_and_format_context(args.query, candidate_ids, top_k=args.top_k, ensemble=args.ensemble)
    if not context:
        print("No relevant context could be found after re-ranking.")
        return
    if args.show_chunks:
        print("\n" + "=" * 60 + "\nDETAILED CONTEXT CHUNKS\n" + "=" * 60)
        print(context)
        print("=" * 60)

    # Step 1: Thought Process
    system_thought = read_prompt_template("thought_process")
    prompt_thought = f"**User Question:**\n{args.query}\n\n---\n**Context:**\n{context}"
    thought_output = llama3_call(prompt_thought, system_thought)
    print("\n" + "=" * 20 + " Thought Process " + "=" * 20)
    print(thought_output)

    # Step 2: Persona Answer
    system_persona = read_prompt_template(args.persona)
    prompt_persona = f"Based ONLY on the reasoning below, provide the final answer:\n\n{thought_output}"
    answer_output = llama3_call(prompt_persona, system_persona)
    print("\n" + "=" * 20 + f" Answer ({args.persona}) " + "=" * 20)
    print(answer_output)

    # Step 3: Optional Review
    if args.review:
        system_review = read_prompt_template("reviewer")
        prompt_review = f"Context:\n{context}\n\nAnswer:\n{answer_output}"
        review_output = llama3_call(prompt_review, system_review)
        print("\n" + "=" * 20 + " Review ")
        print(review_output)


if __name__ == "__main__":
    main()
