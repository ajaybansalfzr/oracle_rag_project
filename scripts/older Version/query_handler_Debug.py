import argparse
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import pickle
from rank_bm25 import BM25Okapi
import nltk
import sys
from typing import Tuple, List ##, Dict, Any # Add this to your imports at the top

# import requests
# from pathlib import Path
# from nltk.tokenize import word_tokenize
# from collections import defaultdict


from scripts import config
from scripts.utils.utils import get_logger
from scripts.utils.llm_utils import llama3_call


# import logging # Add this to your imports at the top

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = get_logger(__name__)
 # Add this after your imports
# --- Setup NLTK ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

# --- Global Resources ---
MODELS = {} # Will store loaded models, indexes, and connections

def load_resources():
    """
    Loads all required models and index files for both tiers (chunk and section).
    This function is critical for the query engine to operate.
    """
    logger.info("Loading all retrieval, reranking models, and indexes...")
    try:
        MODELS['retriever'] = SentenceTransformer(config.MODEL_NAME_RETRIEVER)
        MODELS['reranker'] = CrossEncoder(config.MODEL_NAME_RERANKER)
        
        for model_name in config.EMBEDDING_MODELS:
            safe_name = model_name.replace('/', '_')
            MODELS[model_name] = {}
            
            # Load Tier 1 (Chunk) Artifacts
            MODELS[model_name]['chunk_faiss'] = faiss.read_index(str(config.EMBEDDINGS_DIR / f"chunk_index_{safe_name}.faiss"))
            with open(config.EMBEDDINGS_DIR / f"chunk_bm25_{safe_name}.pkl", "rb") as f:
                MODELS[model_name]['chunk_bm25'] = pickle.load(f)
            MODELS[model_name]['chunk_meta_conn'] = sqlite3.connect(f"file:{config.EMBEDDINGS_DIR / f'chunk_metadata_{safe_name}.db'}?mode=ro", uri=True)
            MODELS[model_name]['chunk_meta_conn'].row_factory = sqlite3.Row

            # Load Tier 2 (Section) Artifacts
            MODELS[model_name]['section_faiss'] = faiss.read_index(str(config.EMBEDDINGS_DIR / f"section_index_{safe_name}.faiss"))
            # BM25 for sections is not used in this logic but loaded for potential future use
            with open(config.EMBEDDINGS_DIR / f"section_bm25_{safe_name}.pkl", "rb") as f:
                MODELS[model_name]['section_bm25'] = pickle.load(f)
            MODELS[model_name]['section_meta_conn'] = sqlite3.connect(f"file:{config.EMBEDDINGS_DIR / f'section_metadata_{safe_name}.db'}?mode=ro", uri=True)
            MODELS[model_name]['section_meta_conn'].row_factory = sqlite3.Row
            
            logger.info(f"  - Successfully loaded artifacts for: {model_name}")

    except FileNotFoundError as e:
        logger.error(f"FATAL: A required model artifact is missing: {e.filename}")
        if config.STRICT_ENSEMBLE_MODE:
            logger.error("STRICT_ENSEMBLE_MODE is True. Exiting.")
            sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during model loading: {e}", exc_info=True)
        sys.exit(1)
    logger.info("All models and resources loaded successfully.")

# --- Add this new function above hierarchical_search ---

def expand_query_with_llm(query: str) -> List[str]:
    """
    Uses an LLM to expand a single user query into multiple variations (CR-007).
    """
    logger.info("Expanding user query with LLM...")
    prompt = read_prompt_template("query_expansion").format(question=query)
    system_prompt = "You are an expert search query generator."
    response = llama3_call(prompt, system_prompt)
    
    expanded_queries = [query] 
    generated_queries = [q.strip() for q in response.split('\n') if q.strip()]
    expanded_queries.extend(generated_queries)
    
    unique_queries = list(dict.fromkeys(expanded_queries))
    
    logger.info(f"Generated {len(unique_queries)} unique queries for search.")
    logger.debug(f"Expanded queries: {unique_queries}")
    return unique_queries

# --- Replace the old hierarchical_search function with this one ---

def hierarchical_search(query: str, ensemble: bool = True) -> List[int]:
    """
    Implements a robust multi-step search: Query Expansion -> Section Hybrid Search -> Chunk Hybrid Search.
    """
    logger.info("Executing advanced hierarchical search with query expansion...")
    
    # --- Step 1: Expand Query (CR-007) ---
    expanded_queries = expand_query_with_llm(query)
    
    fused_section_scores = {}
    models_to_search = config.EMBEDDING_MODELS if ensemble else [config.MODEL_NAME_RETRIEVER]

    # --- Step 2: Hybrid Search on Tier 2 (Sections) for each expanded query ---
    for i, q in enumerate(expanded_queries):
        logger.info(f"Running Tier 2 search for expanded query {i+1}/{len(expanded_queries)}: '{q}'")
        q_vec = MODELS['retriever'].encode([q], convert_to_tensor=True).cpu().numpy()
        q_tokens = nltk.word_tokenize(q.lower())

        for model_name in models_to_search:
            model_data = MODELS[model_name]
            
            _, faiss_top_ids = model_data['section_faiss'].search(q_vec, config.HIERARCHICAL_SEARCH_TOP_N_SECTIONS * 2)
            faiss_ranks = {int(id): rank for rank, id in enumerate(faiss_top_ids[0]) if id != -1}

            bm25_index = model_data['section_bm25']['index']
            bm25_doc_map = model_data['section_bm25']['doc_map']
            bm25_scores = bm25_index.get_scores(q_tokens)
            top_n_bm25_indices = np.argsort(bm25_scores)[::-1][:config.HIERARCHICAL_SEARCH_TOP_N_SECTIONS * 2]
            bm25_ranks = {bm25_doc_map[i]: rank for rank, i in enumerate(top_n_bm25_indices)}

            for section_id, rank in faiss_ranks.items():
                fused_section_scores[section_id] = fused_section_scores.get(section_id, 0) + (1 / (config.RRF_K + rank + 1))
            for section_id, rank in bm25_ranks.items():
                fused_section_scores[section_id] = fused_section_scores.get(section_id, 0) + (1 / (config.RRF_K + rank + 1))

    sorted_section_ids = sorted(fused_section_scores, key=fused_section_scores.get, reverse=True)
    candidate_section_ids = sorted_section_ids[:config.HIERARCHICAL_SEARCH_TOP_N_SECTIONS]

    if not candidate_section_ids:
        logger.warning("No relevant sections found after HYBRID search with query expansion. Aborting.")
        return []
    
    logger.info(f"Found {len(candidate_section_ids)} unique relevant sections after fusion.")

    # --- Step 3: Targeted Hybrid Search on Tier 1 (Chunks) ---
    logger.info("Performing targeted hybrid search on Tier 1 chunks within relevant sections.")
    fused_chunk_scores = {}
    
    main_db_conn = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    main_db_conn.row_factory = sqlite3.Row
    section_ids_as_py_int = [int(sid) for sid in candidate_section_ids]
    placeholders = ','.join('?' for _ in section_ids_as_py_int)
    
    relevant_chunks_query = f"SELECT chunk_id FROM chunks WHERE section_id IN ({placeholders})"
    candidate_chunk_ids = {row['chunk_id'] for row in main_db_conn.execute(relevant_chunks_query, section_ids_as_py_int).fetchall()}
    main_db_conn.close()

    if not candidate_chunk_ids:
        logger.warning("Tier 2 search found sections, but no associated chunks were found.")
        return []
    logger.info(f"Found {len(candidate_chunk_ids)} candidate chunks to search within.")

    original_query_vec = MODELS['retriever'].encode([query], convert_to_tensor=True).cpu().numpy()
    original_query_tokens = nltk.word_tokenize(query.lower())
    
    for model_name in models_to_search:
        model_data = MODELS[model_name]
        
        placeholders = ','.join('?' for _ in candidate_chunk_ids)
        cursor = model_data['chunk_meta_conn'].cursor()
        cursor.execute(f"SELECT chunk_id, summary FROM metadata WHERE chunk_id IN ({placeholders})", list(candidate_chunk_ids))
        
        candidate_summaries = {row['chunk_id']: row['summary'] for row in cursor.fetchall()}
        aligned_ids = list(candidate_summaries.keys())
        aligned_summaries = list(candidate_summaries.values())

        if not aligned_ids: continue

        candidate_vectors = model_data['chunk_faiss'].index.reconstruct_batch(aligned_ids)
        valid_mask = [v.any() for v in candidate_vectors]
        
        final_ids = [aid for i, aid in enumerate(aligned_ids) if valid_mask[i]]
        final_summaries = [s for i, s in enumerate(aligned_summaries) if valid_mask[i]]
        final_vectors = np.array([v for i, v in enumerate(candidate_vectors) if valid_mask[i]])

        if final_vectors.size == 0:
            faiss_ranks = {}
        else:
            temp_index = faiss.IndexFlatL2(final_vectors.shape[1])
            temp_index.add(final_vectors)
            _, temp_indices = temp_index.search(original_query_vec, min(len(final_ids), config.HIERARCHICAL_SEARCH_TOP_N_CHUNKS))
            faiss_ranks = {final_ids[i]: rank for rank, i in enumerate(temp_indices[0]) if i != -1}

        tokenized_summaries = [nltk.word_tokenize(s.lower()) for s in final_summaries]
        if not tokenized_summaries:
            bm25_ranks = {}
        else:
            temp_bm25 = BM25Okapi(tokenized_summaries)
            bm25_scores = temp_bm25.get_scores(original_query_tokens)
            sorted_indices = np.argsort(bm25_scores)[::-1]
            bm25_ranks = {final_ids[i]: rank for rank, i in enumerate(sorted_indices)}

        for chunk_id, rank in faiss_ranks.items():
            fused_chunk_scores[chunk_id] = fused_chunk_scores.get(chunk_id, 0) + (1 / (config.RRF_K + rank + 1))
        for chunk_id, rank in bm25_ranks.items():
            fused_chunk_scores[chunk_id] = fused_chunk_scores.get(chunk_id, 0) + (1 / (config.RRF_K + rank + 1))

    sorted_chunk_ids = sorted(fused_chunk_scores, key=fused_chunk_scores.get, reverse=True)
    logger.info(f"Total fused and ranked candidate chunks: {len(sorted_chunk_ids)}")
    return sorted_chunk_ids[:config.HIERARCHICAL_SEARCH_TOP_N_CHUNKS]

def rerank_and_format_context(query: str, chunk_ids: List[int], top_k: int) -> Tuple[str, List[str]]:
    """Reranks chunks and formats the final rich context for the LLM."""
    if not chunk_ids: return "", []
    logger.info(f"Step 2: Re-ranking {len(chunk_ids)} candidates to select top {top_k}...")

    # For reranking, we can use the metadata from the primary retriever model
    meta_conn = MODELS[config.MODEL_NAME_RETRIEVER]['chunk_meta_conn']
    placeholders = ','.join('?' for _ in chunk_ids)
    
    # Fetch the raw_text for reranking
    id_to_text_map = {row['chunk_id']: row['raw_text'] for row in meta_conn.execute(f"SELECT chunk_id, raw_text FROM metadata WHERE chunk_id IN ({placeholders})", tuple(chunk_ids)).fetchall()}
    
    rerank_pairs = [[query, id_to_text_map[cid]] for cid in chunk_ids if cid in id_to_text_map]
    if not rerank_pairs:
        logger.warning("No text found for candidate chunk IDs to rerank.")
        return "",[]
        
    scores = MODELS['reranker'].predict(rerank_pairs)
    
    reranked_ids_with_scores = sorted(zip(chunk_ids, scores), key=lambda x: x[1], reverse=True)
    top_chunk_ids = [cid for cid, score in reranked_ids_with_scores[:top_k]]

    # Fetch rich context for the top_k chunks
    placeholders_top = ','.join('?' for _ in top_chunk_ids)
    order_map = {cid: i for i, cid in enumerate(top_chunk_ids)}
    
    final_rows = meta_conn.execute(f"SELECT * FROM metadata WHERE chunk_id IN ({placeholders_top})", tuple(top_chunk_ids)).fetchall()
    final_rows.sort(key=lambda r: order_map[r['chunk_id']]) # Sort by reranked order

    # FR-19: Rich Context Formatting
    context_parts, citations = [], []
    for row in final_rows:
        context_parts.append(
            f"---\n"
            f"Source Document: {row['doc_name']}\n"
            f"Page: {row['page_num']}\n"
            f"Section: {row['section_header']}\n"
            f"Content: {row['raw_text']}\n"
            f"---"
        )
        citations.append(f"Source: {row['doc_name']}, Page: {row['page_num']}")
    
    return "\n\n".join(context_parts), list(dict.fromkeys(citations))

def read_prompt_template(name: str) -> str:
    path = config.PROMPTS_DIR / f"{name}.txt"
    try:
        return path.read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"Could not read prompt template '{name}': {e}")
        return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oracle RAG - Query Engine v2.0")
    parser.add_argument("--query", type=str, required=True, help="User question.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks for final context.")
    parser.add_argument("--no_ensemble", action="store_true", help="Use only the primary embedding model.")
    parser.add_argument("--persona", choices=["consultant_answer", "developer_answer", "user_answer"], default="user_answer")
    args = parser.parse_args()

    load_resources()
    
    top_chunk_ids = hierarchical_search(args.query, ensemble=(not args.no_ensemble))
    
    if not top_chunk_ids:
        logger.warning("Could not find any relevant documents for the query.")
        sys.exit()

    context, sources = rerank_and_format_context(args.query, top_chunk_ids, args.top_k)
    if not context:
        logger.warning("No relevant context found after reranking.")
        sys.exit()
    
    # FR-18: Two-Step "Thinking" LLM Pipeline
    logger.info("--- Step 3: Generating Thought Process ---")
    thought_prompt = read_prompt_template("thought_process").format(question=args.query, context=context)
    thought_process = llama3_call(thought_prompt, "You are a reasoning engine.")
    logger.info("LLM Thought Process:\n" + thought_process)

    logger.info(f"--- Step 4: Generating Final Answer (Persona: {args.persona}) ---")
    final_prompt = read_prompt_template(args.persona).format(thought_output=thought_process)
    final_answer = llama3_call(final_prompt, "You are a helpful assistant.")
    
    # Use print for the final, user-facing output
    print("\n" + "="*50)
    print("Final Answer:")
    print(final_answer)
    print("\nSources:")
    for source in sources:
        print(f"- {source}")
    print("="*50)

