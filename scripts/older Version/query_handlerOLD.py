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

# # --- Configuration ---
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# EMBEDDINGS_FOLDER = PROJECT_ROOT / "embeddings"
# PROMPT_FOLDER = PROJECT_ROOT / "prompts"
# # Model 1 resources
# DB_PATH_1 = EMBEDDINGS_FOLDER / "oracle_metadata_all-MiniLM-L6-v2.db"
# FAISS_PATH_1 = EMBEDDINGS_FOLDER / "oracle_index_all-MiniLM-L6-v2.faiss"
# BM25_PATH_1 = EMBEDDINGS_FOLDER / "oracle_bm25_all-MiniLM-L6-v2.pkl"
# # Model 2 resources
# DB_PATH_2 = EMBEDDINGS_FOLDER / "oracle_metadata_BAAI-bge-small-en-v1.5.db"
# FAISS_PATH_2 = EMBEDDINGS_FOLDER / "oracle_index_BAAI-bge-small-en-v1.5.faiss"
# BM25_PATH_2 = EMBEDDINGS_FOLDER / "oracle_bm25_BAAI-bge-small-en-v1.5.pkl"
# OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL_NAME_RETRIEVER = 'all-MiniLM-L6-v2'
# MODEL_NAME_RERANKER = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# # --- Load Models and Indexes ---
# try:
#     # print("Loading models and indexes...")
#     logger.info("Loading models and indexes...")
#     model_retriever = SentenceTransformer(MODEL_NAME_RETRIEVER)
#     model_reranker = CrossEncoder(MODEL_NAME_RERANKER)
#     # Model 1
#     index_faiss_1 = faiss.read_index(str(FAISS_PATH_1))
#     with open(BM25_PATH_1, "rb") as f:
#         bm25_data_1 = pickle.load(f)
#     index_bm25_1, bm25_doc_map_1 = bm25_data_1["index"], bm25_data_1["doc_map"]
#     conn1 = sqlite3.connect(f"file:{DB_PATH_1}?mode=ro", uri=True)
#     conn1.row_factory = sqlite3.Row
#     c1 = conn1.cursor()
#     # Model 2
#     index_faiss_2 = faiss.read_index(str(FAISS_PATH_2))
#     with open(BM25_PATH_2, "rb") as f:
#         bm25_data_2 = pickle.load(f)
#     index_bm25_2, bm25_doc_map_2 = bm25_data_2["index"], bm25_data_2["doc_map"]
#     conn2 = sqlite3.connect(f"file:{DB_PATH_2}?mode=ro", uri=True)
#     conn2.row_factory = sqlite3.Row
#     c2 = conn2.cursor()
#     logger.info("Models and indexes loaded successfully.")
# except Exception as e:
#     logger.critical(f"FATAL ERROR: Could not load models or database. Details: {e}", exc_info=True)
#     sys.exit(1)

# # --- Configuration (from FSD) ---
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
# PROMPT_FOLDER = PROJECT_ROOT / "prompts"
# DB_PATH = PROJECT_ROOT / "output" / "project_data.db"

# # System Parameters
# MODEL_NAME_RETRIEVER = 'all-MiniLM-L6-v2'
# MODEL_NAME_RERANKER = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
# OLLAMA_URL = "http://localhost:11434/api/generate"
# EMBEDDING_MODELS_LIST = ['all-MiniLM-L6-v2', 'BAAI/bge-small-en-v1.5']
# RRF_K = 60  # Reciprocal Rank Fusion k value
# STRICT_ENSEMBLE_MODE = True # As per FSD


# --- Global Resources ---
MODELS = {} # Will store loaded models, indexes, and connections

def load_resources():
    """
    Loads all required models and index files for all tiers.
    """
    logger.info("Loading all retrieval, reranking models, and indexes...")
    try:
        # Load the primary retriever and reranker models
        retriever_path = config.MODEL_PATHS[config.PRIMARY_RETRIEVER_MODEL]
        reranker_path = config.MODEL_PATHS[config.RERANKER_MODEL]
        # Load models from local paths for offline resilience
        logger.info(f"Loading retriever from local path: {retriever_path}")
        MODELS['retriever'] = SentenceTransformer(retriever_path)
        
        logger.info(f"Loading reranker from local path: {reranker_path}")
        MODELS['reranker'] = CrossEncoder(reranker_path)
        
        for model_name in config.EMBEDDING_MODELS_LIST:
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

# def load_resources():
#     """
#     Loads all required models and index files for all tiers.
#     """
#     logger.info("Loading all retrieval, reranking models, and indexes...")
#     try:
#         MODELS['retriever'] = SentenceTransformer(config.PRIMARY_RETRIEVER_MODEL)
#         MODELS['reranker'] = CrossEncoder(config.MODEL_NAME_RERANKER)
        
#         for model_name in config.EMBEDDING_MODELS_LIST:
#             safe_name = model_name.replace('/', '_')
#             MODELS[model_name] = {}
            
#             # Load Tier 1 (Chunk) Artifacts
#             MODELS[model_name]['chunk_faiss'] = faiss.read_index(str(config.EMBEDDINGS_DIR / f"chunk_index_{safe_name}.faiss"))
#             with open(config.EMBEDDINGS_DIR / f"chunk_bm25_{safe_name}.pkl", "rb") as f:
#                 MODELS[model_name]['chunk_bm25'] = pickle.load(f)
#             MODELS[model_name]['chunk_meta_conn'] = sqlite3.connect(f"file:{config.EMBEDDINGS_DIR / f'chunk_metadata_{safe_name}.db'}?mode=ro", uri=True)
#             MODELS[model_name]['chunk_meta_conn'].row_factory = sqlite3.Row

#             # Load Tier 2 (Section) Artifacts
#             MODELS[model_name]['section_faiss'] = faiss.read_index(str(config.EMBEDDINGS_DIR / f"section_index_{safe_name}.faiss"))
#             # BM25 for sections is not used in this logic but loaded for potential future use
#             with open(config.EMBEDDINGS_DIR / f"section_bm25_{safe_name}.pkl", "rb") as f:
#                 MODELS[model_name]['section_bm25'] = pickle.load(f)
#             MODELS[model_name]['section_meta_conn'] = sqlite3.connect(f"file:{config.EMBEDDINGS_DIR / f'section_metadata_{safe_name}.db'}?mode=ro", uri=True)
#             MODELS[model_name]['section_meta_conn'].row_factory = sqlite3.Row
            
#             logger.info(f"  - Successfully loaded artifacts for: {model_name}")

#     except FileNotFoundError as e:
#         logger.error(f"FATAL: A required model artifact is missing: {e.filename}")
#         if config.STRICT_ENSEMBLE_MODE:
#             logger.error("STRICT_ENSEMBLE_MODE is True. Exiting.")
#             sys.exit(1)
#     except Exception as e:
#         logger.critical(f"An unexpected error occurred during model loading: {e}", exc_info=True)
#         sys.exit(1)
#     logger.info("All models and resources loaded successfully.")


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
    return unique_queries

def hierarchical_search(query: str, ensemble: bool = True) -> List[int]:
    """
    Implements a robust multi-step search: Query Expansion -> Section Hybrid Search -> Chunk Hybrid Search.
    """
    logger.info("Executing advanced hierarchical search with query expansion...")
    
    # Step 1: Expand Query
    expanded_queries = expand_query_with_llm(query)
    
    fused_section_scores = {}
    models_to_search = config.EMBEDDING_MODELS_LIST if ensemble else [config.PRIMARY_RETRIEVER_MODEL]

    # Step 2: Hybrid Search on Sections for each expanded query
    for i, q in enumerate(expanded_queries):
        logger.info(f"Running Tier 2 search for expanded query {i+1}/{len(expanded_queries)}: '{q}'")
        q_vec = MODELS['retriever'].encode([q], convert_to_tensor=True).cpu().numpy()
        q_tokens = nltk.word_tokenize(q.lower())

        for model_name in models_to_search:
            model_data = MODELS[model_name]
            _, faiss_ids = model_data['section_faiss'].search(q_vec, config.HIERARCHICAL_SEARCH_TOP_N_SECTIONS * 2)
            faiss_ranks = {int(id): r for r, id in enumerate(faiss_ids[0]) if id != -1}

            bm25_index = model_data['section_bm25']['index']
            bm25_map = model_data['section_bm25']['doc_map']
            bm25_scores = bm25_index.get_scores(q_tokens)
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:config.HIERARCHICAL_SEARCH_TOP_N_SECTIONS * 2]
            bm25_ranks = {bm25_map[i]: r for r, i in enumerate(top_bm25_indices)}

            for sec_id, rank in faiss_ranks.items():
                fused_section_scores[sec_id] = fused_section_scores.get(sec_id, 0) + (1 / (config.RRF_K + rank + 1))
            for sec_id, rank in bm25_ranks.items():
                fused_section_scores[sec_id] = fused_section_scores.get(sec_id, 0) + (1 / (config.RRF_K + rank + 1))

    sorted_section_ids = sorted(fused_section_scores, key=fused_section_scores.get, reverse=True)
    candidate_section_ids = sorted_section_ids[:config.HIERARCHICAL_SEARCH_TOP_N_SECTIONS]

    if not candidate_section_ids:
        logger.warning("No relevant sections found after HYBRID search with query expansion. Aborting.")
        return []
    
    logger.info(f"Found {len(candidate_section_ids)} unique relevant sections after fusion.")

    # Step 3: Targeted Hybrid Search on Chunks
    logger.info("Performing targeted hybrid search on Tier 1 chunks within relevant sections.")
    fused_chunk_scores = {}
    
    main_conn = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    main_conn.row_factory = sqlite3.Row
    placeholders = ','.join('?' for _ in candidate_section_ids)
    candidate_chunk_ids = {r['chunk_id'] for r in main_conn.execute(f"SELECT chunk_id FROM chunks WHERE section_id IN ({placeholders})", candidate_section_ids).fetchall()}
    main_conn.close()

    if not candidate_chunk_ids:
        logger.warning("Tier 2 search found sections, but no associated chunks were found.")
        return []
    logger.info(f"Found {len(candidate_chunk_ids)} candidate chunks to search within.")

    original_query_vec = MODELS['retriever'].encode([query], convert_to_tensor=True).cpu().numpy()
    original_query_tokens = nltk.word_tokenize(query.lower())
    
    for model_name in models_to_search:
        model_data = MODELS[model_name]
        
        cursor = model_data['chunk_meta_conn'].cursor()
        placeholders = ','.join('?' for _ in candidate_chunk_ids)
        cursor.execute(f"SELECT chunk_id, summary FROM metadata WHERE chunk_id IN ({placeholders})", list(candidate_chunk_ids))
        
        summaries = {r['chunk_id']: r['summary'] for r in cursor.fetchall()}
        ids, summary_texts = list(summaries.keys()), list(summaries.values())
        if not ids: continue

        vectors = model_data['chunk_faiss'].index.reconstruct_batch(ids)
        valid_mask = [v.any() for v in vectors]
        
        final_ids = [id for i, id in enumerate(ids) if valid_mask[i]]
        final_summaries = [s for i, s in enumerate(summary_texts) if valid_mask[i]]
        final_vectors = np.array([v for i, v in enumerate(vectors) if valid_mask[i]])

        faiss_ranks, bm25_ranks = {}, {}
        if final_vectors.size > 0:
            temp_index = faiss.IndexFlatL2(final_vectors.shape[1])
            temp_index.add(final_vectors)
            _, temp_indices = temp_index.search(original_query_vec, min(len(final_ids), config.HIERARCHICAL_SEARCH_TOP_N_CHUNKS))
            faiss_ranks = {final_ids[i]: r for r, i in enumerate(temp_indices[0]) if i != -1}

        tokenized_summaries = [nltk.word_tokenize(s.lower()) for s in final_summaries]
        if tokenized_summaries:
            temp_bm25 = BM25Okapi(tokenized_summaries)
            bm25_scores = temp_bm25.get_scores(original_query_tokens)
            sorted_indices = np.argsort(bm25_scores)[::-1]
            bm25_ranks = {final_ids[i]: r for r, i in enumerate(sorted_indices)}

        for cid, rank in faiss_ranks.items():
            fused_chunk_scores[cid] = fused_chunk_scores.get(cid, 0) + (1 / (config.RRF_K + rank + 1))
        for cid, rank in bm25_ranks.items():
            fused_chunk_scores[cid] = fused_chunk_scores.get(cid, 0) + (1 / (config.RRF_K + rank + 1))

    sorted_chunk_ids = sorted(fused_chunk_scores, key=fused_chunk_scores.get, reverse=True)
    logger.info(f"Total fused and ranked candidate chunks: {len(sorted_chunk_ids)}")
    return sorted_chunk_ids[:config.HIERARCHICAL_SEARCH_TOP_N_CHUNKS]

# 

def rerank_and_format_context(query: str, chunk_ids: List[int], top_k: int) -> Tuple[str, List[str]]:
    """Reranks chunks and formats the final rich context for the LLM."""
    if not chunk_ids: return "", []
    logger.info(f"Step 2: Re-ranking {len(chunk_ids)} candidates to select top {top_k}...")

    # For reranking, we can use the metadata from the primary retriever model
    meta_conn = MODELS[config.PRIMARY_RETRIEVER_MODEL]['chunk_meta_conn']
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
