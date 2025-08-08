from transformers import AutoTokenizer
import argparse
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import pickle
from rank_bm25 import BM25Okapi
import nltk
import sys
import time
from typing import Tuple, List ##, Dict, Any # Add this to your imports at the top

from scripts import config
from scripts.utils.database_utils import get_db_connection
from scripts.utils.utils import get_logger,get_safe_model_name
from scripts.utils.llm_utils import llama3_call


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = get_logger(__name__)
 # Add this after your imports
# --- Setup NLTK ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)


# --- Global Resources ---
MODELS = {}
TOKENIZER = None # Will store loaded models, indexes, and connections


def load_resources():
    """
    Loads all required models from the local cache and index files for all tiers.
    """
    logger.info("Loading all retrieval, reranking models, and indexes...")
    global TOKENIZER
    try:
        # NFR-5: Load the primary retriever and reranker models from local cache
        retriever_path = config.MODEL_PATHS[config.PRIMARY_RETRIEVER_MODEL]
        reranker_path = config.MODEL_PATHS[config.RERANKER_MODEL]
        logger.info(f"Loading retriever from local path: {retriever_path}")
        MODELS['retriever'] = SentenceTransformer(retriever_path)
        logger.info(f"Loading reranker from local path: {reranker_path}")
        MODELS['reranker'] = CrossEncoder(reranker_path)
        
        for model_name in config.EMBEDDING_MODELS_LIST:
            safe_name = get_safe_model_name(model_name)
            MODELS[model_name] = {}
            
            # Load Tier 1 (Chunk) Artifacts
            MODELS[model_name]['chunk_faiss'] = faiss.read_index(str(config.EMBEDDINGS_DIR / f"chunk_index_{safe_name}.faiss"))
            with open(config.EMBEDDINGS_DIR / f"chunk_bm25_{safe_name}.pkl", "rb") as f:
                # MODELS[model_name]['chunk_bm25'] = pickle.load(f)
                bm25_data = pickle.load(f)
                MODELS[model_name]['chunk_bm25_index'] = bm25_data['index']
                MODELS[model_name]['chunk_bm25_map'] = bm25_data['doc_map']
                MODELS[model_name]['chunk_bm25_corpus'] = bm25_data['corpus']
                
            # MODELS[model_name]['chunk_meta_conn'] = sqlite3.connect(f"file:{config.EMBEDDINGS_DIR / f'chunk_metadata_{safe_name}.db'}?mode=ro", uri=True)
            # MODELS[model_name]['chunk_meta_conn'].row_factory = sqlite3.Row
            # --- START CORRECTION ---
            # Store the PATH to the metadata DB, not the connection object itself.
            MODELS[model_name]['chunk_meta_path'] = f"file:{config.EMBEDDINGS_DIR / f'chunk_metadata_{safe_name}.db'}?mode=ro"
            # --- END CORRECTION ---

            # Load Tier 2 (Section) Artifacts
            MODELS[model_name]['section_faiss'] = faiss.read_index(str(config.EMBEDDINGS_DIR / f"section_index_{safe_name}.faiss"))
            with open(config.EMBEDDINGS_DIR / f"section_bm25_{safe_name}.pkl", "rb") as f:
                # MODELS[model_name]['section_bm25'] = pickle.load(f)
                bm25_data = pickle.load(f)
                MODELS[model_name]['section_bm25_index'] = bm25_data['index']
                MODELS[model_name]['section_bm25_map'] = bm25_data['doc_map']
                MODELS[model_name]['section_bm25_corpus'] = bm25_data['corpus']

            # MODELS[model_name]['section_meta_conn'] = sqlite3.connect(f"file:{config.EMBEDDINGS_DIR / f'section_metadata_{safe_name}.db'}?mode=ro", uri=True)
            # MODELS[model_name]['section_meta_conn'].row_factory = sqlite3.Row
            # --- START CORRECTION ---
            # Store the PATH to the metadata DB, not the connection object itself.
            MODELS[model_name]['section_meta_path'] = f"file:{config.EMBEDDINGS_DIR / f'section_metadata_{safe_name}.db'}?mode=ro"
            # --- END CORRECTION ---
            
            logger.info(f"  - Successfully loaded artifacts for: {model_name}")

        logger.info("Initializing tokenizer for context packing...")
        # TOKENIZER = AutoTokenizer.from_pretrained(config.PRIMARY_RETRIEVER_MODEL, cache_dir=config.LOCAL_MODELS_CACHE_DIR)
        # Get the full local path from our MODEL_PATHS dictionary in the config
        tokenizer_path = config.MODEL_PATHS[config.PRIMARY_RETRIEVER_MODEL]
        logger.info(f"Loading tokenizer from local path: {tokenizer_path}")
        TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path)

    except FileNotFoundError as e:
        logger.error(f"FATAL: A required model artifact is missing: {e.filename}")
        if config.STRICT_ENSEMBLE_MODE:
            logger.error("STRICT_ENSEMBLE_MODE is True. Exiting.")
            sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during model loading: {e}", exc_info=True)
        sys.exit(1)
    logger.info("All models and resources loaded successfully.")


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

# In query_handler.py
# --- REPLACE the entire existing hierarchical_search() function ---
def read_prompt_template(name: str) -> str:
    path = config.PROMPTS_DIR / f"{name}.txt"
    try:
        return path.read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"Could not read prompt template '{name}': {e}")
        return ""


def hierarchical_search(query: str, ensemble: bool = True) -> List[int]:
    """
    Implements a robust and EFFICIENT multi-step search.
    This is the definitive version that corrects the FAISS reconstruct error.
    """
    logger.info("Executing advanced hierarchical search with query expansion...")
    
    # Step 1: Query Expansion
    expanded_queries = expand_query_with_llm(query)
    models_to_search = config.EMBEDDING_MODELS_LIST if ensemble else [config.PRIMARY_RETRIEVER_MODEL]
    
    # Step 2: Tier 2 Hybrid Search for Sections
    fused_section_scores = {}
    for i, q in enumerate(expanded_queries):
        logger.info(f"Running Tier 2 search for expanded query {i+1}/{len(expanded_queries)}: '{q}'")
        q_vec = MODELS['retriever'].encode([q])
        q_tokens = nltk.word_tokenize(q.lower())
        for model_name in models_to_search:
            model_data = MODELS[model_name]
            _, sec_faiss_ids = model_data['section_faiss'].search(q_vec, config.HIERARCHICAL_SEARCH_TOP_N_SECTIONS * 2)
            faiss_ranks = {int(id): r for r, id in enumerate(sec_faiss_ids[0]) if id != -1}
            
            bm25_index = model_data['section_bm25_index']
            bm25_map = model_data['section_bm25_map']
            bm25_scores = bm25_index.get_scores(q_tokens)
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:config.HIERARCHICAL_SEARCH_TOP_N_SECTIONS * 2]
            bm25_ranks = {bm25_map[i]['id']: r for r, i in enumerate(top_bm25_indices)}

            for sec_id, rank in faiss_ranks.items():
                fused_section_scores[sec_id] = fused_section_scores.get(sec_id, 0) + (1 / (config.RRF_K + rank))
            for sec_id, rank in bm25_ranks.items():
                fused_section_scores[sec_id] = fused_section_scores.get(sec_id, 0) + (1 / (config.RRF_K + rank))
    
    # top_section_ids = sorted(fused_section_scores, key=fused_section_scores.get, reverse=True)[:config.HIERARCHICAL_SEARCH_TOP_N_SECTIONS]
    # if not top_section_ids:
    #     logger.warning("No relevant sections found. Aborting.")
    #     return []
    # logger.info(f"Found {len(top_section_ids)} relevant sections to narrow search.")
    # top_section_ids = sorted(fused_section_scores, key=fused_section_scores.get, reverse=True)[:config.HIERARCHICAL_SEARCH_TOP_N_SECTIONS]

    # if not top_section_ids:
    #     logger.warning("No relevant sections found after Tier 2 search. Aborting.")
    #     return []

    # # --- THIS IS THE FIRST CRITICAL FIX ---
    # # We must verify that these top sections belong to ACTIVE documents.
    # main_conn = get_db_connection()
    # placeholders = ','.join('?' for _ in top_section_ids)
    # active_section_ids_query = f"""
    #     SELECT s.section_id FROM sections s
    #     JOIN documents d ON s.doc_id = d.doc_id
    #     WHERE s.section_id IN ({placeholders}) AND d.lifecycle_status = 'active'
    # """
    # active_top_section_ids = [row['section_id'] for row in main_conn.execute(active_section_ids_query, top_section_ids).fetchall()]
    
    # if not active_top_section_ids:
    #     logger.warning("Top sections found belong to archived documents. No valid candidates.")
    #     main_conn.close()
    #     return []
    # logger.info(f"Filtered to {len(active_top_section_ids)} relevant sections from active documents.")
    top_section_ids_unfiltered = sorted(fused_section_scores, key=fused_section_scores.get, reverse=True)[:config.HIERARCHICAL_SEARCH_TOP_N_SECTIONS]

    if not top_section_ids_unfiltered:
        logger.warning("No relevant sections found after Tier 2 search. Aborting.")
        return []

    # Step 2.5: Critical Filter for Active Documents
    # main_conn = get_db_connection()
    with get_db_connection() as main_conn:
        placeholders = ','.join('?' for _ in top_section_ids_unfiltered)
        active_section_ids_query = f"""
            SELECT s.section_id FROM sections s
            JOIN documents d ON s.doc_id = d.doc_id
            WHERE s.section_id IN ({placeholders}) AND d.lifecycle_status = 'active'
        """
        top_section_ids = [row['section_id'] for row in main_conn.execute(active_section_ids_query, top_section_ids_unfiltered).fetchall()]
    
    if not top_section_ids:
        logger.warning("Top sections found belong only to archived documents. No valid candidates.")
        main_conn.close()
        return []
    logger.info(f"Filtered to {len(top_section_ids)} relevant sections from active documents.")

    # Step 3: Get Candidate Chunk IDs
    # The query here is now against the filtered, active sections.
    placeholders = ','.join('?' for _ in top_section_ids)
    candidate_chunk_ids = {row['chunk_id'] for row in main_conn.execute(f"SELECT chunk_id FROM chunks WHERE section_id IN ({placeholders})", top_section_ids)}
    # main_conn.close()
    
    if not candidate_chunk_ids:
        logger.warning("No chunks found for the relevant active sections.")
        return []
    
    # Step 4: Targeted Hybrid Search on Chunks (using the ORIGINAL query)
    fused_chunk_scores = {}
    original_query_vec = MODELS['retriever'].encode([query])
    original_query_tokens = nltk.word_tokenize(query.lower())
    
    for model_name in models_to_search:
        model_data = MODELS[model_name]
        # FAISS Search with IDSelectorArray Filter
        search_filter = faiss.IDSelectorArray(np.array(list(candidate_chunk_ids), dtype=np.int64))
        search_params = faiss.SearchParameters(sel=search_filter)
        k_oversample = min(len(candidate_chunk_ids), config.HIERARCHICAL_SEARCH_TOP_N_CHUNKS * 5)
        _, chk_faiss_ids = model_data['chunk_faiss'].search(original_query_vec, k=k_oversample, params=search_params)
        faiss_ranks = {int(id): r for r, id in enumerate(chk_faiss_ids[0]) if id != -1}

        # Efficient BM25 Search (Score-then-Filter)
        bm25_index = model_data['chunk_bm25_index']
        bm25_doc_map = model_data['chunk_bm25_map']
        all_bm25_scores = bm25_index.get_scores(original_query_tokens)
        candidate_bm25_scores = {bm25_doc_map[i]['id']: all_bm25_scores[i] 
                               for i in range(len(all_bm25_scores)) 
                               if bm25_doc_map[i]['id'] in candidate_chunk_ids}
        sorted_bm25_chunks = sorted(candidate_bm25_scores.keys(), key=candidate_bm25_scores.get, reverse=True)
        bm25_ranks = {chunk_id: r for r, chunk_id in enumerate(sorted_bm25_chunks)}
            
        # Fusion
        for cid, rank in faiss_ranks.items():
            fused_chunk_scores[cid] = fused_chunk_scores.get(cid, 0) + (1 / (config.RRF_K + rank))
        for cid, rank in bm25_ranks.items():
            fused_chunk_scores[cid] = fused_chunk_scores.get(cid, 0) + (1 / (config.RRF_K + rank))

    sorted_chunk_ids = sorted(fused_chunk_scores, key=fused_chunk_scores.get, reverse=True)
    return sorted_chunk_ids[:config.HIERARCHICAL_SEARCH_TOP_N_CHUNKS]
#         # --- DEFINITIVE FAISS SEARCH FIX ---
#         # Use IDSelectorArray to filter the search space. This is highly efficient.
#         # We search for a larger k (e.g., k * 5) to increase the chance that the top
#         # results fall within our candidate set.
#         if candidate_chunk_ids:
#             search_filter = faiss.IDSelectorArray(np.array(list(candidate_chunk_ids), dtype=np.int64))
#             # The 'params' object is the correct way to pass a selector to a search.
#             search_params = faiss.SearchParameters(sel=search_filter)
#             # Oversample to ensure we get enough results after filtering.
#             k_oversample = min(len(candidate_chunk_ids), config.HIERARCHICAL_SEARCH_TOP_N_CHUNKS * 5)
            
#             _, chk_faiss_ids = model_data['chunk_faiss'].search(original_query_vec, k=k_oversample, params=search_params)
#             faiss_ranks = {int(id): r for r, id in enumerate(chk_faiss_ids[0]) if id != -1}
#         else:
#             faiss_ranks = {}

#         # --- EFFICIENT BM25 SEARCH (from previous review, still correct) ---
#         bm25_index = model_data['chunk_bm25_index']
#         bm25_doc_map = model_data['chunk_bm25_map']
#         all_bm25_scores = bm25_index.get_scores(original_query_tokens)
        
#         candidate_bm25_scores = {bm25_doc_map[i]['id']: all_bm25_scores[i] 
#                                for i in range(len(all_bm25_scores)) 
#                                if bm25_doc_map[i]['id'] in candidate_chunk_ids}
        
#         sorted_bm25_chunks = sorted(candidate_bm25_scores.keys(), key=candidate_bm25_scores.get, reverse=True)
#         bm25_ranks = {chunk_id: r for r, chunk_id in enumerate(sorted_bm25_chunks)}
            
#         # Fusion
#         for cid, rank in faiss_ranks.items():
#             fused_chunk_scores[cid] = fused_chunk_scores.get(cid, 0) + (1 / (config.RRF_K + rank))
#         for cid, rank in bm25_ranks.items():
#             fused_chunk_scores[cid] = fused_chunk_scores.get(cid, 0) + (1 / (config.RRF_K + rank))

#     sorted_chunk_ids = sorted(fused_chunk_scores, key=fused_chunk_scores.get, reverse=True)
#     return sorted_chunk_ids[:config.HIERARCHICAL_SEARCH_TOP_N_CHUNKS]
# # 

# def rerank_and_format_context(query: str, chunk_ids: List[int], top_k: int) -> Tuple[str, List[str]]:
#     """Reranks chunks and formats the final rich context for the LLM."""
#     if not chunk_ids: return "", []
#     logger.info(f"Step 2: Re-ranking {len(chunk_ids)} candidates to select top {top_k}...")
#     # --- START CORRECTION ---
#     # Retrieve the metadata database PATH and open a new connection within this thread.
#     meta_path = MODELS[config.PRIMARY_RETRIEVER_MODEL]['chunk_meta_path']
#     id_to_text_map = {}
#     final_rows = []

#     with sqlite3.connect(meta_path, uri=True) as meta_conn:
#         meta_conn.row_factory = sqlite3.Row
        
#         # Fetch the raw_text for reranking
#         placeholders = ','.join('?' for _ in chunk_ids)
#         query_text = f"SELECT chunk_id, raw_text FROM metadata WHERE chunk_id IN ({placeholders})"
#         id_to_text_map = {row['chunk_id']: row['raw_text'] for row in meta_conn.execute(query_text, tuple(chunk_ids)).fetchall()}
#     # --- The connection is now closed. ---
 
#     # # For reranking, we can use the metadata from the primary retriever model
#     # meta_conn = MODELS[config.PRIMARY_RETRIEVER_MODEL]['chunk_meta_conn']
#     # placeholders = ','.join('?' for _ in chunk_ids)
    
#     # # Fetch the raw_text for reranking
#     # id_to_text_map = {row['chunk_id']: row['raw_text'] for row in meta_conn.execute(f"SELECT chunk_id, raw_text FROM metadata WHERE chunk_id IN ({placeholders})", tuple(chunk_ids)).fetchall()}
    
#     rerank_pairs = [[query, id_to_text_map[cid]] for cid in chunk_ids if cid in id_to_text_map]
#     if not rerank_pairs:
#         logger.warning("No text found for candidate chunk IDs to rerank.")
#         return "",[]
        
#     scores = MODELS['reranker'].predict(rerank_pairs)
    
#     reranked_ids_with_scores = sorted(zip(chunk_ids, scores), key=lambda x: x[1], reverse=True)
#     top_chunk_ids = [cid for cid, score in reranked_ids_with_scores[:top_k]]

#      # --- Open a new connection to fetch the final, rich context. ---
#     with sqlite3.connect(meta_path, uri=True) as meta_conn:
#         meta_conn.row_factory = sqlite3.Row
#         placeholders_top = ','.join('?' for _ in top_chunk_ids)
#         order_map = {cid: i for i, cid in enumerate(top_chunk_ids)}
        
#         query_text_top = f"SELECT * FROM metadata WHERE chunk_id IN ({placeholders_top})"
#         final_rows_unordered = meta_conn.execute(query_text_top, tuple(top_chunk_ids)).fetchall()
#         final_rows = sorted(final_rows_unordered, key=lambda r: order_map[r['chunk_id']])
#     # --- The connection is now closed. ---
#     # --- END CORRECTION ---

#     # # Fetch rich context for the top_k chunks
#     # placeholders_top = ','.join('?' for _ in top_chunk_ids)
#     # order_map = {cid: i for i, cid in enumerate(top_chunk_ids)}
    
#     # final_rows = meta_conn.execute(f"SELECT * FROM metadata WHERE chunk_id IN ({placeholders_top})", tuple(top_chunk_ids)).fetchall()
#     # final_rows.sort(key=lambda r: order_map[r['chunk_id']]) # Sort by reranked order

#     # FR-19: Rich Context Formatting
#     context_parts, citations = [], []
#     for row in final_rows:
#         context_parts.append(
#             f"---\n"
#             f"Source Document: {row['doc_name']}\n"
#             f"Page: {row['page_num']}\n"
#             f"Section: {row['section_header']}\n"
#             f"Content: {row['raw_text']}\n"
#             f"---"
#         )
#         citations.append(f"Source: {row['doc_name']}, Page: {row['page_num']}")
    
#     return "\n\n".join(context_parts), list(dict.fromkeys(citations))

# def read_prompt_template(name: str) -> str:
#     path = config.PROMPTS_DIR / f"{name}.txt"
#     try:
#         return path.read_text(encoding='utf-8')
#     except Exception as e:
#         logger.error(f"Could not read prompt template '{name}': {e}")
#         return ""

# # In scripts/query_handler.py

# ... (keep all existing functions: load_resources, expand_query, hierarchical_search, etc.) ...

def rerank_and_format_context(query: str, chunk_ids: List[int], max_tokens: int) -> Tuple[str, List[str]]:
    """
    Reranks chunks, dynamically packs them into the context window up to max_tokens,
    and formats the final rich context for the LLM.
    """
    if not chunk_ids: return "", []
    
    logger.info(f"Step 2: Re-ranking {len(chunk_ids)} candidates to pack into a {max_tokens} token window...")
    
    meta_path = MODELS[config.PRIMARY_RETRIEVER_MODEL]['chunk_meta_path']
    id_to_text_map = {}
    with sqlite3.connect(meta_path, uri=True) as meta_conn:
        meta_conn.row_factory = sqlite3.Row
        placeholders = ','.join('?' for _ in chunk_ids)
        query_text = f"SELECT chunk_id, raw_text FROM metadata WHERE chunk_id IN ({placeholders})"
        id_to_text_map = {row['chunk_id']: row['raw_text'] for row in meta_conn.execute(query_text, tuple(chunk_ids)).fetchall()}

    rerank_pairs = [[query, id_to_text_map[cid]] for cid in chunk_ids if cid in id_to_text_map]
    if not rerank_pairs:
        return "", []
        
    scores = MODELS['reranker'].predict(rerank_pairs)
    reranked_ids = [cid for cid, score in sorted(zip(chunk_ids, scores), key=lambda x: x[1], reverse=True)]
    
    # --- DYNAMIC PACKING LOGIC ---
    packed_chunk_ids = []
    current_token_count = 0
    TOKEN_BUFFER = 1024 
    
    for cid in reranked_ids:
        if cid not in id_to_text_map: continue
        
        chunk_text = id_to_text_map[cid]
        chunk_token_count = len(TOKENIZER.encode(chunk_text))

        if current_token_count + chunk_token_count <= max_tokens - TOKEN_BUFFER:
            packed_chunk_ids.append(cid)
            current_token_count += chunk_token_count
        else:
            break
            
    if not packed_chunk_ids:
        return "", []

    logger.info(f"Packed {len(packed_chunk_ids)} chunks into a context of ~{current_token_count} tokens.")

    # --- Fetch rich context for the packed chunks ---
    with sqlite3.connect(meta_path, uri=True) as meta_conn:
        meta_conn.row_factory = sqlite3.Row
        placeholders_packed = ','.join('?' for _ in packed_chunk_ids)
        order_map = {cid: i for i, cid in enumerate(packed_chunk_ids)}
        query_text_packed = f"SELECT * FROM metadata WHERE chunk_id IN ({placeholders_packed})"
        final_rows_unordered = meta_conn.execute(query_text_packed, tuple(packed_chunk_ids)).fetchall()
        final_rows = sorted(final_rows_unordered, key=lambda r: order_map[r['chunk_id']])

    context_parts, citations = [], []
    for row in final_rows:
        context_parts.append(f"---\nSource Document: {row['doc_name']}\nPage: {row['page_num']}\nSection: {row['section_header']}\nContent: {row['raw_text']}\n---")
        citations.append(f"Source: {row['doc_name']}, Page: {row['page_num']}")
    
    return "\n\n".join(context_parts), list(dict.fromkeys(citations))

def _condense_question_with_history(query: str, chat_history: List[dict]) -> str:
    """
    Uses an LLM to condense chat history and a new query into a standalone question.
    """
    if len(chat_history) <= 2:
        logger.info("First query in conversation. Using it directly.")
        return query

    logger.info("Follow-up question detected. Condensing with chat history...")
    
    formatted_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[:-1]])
    
    condensation_prompt_template = read_prompt_template("query_condensation")
    prompt = condensation_prompt_template.format(chat_history=formatted_history, question=query)
    
    system_prompt = "You are a query rewriting expert..." # Keep your system prompt
    
    standalone_question = llama3_call(prompt, system_prompt)
    logger.info(f"Condensed Query: '{standalone_question}'")
    
    return standalone_question

# def query_rag_pipeline(query: str, persona: str, top_k: int, ensemble: bool) -> Tuple[str, List[str]]:
#     """
#     A single, callable function that runs the entire RAG query pipeline.
#     """
#     if 'retriever' not in MODELS:
#         logger.warning("Models not found in memory. Initializing them now.")
#         load_resources()
#     top_chunk_ids = hierarchical_search(query, ensemble=ensemble)
    
#     if not top_chunk_ids:
#         logger.warning("Could not find any relevant documents for the query.")
#         return "I could not find any relevant documents to answer your question.", []

#     context, sources = rerank_and_format_context(query, top_chunk_ids, top_k)
#     if not context:
#         logger.warning("No relevant context found after reranking.")
#         return "I found some potentially relevant documents, but could not extract a clear answer after re-ranking.", []

#     logger.info("--- Generating Thought Process ---")
#     thought_prompt = read_prompt_template("thought_process").format(question=query, context=context)
#     thought_process = llama3_call(thought_prompt, "You are a reasoning engine.")
#     logger.info("LLM Thought Process:\n" + thought_process)
    
#     # --- START OF CRITICAL FIX ---
#     # If the thought process generation failed or returned empty, stop here.
#     if not thought_process or "[LLM Error" in thought_process:
#         logger.error("Failed to generate a thought process. Aborting final answer generation.")
#         return "I apologize, but I encountered an error while trying to reason about your question. Please try again.", []

#     # Add a brief "cool-down" period to allow the LLM server to release resources.
#     logger.info("Pausing for 2 seconds to allow server resources to cool down before final generation.")
#     time.sleep(2) 
#     # --- END OF CRITICAL FIX ---

#     logger.info(f"--- Generating Final Answer (Persona: {persona}) ---")
#     final_prompt = read_prompt_template(persona).format(thought_output=thought_process)
#     final_answer = llama3_call(final_prompt, "You are a helpful assistant.")
    
#     return final_answer, sources

def query_rag_pipeline(query: str, chat_history: list, persona: str, ensemble: bool) -> Tuple[str, List[str]]:
    """
    A single, callable function that runs the entire RAG query pipeline,
    now with conversational memory and dynamic context packing.
    """
    if 'retriever' not in MODELS or TOKENIZER is None:
        logger.warning("Models or Tokenizer not found in memory. Initializing them now.")
        load_resources()
        
    # STEP 1: Implement conversational memory by creating a standalone question.
    standalone_question = _condense_question_with_history(query, chat_history)

    # STEP 2: Use the standalone question for the entire retrieval process.
    top_chunk_ids = hierarchical_search(standalone_question, ensemble=ensemble)
    
    if not top_chunk_ids:
        logger.warning("Could not find any relevant documents for the query.")
        return "I could not find any relevant documents to answer your question.", []

    # STEP 3: Use dynamic packing instead of a fixed top_k.
    # We pass the standalone question for better reranking scores.
    context, sources = rerank_and_format_context(standalone_question, top_chunk_ids, max_tokens=8192)
    
    if not context:
        logger.warning("No relevant context found after reranking.")
        return "I found some potentially relevant documents, but could not extract a clear answer after re-ranking.", []

    # STEP 4: Use the standalone question for the generation steps.
    logger.info("--- Generating Thought Process ---")
    thought_prompt = read_prompt_template("thought_process").format(question=standalone_question, context=context)
    thought_process = llama3_call(thought_prompt, "You are a reasoning engine.")
    logger.info("LLM Thought Process:\n" + thought_process)
    
    if not thought_process or "[LLM Error" in thought_process:
        logger.error("Failed to generate a thought process. Aborting final answer generation.")
        return "I apologize, but I encountered an error while trying to reason about your question. Please try again.", []

    logger.info("Pausing for 2 seconds to allow server resources to cool down before final generation.")
    time.sleep(2) 

    logger.info(f"--- Generating Final Answer (Persona: {persona}) ---")
    final_prompt = read_prompt_template(persona).format(thought_output=thought_process)
    final_answer = llama3_call(final_prompt, "You are a helpful assistant.")
    
    return final_answer, sources

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oracle RAG - Query Engine v2.0")
    parser.add_argument("--query", type=str, required=True, help="User question.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks for final context.")
    parser.add_argument("--no_ensemble", action="store_true", help="Use only the primary embedding model.")
    parser.add_argument("--persona", choices=["consultant_answer", "developer_answer", "user_answer"], default="user_answer")
    args = parser.parse_args()

    load_resources()
    # persona_arg = args.persona
    # ensemble_flag = not args.no_ensemble
    final_answer, sources = query_rag_pipeline(
        query=args.query,
        chat_history=[], # Add this for command-line compatibility
        persona=args.persona,
        # top_k=args.top_k,
        ensemble=(not args.no_ensemble)
    )
    
    # Check if the pipeline returned an error message
    if not sources and "I could not find" in final_answer:
        logger.warning(final_answer)
        sys.exit(0)
    # final_answer, sources = query_rag_pipeline(args.query, persona_arg, args.top_k, ensemble_flag)
    # Use print for the final, user-facing output
    print("\n" + "="*50)
    print("Final Answer:")
    print(final_answer)
    print("\nSources:")
    for source in sources:
        print(f"- {source}")
    print("="*50)
