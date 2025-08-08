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
# from collections import defaultdict

from typing import Tuple, List, Dict, Any # Add this to your imports at the top
import logging # Add this to your imports at the top

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
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

# --- Configuration (from FSD) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
PROMPT_FOLDER = PROJECT_ROOT / "prompts"
DB_PATH = PROJECT_ROOT / "output" / "project_data.db"

# System Parameters
MODEL_NAME_RETRIEVER = 'all-MiniLM-L6-v2'
MODEL_NAME_RERANKER = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
OLLAMA_URL = "http://localhost:11434/api/generate"
EMBEDDING_MODELS = ['all-MiniLM-L6-v2', 'BAAI/bge-small-en-v1.5']
RRF_K = 60  # Reciprocal Rank Fusion k value
STRICT_ENSEMBLE_MODE = True # As per FSD

# --- Global Resources ---
MODELS = {} # Will store loaded models and indexes

def load_resources():
    """Loads all models, indexes, and DB connections into the global MODELS dict."""
    logger.info("Loading all retrieval and reranking models...")
    try:
        MODELS['retriever'] = SentenceTransformer(MODEL_NAME_RETRIEVER)
        MODELS['reranker'] = CrossEncoder(MODEL_NAME_RERANKER)
        
        for model_name in EMBEDDING_MODELS:
            safe_name = model_name.replace('/', '_')
            faiss_path = EMBEDDINGS_DIR / f"oracle_index_{safe_name}.faiss"
            meta_db_path = EMBEDDINGS_DIR / f"oracle_metadata_{safe_name}.db"
            bm25_path = EMBEDDINGS_DIR / f"oracle_bm25_{safe_name}.pkl"

            MODELS[model_name] = {
                'faiss_index': faiss.read_index(str(faiss_path)),
                'meta_conn': sqlite3.connect(f"file:{meta_db_path}?mode=ro", uri=True),
            }
            with open(bm25_path, "rb") as f:
                MODELS[model_name]['bm25_index'] = pickle.load(f)

            MODELS[model_name]['meta_conn'].row_factory = sqlite3.Row
            logger.info(f"  - Successfully loaded artifacts for: {model_name}")

    except FileNotFoundError as e:
        logger.error(f"FATAL: A required model artifact is missing: {e.filename}")
        if STRICT_ENSEMBLE_MODE:
            logger.error("STRICT_ENSEMBLE_MODE is True. Exiting.")
            sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during model loading: {e}", exc_info=True)
        sys.exit(1)
    logger.info("All models and resources loaded successfully.")

# --- Helper Functions ---
def clean_answer(text: str) -> str:
    return text.strip()

def read_prompt_template(name: str) -> str:
    path = PROMPT_FOLDER / f"{name}.txt"
    try:
        return path.read_text(encoding='utf-8')
    except Exception:
        return ""

def llama3_call(prompt: str, system: str) -> str:
    try:
        res = requests.post(
            OLLAMA_URL,
            json={"model": "llama3", "prompt": prompt, "system": system, "stream": False},
            timeout=120
        )
        res.raise_for_status()
        return clean_answer(res.json().get("response", ""))
    except Exception as e:
        logger.error(f"LLM call failed: {e}", exc_info=True)
        return f"[LLM Error: {e}]"
        # return f"[LLM Error: {e}]"

# # --- RAG Retrieval Functions ---
# def hybrid_search(query: str, top_n: int = 20, ensemble: bool = False) -> list[str]:
#     print(f"\nStep 1: Performing {'Ensemble' if ensemble else 'Hybrid'} Search for query: '{query}'")
#     # Encode query
#     query_vec = model_retriever.encode([query])
#     tokens = word_tokenize(query.lower())
#     fused = {}
    
#     # --- Model 1 (all-MiniLM-L6-v2) Search ---
#     faiss_ids_1 = index_faiss_1.search(query_vec, top_n)[1][0]
#     bm25_scores_1 = index_bm25_1.get_scores(tokens)
#     top_bm25_idx_1 = np.argsort(bm25_scores_1)[::-1][:top_n]
#     bm25_ids_1 = []
#     if top_bm25_idx_1.size:
#         chunk_ids = [bm25_doc_map_1[i] for i in top_bm25_idx_1]
#         placeholders = ','.join('?' for _ in chunk_ids)
#         c1.execute(f"SELECT vector_id FROM metadata WHERE chunk_id IN ({placeholders})", chunk_ids)
#         bm25_ids_1 = [row['vector_id'] for row in c1.fetchall()]
    
#     print(f"FAISS/BM25 scores from Model 1: FAISS={len([v for v in faiss_ids_1 if v != -1])}, BM25={len(bm25_ids_1)}")

#     # --- Logic for Fusing Results ---
#     k = 60 # RRF constant

#     if not ensemble:
#         # --- SINGLE MODEL (HYBRID) FUSION ---
#         # Fuse results from Model 1 only
#         for r, vid in enumerate(faiss_ids_1):
#             if vid != -1:
#                 fused[str(vid)] = fused.get(str(vid), 0) + 0.3 * (1 / (k + r + 1))
#         for r, vid in enumerate(bm25_ids_1):
#             fused[str(vid)] = fused.get(str(vid), 0) + 0.7 * (1 / (k + r + 1))
    
#     else:
#         # --- ENSEMBLE (TWO MODEL) FUSION ---
#         # Add Model 1 results to the fusion list with model prefix
#         for r, vid in enumerate(faiss_ids_1):
#             if vid != -1:
#                 key = f"m1_{vid}"
#                 fused[key] = fused.get(key, 0) + 0.3 * (1 / (k + r + 1))
#         for r, vid in enumerate(bm25_ids_1):
#             key = f"m1_{vid}"
#             fused[key] = fused.get(key, 0) + 0.7 * (1 / (k + r + 1))

#         # --- Model 2 (BAAI-bge-small-en-v1.5) Search ---
#         faiss_ids_2 = index_faiss_2.search(query_vec, top_n)[1][0]
#         bm25_scores_2 = index_bm25_2.get_scores(tokens)
#         top_bm25_idx_2 = np.argsort(bm25_scores_2)[::-1][:top_n]
#         bm25_ids_2 = []
#         if top_bm25_idx_2.size:
#             chunk_ids2 = [bm25_doc_map_2[i] for i in top_bm25_idx_2]
#             placeholders2 = ','.join('?' for _ in chunk_ids2)
#             c2.execute(f"SELECT vector_id FROM metadata WHERE chunk_id IN ({placeholders2})", chunk_ids2)
#             bm25_ids_2 = [row['vector_id'] for row in c2.fetchall()]

#         print(f"FAISS/BM25 scores from Model 2: FAISS={len([v for v in faiss_ids_2 if v != -1])}, BM25={len(bm25_ids_2)}")

#         # Add Model 2 results to the fusion list
#         for r, vid in enumerate(faiss_ids_2):
#             if vid != -1:
#                 key = f"m2_{vid}"
#                 fused[key] = fused.get(key, 0) + 0.3 * (1 / (k + r + 1))
#         for r, vid in enumerate(bm25_ids_2):
#             key = f"m2_{vid}"
#             fused[key] = fused.get(key, 0) + 0.7 * (1 / (k + r + 1))
            
#         sorted_ids = sorted(fused, key=fused.get, reverse=True)
        
#         # Now we de-duplicate the final list to avoid sending redundant chunks for reranking
#         print(f"Total fused chunks before de-duplication: {len(sorted_ids)}")
#         return sorted_ids

# --- RAG Retrieval Functions ---
# def hybrid_search(query: str, top_n: int = 20, ensemble: bool = False) -> list[str]:
#     # print(f"\nStep 1: Performing {'Ensemble' if ensemble else 'Hybrid'} Search for query: '{query}'")
#     logger.info(f"Step 1: Performing {'Ensemble' if ensemble else 'Hybrid'} Search for query: '{query}'")
#     query_vec = model_retriever.encode([query])
#     tokens = word_tokenize(query.lower())
#     fused = Dict[str, float] = {}
#     k = 60  # RRF constant

#     # --- Model 1 Search ---
#     _, faiss_vec_ids_1 = index_faiss_1.search(query_vec, top_n)
#     faiss_vec_ids_1 = faiss_vec_ids_1[0] # Get the actual list of IDs
#     bm25_scores_1 = index_bm25_1.get_scores(tokens)
#     top_bm25_indices_1 = np.argsort(bm25_scores_1)[::-1][:top_n]
#     bm25_chunk_ids_1 = [bm25_doc_map_1[i] for i in top_bm25_indices_1]
    
#     # print(f"FAISS/BM25 results from Model 1: FAISS={len(faiss_vec_ids_1)}, BM25={len(bm25_chunk_ids_1)}")
#     logger.info(f"Model 1 results: FAISS={len(faiss_vec_ids_1)}, BM25={len(bm25_chunk_ids_1)}")

#     # --- Fusion Logic ---
#     # Add FAISS results
#     for r, vec_id in enumerate(faiss_vec_ids_1):
#         if vec_id != -1:
#             key = f"m1_{vec_id}" if ensemble else str(vec_id)
#             fused[key] = fused.get(key, 0) + (1 / (k + r + 1))
    
#     # # Add BM25 results
#     # placeholders = ','.join('?' for _ in bm25_chunk_ids_1)
#     # c1.execute(f"SELECT vector_id, chunk_id FROM metadata WHERE chunk_id IN ({placeholders})", bm25_chunk_ids_1)
#     # for r, row in enumerate(c1.fetchall()):
#     #     vec_id = row['vector_id']
#     #     key = f"m1_{vec_id}" if ensemble else str(vec_id)
#     #     fused[key] = fused.get(key, 0) + (1 / (k + r + 1))

#     if bm25_chunk_ids_1:
#         placeholders = ','.join('?' for _ in bm25_chunk_ids_1)
#         c1.execute(f"SELECT vector_id FROM metadata WHERE chunk_id IN ({placeholders})", bm25_chunk_ids_1)
#         # --- FIX 1: Use a separate counter `j` for BM25 results ---
#         for j, row in enumerate(c1.fetchall()):
#             vec_id = row['vector_id']
#             key = f"m1_{vec_id}" if ensemble else str(vec_id)
#             fused[key] = fused.get(key, 0) + (1 / (k + j + 1))

#     # --- Model 2 Search (if ensemble) ---
#     if ensemble:
#         _, faiss_vec_ids_2 = index_faiss_2.search(query_vec, top_n)
#         faiss_vec_ids_2 = faiss_vec_ids_2[0]
#         bm25_scores_2 = index_bm25_2.get_scores(tokens)
#         top_bm25_indices_2 = np.argsort(bm25_scores_2)[::-1][:top_n]
#         bm25_chunk_ids_2 = [bm25_doc_map_2[i] for i in top_bm25_indices_2]
#         # print(f"FAISS/BM25 results from Model 2: FAISS={len(faiss_vec_ids_2)}, BM25={len(bm25_chunk_ids_2)}")
#         logger.info(f"Model 2 results: FAISS={len(faiss_vec_ids_2)}, BM25={len(bm25_chunk_ids_2)}")

#         for r, vec_id in enumerate(faiss_vec_ids_2):
#             if vec_id != -1:
#                 key = f"m2_{vec_id}"
#                 fused[key] = fused.get(key, 0) + (1 / (k + r + 1))

#         # placeholders = ','.join('?' for _ in bm25_chunk_ids_2)
#         # c2.execute(f"SELECT vector_id, chunk_id FROM metadata WHERE chunk_id IN ({placeholders})", bm25_chunk_ids_2)
#         # for r, row in enumerate(c2.fetchall()):
#         #     vec_id = row['vector_id']
#         #     key = f"m2_{vec_id}"
#         #     fused[key] = fused.get(key, 0) + (1 / (k + r + 1))
#         if bm25_chunk_ids_2:
#             placeholders = ','.join('?' for _ in bm25_chunk_ids_2)
#             c2.execute(f"SELECT vector_id FROM metadata WHERE chunk_id IN ({placeholders})", bm25_chunk_ids_2)
#             for j, row in enumerate(c2.fetchall()):
#                 vec_id = row['vector_id']
#                 key = f"m2_{vec_id}"
#                 fused[key] = fused.get(key, 0) + (1 / (k + j + 1))

#     sorted_fused_ids = sorted(fused.keys(), key=fused.get, reverse=True)
#     print(f"Total fused candidate IDs to be reranked: {len(sorted_fused_ids)}")
#     return sorted_fused_ids

# # def rerank_and_format_context(query: str, vector_ids: list, top_k: int, ensemble: bool = False) -> (str, list[str]):
# #     print(f"\nStep 2: Re-ranking {len(vector_ids)} candidates to select top {top_k}...")
# #     if not vector_ids:
# #         return "", []

# def rerank_and_format_context(query: str, vector_ids: List[str], top_k: int, ensemble: bool = False) -> Tuple[str, List[str]]:
#     logger.info(f"Step 2: Re-ranking {len(vector_ids)} candidates to select top {top_k}...")
#     if not vector_ids:
#         return "", []
    
#     # --- Step 1: Bulk-Fetch candidate text for reranking ---
#     vec_ids_m1 = [int(vid.split('_')[1]) for vid in vector_ids if vid.startswith('m1_')]
#     vec_ids_m2 = [int(vid.split('_')[1]) for vid in vector_ids if vid.startswith('m2_')]
#     if not ensemble:
#         vec_ids_m1.extend([int(vid) for vid in vector_ids if not vid.startswith('m')])

#     # --- FIX 3: Guard against empty ID lists ---
#     if not vec_ids_m1 and not vec_ids_m2: return "", []

#     rerank_candidates: List[Dict[str, Any]] = []
#     try:
#         if vec_ids_m1:
#             placeholders = ','.join('?' for _ in vec_ids_m1)
#             c1.execute(f"SELECT chunk_id, indexed_text FROM metadata WHERE vector_id IN ({placeholders})", vec_ids_m1)
#             for row in c1.fetchall():
#                 rerank_candidates.append({'chunk_id': row['chunk_id'], 'text': row['indexed_text'], 'model': 'm1'})
#         if vec_ids_m2:
#             placeholders = ','.join('?' for _ in vec_ids_m2)
#             c2.execute(f"SELECT chunk_id, indexed_text FROM metadata WHERE vector_id IN ({placeholders})", vec_ids_m2)
#             for row in c2.fetchall():
#                 rerank_candidates.append({'chunk_id': row['chunk_id'], 'text': row['indexed_text'], 'model': 'm2'})
#     except sqlite3.Error as e:
#         logger.error(f"Database error while fetching candidates for reranking: {e}", exc_info=True)
#         return "", []
#     # --- Step 2: Rerank the candidates ---
#     if not rerank_candidates: return "", []
#     pairs = [[query, cand['text']] for cand in rerank_candidates]
#     scores = model_reranker.predict(pairs, show_progress_bar=False)
#     for cand, score in zip(rerank_candidates, scores):
#         cand['score'] = score
    
#     ranked = sorted(rerank_candidates, key=lambda x: x['score'], reverse=True)
#     top_chunks = ranked[:top_k]
#     if not top_chunks: return "", []

#     # --- Step 3: Bulk-Fetch the full, rich metadata for ONLY the top chunks ---
#     final_rows = []
#     chunk_ids_m1 = [c['chunk_id'] for c in top_chunks if c['model'] == 'm1']
#     chunk_ids_m2 = [c['chunk_id'] for c in top_chunks if c['model'] == 'm2']

#     try:
#         if chunk_ids_m1:
#             placeholders = ','.join('?' for _ in chunk_ids_m1)
#             c1.execute(f"SELECT * FROM metadata WHERE chunk_id IN ({placeholders})", chunk_ids_m1)
#             final_rows.extend(c1.fetchall())
#         if chunk_ids_m2:
#             placeholders = ','.join('?' for _ in chunk_ids_m2)
#             c2.execute(f"SELECT * FROM metadata WHERE chunk_id IN ({placeholders})", chunk_ids_m2)
#             final_rows.extend(c2.fetchall())
#     except sqlite3.Error as e:
#         logger.error(f"Database error while fetching final context: {e}", exc_info=True)
#         return "", []
        
#     # --- FIX 2: Avoid O(N²) sort by creating an order map ---
#     top_chunk_ids_ordered = [c['chunk_id'] for c in top_chunks]
#     order_map = {cid: i for i, cid in enumerate(top_chunk_ids_ordered)}
#     final_rows.sort(key=lambda r: order_map.get(r['chunk_id'], float('inf')))

#     # --- Step 4: Format the final context ---
#     final_context_parts = []
#     citations = []
#     for row in final_rows:
#         part = (
#             f" Doc: {row['document_name']}\n"
#             f" Section: {row['section_id']}\n"
#             f" Page: {row['page_num']}\n\n"
#             f" Text (Summary):\n---\n{row['indexed_text']}\n---"
#         )
#         if row["hyperlink_text"]: part += f"\n\n Links: {row['hyperlink_text']}"
#         if row["table_text"]: part += f"\n\n Table:\n{row['table_text']}"
#         if row["header_text"] or row["footer_text"]: part += f"\n\n🧾 Header/Footer: {row['header_text']} / {row['footer_text']}"
#         final_context_parts.append(part)
#         citations.append(f"Doc: {row['document_name']} | Page: {row['page_num']}")

#     return "\n\n".join(final_context_parts), citations

# def hierarchical_hybrid_search(query: str, ensemble: bool = False, top_n_sections: int = 5, top_n_chunks: int = 10) -> List[int]:
#     """
#     FR-23: Performs a two-step hierarchical search.
#     1. Broad search on section summaries to find relevant sections.
#     2. Focused hybrid search on chunks within those sections.
#     """
#     logger.info(f"Step 1a: Performing hierarchical search to find top {top_n_sections} sections...")
#     query_vec = MODELS['retriever'].encode([query])
    
#     # For simplicity, we'll use the first configured model's index for the section-level search.
#     # A more advanced implementation might search all and fuse the results.
#     main_model_name = EMBEDDING_MODELS[0]
    
#     # This is a placeholder for section-level search. We are querying the chunk metadata for now.
#     # In a full FR-9 implementation, you would have a separate FAISS index for section summaries.
#     # For now, we simulate by finding top chunks and then getting their unique parent sections.
#     _, top_chunk_ids = MODELS[main_model_name]['faiss_index'].search(query_vec, top_n_sections * 5)
    
#     meta_conn = MODELS[main_model_name]['meta_conn']
#     placeholders = ','.join('?' for _ in top_chunk_ids[0])
    
#     # We are using chunk_id which is equivalent to vector_id in our current setup
#     # A real implementation would query a sections table in the metadata DB
#     # query_for_sections = f"SELECT DISTINCT section_id FROM metadata WHERE chunk_id IN ({placeholders})"
#     # relevant_section_ids = [row['section_id'] for row in meta_conn.execute(query_for_sections, tuple(top_chunk_ids[0])).fetchall()]
#     # CORRECTED: Query the correct column name 'section_header'
#     query_for_sections = f"SELECT DISTINCT section_header FROM metadata WHERE chunk_id IN ({placeholders})"
#     relevant_section_ids = [row['section_header'] for row in meta_conn.execute(query_for_sections, tuple(top_chunk_ids[0])).fetchall()]


#     if not relevant_section_ids:
#         logger.warning("No relevant sections found in the first pass.")
#         return []
#     logger.info(f"Found {len(relevant_section_ids)} relevant parent sections. Now performing focused search...")

#     # Step 2: Focused Hybrid Search within the identified sections
#     fused_scores = {}
#     query_tokens = query.lower().split()

#     models_to_search = EMBEDDING_MODELS if ensemble else [main_model_name]

#     for model_name in models_to_search:
#         model_data = MODELS[model_name]
#         meta_conn = model_data['meta_conn']
        
#         # Get all chunk_ids that belong to the relevant sections
#         section_placeholders = ','.join('?' for _ in relevant_section_ids)
        
#         # This is where the filtering happens
#         # sql = f"SELECT chunk_id, summary FROM metadata WHERE section_id IN ({section_placeholders})"
#         # candidate_chunks = {row['chunk_id']: row['summary'] for row in meta_conn.execute(sql, tuple(relevant_section_ids)).fetchall()}
#         # CORRECTED: Filter by the correct column name 'section_header'
#         sql = f"SELECT chunk_id, summary FROM metadata WHERE section_header IN ({section_placeholders})"
#         candidate_chunks = {row['chunk_id']: row['summary'] for row in meta_conn.execute(sql, tuple(relevant_section_ids)).fetchall()}
        
#         if not candidate_chunks:
#             continue

#         candidate_ids = list(candidate_chunks.keys())
#         candidate_summaries = list(candidate_chunks.values())
        
#         # FAISS search on filtered candidates
#         candidate_embeddings = model_data['faiss_index'].reconstruct_batch(candidate_ids)
#         # Using faiss to search within a subset is complex; we'll simulate by filtering results post-search for this implementation
#         # A more performant way is to build a sub-index, but that's beyond this scope.
        
#         # Simplified: Perform a global search and keep only the candidates we care about
#         _, D, I = MODELS['retriever'].search(query_vec, len(candidate_ids)) # Search all
#         faiss_results = {int(I[0][i]): (i+1) for i in range(len(I[0])) if int(I[0][i]) in candidate_ids}

#         # BM25 search on filtered candidates
#         tokenized_summaries = [s.split() for s in candidate_summaries]
#         bm25_index = BM25Okapi(tokenized_summaries)
#         bm25_scores = bm25_index.get_scores(query_tokens)
#         bm25_ranked_indices = np.argsort(bm25_scores)[::-1]
#         bm25_results = {candidate_ids[i]: (rank+1) for rank, i in enumerate(bm25_ranked_indices)}

#         # FR-17: Reciprocal Rank Fusion
#         for chunk_id, rank in faiss_results.items():
#             fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + (1 / (RRF_K + rank))
#         for chunk_id, rank in bm25_results.items():
#             fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + (1 / (RRF_K + rank))

#     sorted_chunk_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
#     logger.info(f"Total fused and ranked candidate chunks: {len(sorted_chunk_ids)}")
#     return sorted_chunk_ids[:top_n_chunks]

def fused_hybrid_search(query: str, ensemble: bool = False, top_n: int = 20) -> List[int]:
    """
    Performs hybrid search (FAISS + BM25) and fuses results using RRF.
    This corrected function replaces the flawed hierarchical search for better reliability.
    """
    logger.info(f"Step 1: Performing {'Ensemble' if ensemble else 'Single Model'} Hybrid Search...")
    query_vec = MODELS['retriever'].encode([query])
    query_tokens = query.lower().split()
    
    fused_scores = {}
    models_to_search = EMBEDDING_MODELS if ensemble else [EMBEDDING_MODELS[0]]

    for model_name in models_to_search:
        model_data = MODELS[model_name]
        faiss_index = model_data['faiss_index']
        bm25_index = model_data['bm25_index'] # This is the BM25Okapi object
        
        # FAISS Search
        _, faiss_ids = faiss_index.search(np.array(query_vec, dtype=np.float32), top_n)
        faiss_ids = faiss_ids[0]

        # BM25 Search
        # The BM25 index was built on the summaries. We need to fetch them to search.
        meta_conn = model_data['meta_conn']
        all_summaries = {row['chunk_id']: row['summary'] for row in meta_conn.execute("SELECT chunk_id, summary FROM metadata").fetchall()}
        
        corpus = list(all_summaries.values())
        corpus_ids = list(all_summaries.keys())
        
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(query_tokens)
        
        # Get the chunk_ids of the top N results
        top_n_bm25_indices = np.argsort(bm25_scores)[::-1][:top_n]
        bm25_ids = [corpus_ids[i] for i in top_n_bm25_indices]
        
        logger.info(f"  - Model '{model_name}' initial results: FAISS={len(faiss_ids)}, BM25={len(bm25_ids)}")

        # FR-17: Reciprocal Rank Fusion
        for rank, doc_id in enumerate(faiss_ids):
            if doc_id != -1:
                fused_scores[int(doc_id)] = fused_scores.get(int(doc_id), 0) + (1 / (RRF_K + rank + 1))
        
        for rank, chunk_id in enumerate(bm25_ids):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + (1 / (RRF_K + rank + 1))

    sorted_chunk_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    logger.info(f"Total fused and ranked candidate chunks: {len(sorted_chunk_ids)}")
    return sorted_chunk_ids[:top_n]

def rerank_and_format_context(query: str, chunk_ids: List[int], top_k: int) -> Tuple[str, List[str]]:
    """
    FR-5 & FR-19: Reranks chunks and formats the final rich context.
    """
    if not chunk_ids: return "", []
    logger.info(f"Step 2: Re-ranking top {len(chunk_ids)} candidates to select {top_k}...")

    # We need to fetch text from any of the metadata DBs, as they are duplicated
    meta_conn = MODELS[EMBEDDING_MODELS[0]]['meta_conn']
    placeholders = ','.join('?' for _ in chunk_ids)
    sql = f"SELECT chunk_id, raw_text FROM metadata WHERE chunk_id IN ({placeholders})"
    
    # Create a map to preserve order after fetching
    id_to_text_map = {row['chunk_id']: row['raw_text'] for row in meta_conn.execute(sql, tuple(chunk_ids)).fetchall()}
    
    # Rerank
    rerank_pairs = [[query, id_to_text_map[cid]] for cid in chunk_ids if cid in id_to_text_map]
    scores = MODELS['reranker'].predict(rerank_pairs)
    
    reranked_ids = [chunk_ids[i] for i in np.argsort(scores)[::-1]]
    top_chunk_ids = reranked_ids[:top_k]

    # Fetch rich context for the top_k chunks
    placeholders_top = ','.join('?' for _ in top_chunk_ids)
    sql_top = f"SELECT * FROM metadata WHERE chunk_id IN ({placeholders_top})"
    
    # Create an order map to sort the final results by reranked score
    order_map = {cid: i for i, cid in enumerate(top_chunk_ids)}
    
    final_rows = meta_conn.execute(sql_top, tuple(top_chunk_ids)).fetchall()
    final_rows.sort(key=lambda r: order_map[r['chunk_id']]) # Sort by reranked order

    # FR-19: Rich Context Formatting
    context_parts, citations = [], []
    for row in final_rows:
        context_parts.append(
            f"--- START OF CONTEXT ---\n"
            f"Document: {row['doc_name']}\n"
            f"Page: {row['page_num']}\n"
            f"Section: {row['section_header']}\n"
            f"Content: {row['raw_text']}\n"
            f"Summary: {row['summary']}\n"
            f"--- END OF CONTEXT ---"
        )
        citations.append(f"Source: {row['doc_name']}, Page: {row['page_num']}")
    
    return "\n\n".join(context_parts), list(dict.fromkeys(citations))

# # --- CLI Entrypoint ---
# def main():
#     parser = argparse.ArgumentParser(description="Oracle RAG v6 - Thinking Pipeline")
#     parser.add_argument("--query", type=str, required=True, help="The user question to answer.")
#     parser.add_argument("--top_k", type=int, default=3, help="Number of top chunks to use for context.")
#     parser.add_argument("--show_chunks", action="store_true", help="Display retrieved chunks.")
#     parser.add_argument("--persona", choices=["consultant_answer", "developer_answer", "user_answer"], default="consultant_answer", help="Which persona answer to generate.")
#     parser.add_argument("--review", action="store_true", help="Perform optional review pass.")
#     parser.add_argument("--ensemble", action="store_true", help="Enable multi-model retrieval.")
#     args = parser.parse_args()

#     logger.info("--- Starting RAG Pipeline ---")
#     candidate_ids = hybrid_search(args.query, top_n=20, ensemble=args.ensemble)
#     if not candidate_ids:
#         logger.warning("No candidate documents found from hybrid search.")
#         return

#     context, sources = rerank_and_format_context(args.query, candidate_ids, top_k=args.top_k, ensemble=args.ensemble)
#     if not context:
#         logger.warning("No relevant context could be found after re-ranking.")
#         return

#     if args.show_chunks:
#         logger.info("="*60 + "\nDETAILED CONTEXT CHUNKS\n" + "="*60 + "\n" + context)

#     system_thought = read_prompt_template("thought_process")
#     prompt_thought = f"**User Question:**\n{args.query}\n\n---\n**Context:**\n{context}"
#     thought_output = llama3_call(prompt_thought, system_thought)
#     logger.info("="*20 + " Thought Process " + "="*20 + "\n" + thought_output)

#     system_persona = read_prompt_template(args.persona)
#     prompt_persona = f"Based ONLY on the reasoning below, provide the final answer:\n\n{thought_output}"
#     answer_output = llama3_call(prompt_persona, system_persona)
#     logger.info("="*20 + f" Answer ({args.persona}) " + "="*20 + "\n" + answer_output)

#     if args.review:
#         system_review = read_prompt_template("reviewer")
#         prompt_review = f"Context:\n{context}\n\nAnswer:\n{answer_output}"
#         review_output = llama3_call(prompt_review, system_review)
#         logger.info("="*20 + " Review " + "="*20 + "\n" + review_output)

# if __name__ == "__main__":
#     main()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Oracle RAG - Query Engine")
#     parser.add_argument("--query", type=str, required=True, help="User question.")
#     parser.add_argument("--top_k", type=int, default=3, help="Number of chunks for final context.")
#     parser.add_argument("--ensemble", action="store_true", help="Use multiple embedding models.")
#     parser.add_argument("--persona", choices=["consultant_answer", "developer_answer", "user_answer"], default="user_answer")
#     args = parser.parse_args()

#     load_resources() # Load all models and indexes on startup
    
#     top_chunk_ids = hierarchical_hybrid_search(args.query, ensemble=args.ensemble)
#     if not top_chunk_ids:
#         logger.warning("Could not find any relevant documents for the query.")
#         sys.exit()

#     context, sources = rerank_and_format_context(args.query, top_chunk_ids, args.top_k)
#     if not context:
#         logger.warning("No relevant context found after reranking.")
#         sys.exit()
    
#     # FR-18: Two-Step "Thinking" LLM Pipeline
#     logger.info("\n--- Step 3: Generating Thought Process ---")
#     prompt_thought = read_prompt_template("thought_process").format(question=args.query, context=context)
#     thought_process = llama3_call(prompt_thought, "You are a reasoning engine.")
#     logger.info(thought_process)

#     logger.info(f"\n--- Step 4: Generating Final Answer (Persona: {args.persona}) ---")
#     prompt_final = read_prompt_template(args.persona).format(thought_output=thought_process)
#     final_answer = llama3_call(prompt_final, "You are a helpful assistant.")
    
#     print("\n" + "="*50)
#     print(" Final Answer:")
#     print(final_answer)
#     print("\nSources:")
#     for source in sources:
#         print(f"- {source}")
#     print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oracle RAG - Query Engine")
    parser.add_argument("--query", type=str, required=True, help="User question.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks for final context.")
    parser.add_argument("--ensemble", action="store_true", help="Use multiple embedding models.")
    parser.add_argument("--persona", choices=["consultant_answer", "developer_answer", "user_answer"], default="user_answer")
    args = parser.parse_args()

    load_resources()
    
    # CORRECTED: Call the new, more reliable search function
    top_chunk_ids = fused_hybrid_search(args.query, ensemble=args.ensemble)
    
    if not top_chunk_ids:
        logger.warning("Could not find any relevant documents for the query.")
        sys.exit()

    context, sources = rerank_and_format_context(args.query, top_chunk_ids, args.top_k)
    if not context:
        logger.warning("No relevant context found after reranking.")
        sys.exit()
    
    # FR-18: Two-Step "Thinking" LLM Pipeline
    logger.info("\n--- Step 3: Generating Thought Process ---")
    prompt_thought = read_prompt_template("thought_process").format(question=args.query, context=context)
    thought_process = llama3_call(prompt_thought, "You are a reasoning engine.")
    logger.info(thought_process)

    logger.info(f"\n--- Step 4: Generating Final Answer (Persona: {args.persona}) ---")
    prompt_final = read_prompt_template(args.persona).format(thought_output=thought_process)
    final_answer = llama3_call(prompt_final, "You are a helpful assistant.")
    
    print("\n" + "="*50)
    print("Final Answer:")
    print(final_answer)
    print("\nSources:")
    for source in sources:
        print(f"- {source}")
    print("="*50)    # # Split ids by model for fetching
    # ids1, ids2 = [], []
    # if ensemble:
    #     for vid in vector_ids:
    #         if vid.startswith("m1_"):
    #             ids1.append(int(vid.split("_")[1]))
    #         elif vid.startswith("m2_"):
    #             ids2.append(int(vid.split("_")[1]))
    # else:
    #     ids1 = [int(v) for v in vector_ids]

    # Fetch rows for reranking, ensuring no duplicates from different models
    # rows_for_reranking = []
    # if ids1:
    #     ph1 = ','.join('?' for _ in ids1)
    #     c1.execute(f"SELECT chunk_id, indexed_text FROM metadata WHERE vector_id IN ({ph1})", ids1)
    #     rows_for_reranking.extend(c1.fetchall())
    # if ensemble and ids2:
    #     ph2 = ','.join('?' for _ in ids2)
    #     c2.execute(f"SELECT chunk_id, indexed_text FROM metadata WHERE vector_id IN ({ph2})", ids2)
    #     existing_chunk_ids = {r['chunk_id'] for r in rows_for_reranking}
    #     rows_for_reranking.extend([r for r in c2.fetchall() if r['chunk_id'] not in existing_chunk_ids])

    # if not rows_for_reranking:
    #     return "", []
    # rows_for_reranking = []
    # try:
    #     if ids1:
    #         ph1 = ','.join('?' for _ in ids1)
    #         c1.execute(f"SELECT chunk_id, indexed_text FROM metadata WHERE vector_id IN ({ph1})", ids1)
    #         rows_for_reranking.extend(c1.fetchall())
    #     if ensemble and ids2:
    #         ph2 = ','.join('?' for _ in ids2)
    #         c2.execute(f"SELECT chunk_id, indexed_text FROM metadata WHERE vector_id IN ({ph2})", ids2)
    #         existing_chunk_ids = {r['chunk_id'] for r in rows_for_reranking}
    #         rows_for_reranking.extend([r for r in c2.fetchall() if r['chunk_id'] not in existing_chunk_ids])
    # except sqlite3.Error as e:
    #     logger.error(f"Database error while fetching candidates for reranking: {e}", exc_info=True)
    #     return "", []

    # if not rows_for_reranking:
    #     return "", []

    # # # Rerank using CrossEncoder
    # # chunks = [{"id": row["chunk_id"], "text": row["indexed_text"]} for row in rows_for_reranking]
    # # pairs = [[query, ch["text"]] for ch in chunks]
    # # scores = model_reranker.predict(pairs, show_progress_bar=False)
    # # for ch, score in zip(chunks, scores):
    # #     ch["score"] = score
    
    # # Rerank using CrossEncoder
    # chunks = [{"id": row["chunk_id"], "text": row["indexed_text"]} for row in rows_for_reranking]
    # pairs = [[query, ch["text"]] for ch in chunks]
    # scores = model_reranker.predict(pairs, show_progress_bar=False) # Keep show_progress_bar=False
    # for ch, score in zip(chunks, scores):
    #     ch["score"] = score

    # ranked = sorted(chunks, key=lambda x: x["score"], reverse=True)

    # # === NEW LOGIC: DEDUPLICATION AND SELECTION ===
    # # This ensures we only select unique chunks based on the highest reranked score.
    # seen_chunk_texts = set()
    # unique_top_chunks = []
    # for ch in ranked:
    #     if ch["text"] not in seen_chunk_texts:
    #         seen_chunk_texts.add(ch["text"])
    #         unique_top_chunks.append(ch)
    #         if len(unique_top_chunks) == top_k:
    #             break

    # if not unique_top_chunks:
    #     return "", []

    # selected_ids = [ch["id"] for ch in unique_top_chunks]

    # # Retrieve full metadata for the final context
    # ph_sel = ','.join('?' for _ in selected_ids)
    
    # # final_rows = []
    # # # Query model1's DB
    # # c1.execute(f"SELECT * FROM metadata WHERE chunk_id IN ({ph_sel})", selected_ids)
    # # final_rows.extend(c1.fetchall())
    
    # # # Query model2's DB if in ensemble mode for any remaining chunks
    # # if ensemble:
    # #     retrieved_chunk_ids = {r['chunk_id'] for r in final_rows}
    # #     ids_to_fetch_from_c2 = [sid for sid in selected_ids if sid not in retrieved_chunk_ids]
    # #     if ids_to_fetch_from_c2:
    # #         ph_sel_c2 = ','.join('?' for _ in ids_to_fetch_from_c2)
    # #         c2.execute(f"SELECT * FROM metadata WHERE chunk_id IN ({ph_sel_c2})", ids_to_fetch_from_c2)
    # #         final_rows.extend(c2.fetchall())
    # # Retrieve full metadata for the final context
    # final_rows = []
    # try:
    #     ph_sel = ','.join('?' for _ in selected_ids)
    #     c1.execute(f"SELECT * FROM metadata WHERE chunk_id IN ({ph_sel})", selected_ids)
    #     final_rows.extend(c1.fetchall())
    #     if ensemble:
    #         retrieved_chunk_ids = {r['chunk_id'] for r in final_rows}
    #         ids_to_fetch_from_c2 = [sid for sid in selected_ids if sid not in retrieved_chunk_ids]
    #         if ids_to_fetch_from_c2:
    #             ph_sel_c2 = ','.join('?' for _ in ids_to_fetch_from_c2)
    #             c2.execute(f"SELECT * FROM metadata WHERE chunk_id IN ({ph_sel_c2})", ids_to_fetch_from_c2)
    #             final_rows.extend(c2.fetchall())
    # except sqlite3.Error as e:
    #     logger.error(f"Database error while fetching final context: {e}", exc_info=True)
    #     return "", []

    # # === NEW LOGIC: DOCUMENT ORDER SORTING ===
    # # Sort the final chunks by their ID to ensure they are in logical document order.
    # # This is critical for the LLM to understand procedural instructions.
    # final_rows.sort(key=lambda r: r['chunk_id'])
    
    # # === THE FULLY CORRECTED LOGIC: HIERARCHICAL GROUPING AND FORMATTING ===
    # # 1. Define the data structure to hold all fields
    # grouped_context = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
    #     'texts': [], 
    #     'headers': set(), 
    #     'footers': set(),
    #     'hyperlinks': set(), # ADDED
    #     'tables': set()      # ADDED
    # })))
    
    # # 2. Populate the structure with all data from the rows
    # for row in final_rows:
    #     group = grouped_context[row['document_name']][row['section_id']][row['page_num']]
        
    #     group['texts'].append(row['indexed_text'])
    #     if row['header_text']:
    #         group['headers'].add(row['header_text'])
    #     if row['footer_text']:
    #         group['footers'].add(row['footer_text'])
    #     if row['hyperlink_text']: # ADDED
    #         group['hyperlinks'].add(row['hyperlink_text'])
    #     if row['table_text']: # ADDED
    #         group['tables'].add(row['table_text'])

    # # 3. Build the final, structured context string and citations from the grouped data
    # final_context_parts = []
    # citations = []
    # for doc, sections in grouped_context.items():
    #     final_context_parts.append(f" Doc: {doc}\n==============================")
    #     for section, pages in sections.items():
    #         final_context_parts.append(f"\n Section: {section}\n------------------------------")
    #         for page, data in pages.items():
    #             final_context_parts.append(f"   Page: {page}")
                
    #             # Combine all text chunks from this page
    #             combined_text = "\n...\n".join(data['texts'])
    #             final_context_parts.append(f"     Text:\n    ---\n    {combined_text}\n    ---")
                
    #             # ADDED: Combine and display unique hyperlinks
    #             if data['hyperlinks']:
    #                 all_links = "\n    ".join(sorted(list(data['hyperlinks'])))
    #                 final_context_parts.append(f"\n     Links:\n    {all_links}")

    #             # ADDED: Combine and display unique tables
    #             if data['tables']:
    #                 all_tables = "\n    ".join(sorted(list(data['tables'])))
    #                 final_context_parts.append(f"\n     Table:\n    {all_tables}")
                
    #             # Combine and display unique headers and footers
    #             header = " / ".join(sorted(list(data['headers'])))
    #             footer = " / ".join(sorted(list(data['footers'])))
    #             if header or footer:
    #                 final_context_parts.append(f"\n    🧾 Header/Footer:\n    {header} / {footer}")
                
    #             # Create a single citation for this entire grouped block
    #             citations.append(f"Doc: {doc} | Section: {section} | Page: {page}")
    
    # final_context_string = "\n\n".join(final_context_parts)
    # # ======================================================================
    
    # return final_context_string, citations


# # --- CLI Entrypoint ---
# def main():
#     parser = argparse.ArgumentParser(description="Oracle RAG v6 - Thinking Pipeline")
#     parser.add_argument("--query", type=str, required=True, help="The user question to answer.")
#     parser.add_argument("--top_k", type=int, default=3, help="Number of top chunks to use for context.")
#     parser.add_argument("--show_chunks", action="store_true", help="Display retrieved chunks.")
#     parser.add_argument("--persona", choices=["consultant_answer", "developer_answer", "user_answer"], default="consultant_answer", help="Which persona answer to generate.")
#     parser.add_argument("--review", action="store_true", help="Perform optional review pass.")
#     parser.add_argument("--ensemble", action="store_true", help="Enable multi-model retrieval (across all-MiniLM and BGE models)")
#     args = parser.parse_args()

#     # Retrieval
#     candidate_ids = hybrid_search(args.query, top_n=20, ensemble=args.ensemble)
#     if not candidate_ids:
#         print("No candidate documents found from hybrid search.")
#         return
#     context, sources = rerank_and_format_context(args.query, candidate_ids, top_k=args.top_k, ensemble=args.ensemble)
#     if not context:
#         print("No relevant context could be found after re-ranking.")
#         return
#     if args.show_chunks:
#         print("\n" + "="*60 + "\nDETAILED CONTEXT CHUNKS\n" + "="*60)
#         print(context)
#         print("="*60)

#     # Step 1: Thought Process
#     system_thought = read_prompt_template("thought_process")
#     prompt_thought = f"**User Question:**\n{args.query}\n\n---\n**Context:**\n{context}"
#     thought_output = llama3_call(prompt_thought, system_thought)
#     print("\n" + "="*20 + " Thought Process " + "="*20)
#     print(thought_output)

#     # Step 2: Persona Answer
#     system_persona = read_prompt_template(args.persona)
#     prompt_persona = f"Based ONLY on the reasoning below, provide the final answer:\n\n{thought_output}"
#     answer_output = llama3_call(prompt_persona, system_persona)
#     print("\n" + "="*20 + f" Answer ({args.persona}) " + "="*20)
#     print(answer_output)

#     # Step 3: Optional Review
#     if args.review:
#         system_review = read_prompt_template("reviewer")
#         prompt_review = f"Context:\n{context}\n\nAnswer:\n{answer_output}"
#         review_output = llama3_call(prompt_review, system_review)
#         print("\n" + "="*20 + " Review ")
#         print(review_output)

# if __name__ == "__main__":
#     main()