# scripts/config.py
from pathlib import Path

# --- Core Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = OUTPUT_DIR / "logs"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DB_PATH = OUTPUT_DIR / "project_data.db"
# --- Local Model Cache Paths (for offline resilience) ---
LOCAL_MODELS_CACHE_DIR = PROJECT_ROOT / "local_models_cache"
# LOCAL_MODELS_CACHE_DIR = PROJECT_ROOT / "local_models_cache"


# --- LLM & Model Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3"  # For summarization and generation

# --- Embedding & Retrieval Models ---
# List of all models that need to be available for creating embeddings
EMBEDDING_MODELS_LIST = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]

# The primary model used for generating user query vectors
# IMPORTANT: This MUST be one of the models from the list above.
PRIMARY_RETRIEVER_MODEL = "all-MiniLM-L6-v2"

# The model used for the final reranking step
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- DO NOT EDIT BELOW THIS LINE ---
# Programmatically create paths for all required models
MODEL_PATHS = {
    # Add all embedding models to the paths dictionary
    **{model_name: str(LOCAL_MODELS_CACHE_DIR / model_name.replace("/", "_")) for model_name in EMBEDDING_MODELS_LIST},
    # Add the reranker model to the paths dictionary
    RERANKER_MODEL: str(LOCAL_MODELS_CACHE_DIR / RERANKER_MODEL.replace("/", "_")),
}

# EMBEDDING_MODELS = ['all-MiniLM-L6-v2', 'BAAI/bge-small-en-v1.5']

# # # The retriever model is used to generate the query vector. It MUST be one of the EMBEDDING_MODELS.
# CACHE_DIR = PROJECT_ROOT / "local_models_cache"
# MODEL_PATH_RETRIEVER = str(CACHE_DIR / "sentence-transformers_all-MiniLM-L6-v2")
# MODEL_PATH_RERANKER = str(CACHE_DIR / "cross-encoder_ms-marco-MiniLM-L-6-v2")

# MODEL_NAME_RETRIEVER = 'all-MiniLM-L6-v2'
# MODEL_NAME_RERANKER = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# --- Chunking Parameters (Ref: FSD Hardcoded Parameters Table) ---
CHUNK_SIZE_TOKENS = 256
CHUNK_MIN_TOKENS = 32
CHUNK_OVERLAP_SENTENCES = 1

# --- Search & Ranking Parameters ---
RRF_K = 60  # Reciprocal Rank Fusion k value
HIERARCHICAL_SEARCH_TOP_N_SECTIONS = 5  # Number of sections to find in the first pass
HIERARCHICAL_SEARCH_TOP_N_CHUNKS = 10  # Number of chunks to retrieve in the second pass for reranking

# --- Operational Settings ---
# If True, the query handler will exit if any model's artifacts are missing.
# If False, it will log a warning and proceed with the available models.
STRICT_ENSEMBLE_MODE = True
top_k = 5


# (after the MODEL_PATHS dictionary)

# --- Extraction Parameters (Ref: FSD Hardcoded Parameters Table) ---
HEADER_Y_RATIO = 0.08
FOOTER_Y_RATIO = 0.915

# --- Chunking Parameters (Ref: FSD Hardcoded Parameters Table) ---
# ... (rest of the file)
