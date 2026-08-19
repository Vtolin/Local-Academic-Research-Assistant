"""
Central configuration for the Thesis RAG Assistant.
"""

# --- Storage / corpus ---
PERSIST_DIR = "./chroma_db"
DOC_FOLDER = "./pfolder"

# --- Chunking (ingestion) ---
CHUNK_SIZE = 4700
CHUNK_OVERLAP = 880

# --- Retrieval: narrow ---
SIMILARITY_K = 10
BM25_K = 10
HYBRID_TOP_N = 8

# --- Retrieval: broad ---
BROAD_K = 15
BROAD_FETCH_K = 30
BROAD_BM25_K = 15
BROAD_TOP_N = 12

BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6

# --- Cross-encoder re-ranking ---
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
RERANK_CACHE_DIR = "./rerank_cache"

# --- Generation ---
# Map/extract: 4B is fast and adequate for bulk fact extraction (batches
# are short, mechanical bullets). Change to match your actual Ollama tag.
LLM_MODEL = "qwen2.5:7b" # You may want to use Qwen3.5:4b here if you have 4gb VRAM unless you're okay sacrificing speed for precision

# Final synthesis: 14B is strongly recommended for survey/journal depth.
# If 14B is too slow, use qwen2.5:7b temporarily, but quality will drop.
SYNTHESIS_MODEL = "qwen2.5:7b"

NUM_CTX = 21560
CTX_SAFETY_MARGIN = 800
GENERATION_TEMPERATURE = 0.0

# Stage-specific caps
DOC_TYPE_NUM_CTX = 4096
DOC_TYPE_NUM_PREDICT = 32

MAP_NUM_CTX = 6144
MAP_NUM_PREDICT = 3072
MAP_BUDGET_RATIO = 0.55

REDUCE_NUM_CTX = 12288
REDUCE_NUM_PREDICT = 2048
REDUCE_BUDGET_RATIO = 0.60

SYNTHESIS_NUM_PREDICT = 5120

# Force intermediate reduce when more than this many extracts exist,
# even if they fit in the final context window.
FORCE_REDUCE_EXTRACT_THRESHOLD = 8

EMBED_MODEL = "nomic-embed-text"

# --- Conversation memory ---
MAX_HISTORY_TURNS = 3

# --- Compare mode ---
COMPARE_TOP_N_PER_SOURCE = 5

# --- Whole-document summarization ---
SUMMARY_BUDGET_RATIO = 0.6
