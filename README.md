
# Academic Research Assistant with RAG

Local RAG assistant over your PDF library, running entirely against a local Ollama instance. Hybrid BM25+vector retrieval, cross-encoder re-ranking, section/pinpoint/year/jurisdiction extraction, whole‑document summarization, cross‑source comparison with conflict detection, toggleable conversation memory, and an automatic research trail.

## Setup

### Prerequisites
- **Python 3.9+** (tested with 3.13)
- **Ollama** installed and running locally (download from [ollama.com](https://ollama.com))
- **At least 8GB VRAM** (the more the better) – adjust models in `config.py` if you have limited resources.

### 1. Clone or download the repository
```bash
git clone https://github.com/Vtolin/Local-Academic-Research-Assistant/blob/main/README.md
cd local-academic-research-assistant
```

### 2. Choose your Python environment

#### Option A: Virtual environment 
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# or
venv\Scripts\activate           # Windows
```

#### Option B: System Python 
Skip the venv commands and install packages globally.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull the required Ollama models
The system uses two models (configurable in `config.py`):
- **Embedding model:** `nomic-embed-text`
- **Extraction model (fast but not as accurate as larger models, 4B):** `qwen3.5:4b` - you can also use `qwen2.5:7b` (reccomended) if you have atleast 8gb vram (can run on 6gb but may be slower)
- **Synthesis model (quite heavy if you have less than 6gb vram, 7B):** `qwen2.5:7b` – you can also use `qwen3:14b` (12gb vram recommended, if you have 8gb then you can use 7B or sacrifice speed with partial offloading) if your hardware allows (adjust `SYNTHESIS_MODEL` in `config.py`).

Pull them:
```bash
ollama pull nomic-embed-text
ollama pull qwen3.5:4b
ollama pull qwen2.5:7b
```

> If you are on a GPU‑poor machine, you can reduce model sizes (e.g., `qwen2.5:7b` → `qwen3.5:4b`) but quality will drop noticeably.

### 5. Prepare your PDF library
Create a folder named `pfolder` in the project root (or change `DOC_FOLDER` in `config.py`). Place your PDF files inside (subdirectories are supported).

```bash
mkdir pfolder
# copy your PDFs into pfolder/
```

### 6. Run the assistant
```bash
python main.py
```

On first run, it will:
- Detect PDFs in `pfolder`
- Chunk and embed them using `nomic-embed-text`
- Build a persistent Chroma vector store in `chroma_db/`
- Load a BM25 index (rebuilds on‑the‑fly)

Once ready, you’ll see a prompt. Type `reindex` if you add/remove PDFs later (the system also automatically syncs new files on startup).

---

## Usage

| Input | What happens |
|---|---|
| `What is the main methodology?` | Focused hybrid search |
| `What does page 5 say?` / `compare page 10 and page 23` | Exact page(s), no search |
| `summarize journal5` | Auto‑detected file reference, answered as a normal question scoped to that file |
| `filter: filename.pdf \| your question` | Explicit file scope – also accepts a year (`filter: 2021 \| ...`) or jurisdiction (`filter: australia \| ...`) |
| `broad: your question` | Wide, diversity‑optimized sweep |
| `summarize: journal5` / `summarize: 2021` | Whole‑document summary (stuffing or map‑reduce) |
| `compare: doc1, doc2 \| your question` | Separate per‑source retrieval + conflict‑aware synthesis |
| `memory on` / `memory off` / `forget` | Toggle or clear conversation history |
| `export citations` | Write every source cited this session to `citations_export.txt` |
| `reindex` | Rebuild the index from scratch |

---

## Configuration

All tunable parameters live in `config.py`. Key settings:

- **Storage:** `PERSIST_DIR`, `DOC_FOLDER`
- **Chunking:** `CHUNK_SIZE`, `CHUNK_OVERLAP`
- **Retrieval counts:** `SIMILARITY_K`, `BM25_K`, `HYBRID_TOP_N`, `BROAD_K`, etc.
- **Cross‑encoder:** `RERANK_MODEL` (default `ms-marco-MiniLM-L-12-v2`) – first run will download from Hugging Face (~500 MB).
- **LLM models:** `LLM_MODEL` (extraction) and `SYNTHESIS_MODEL` (final synthesis). Change these to match what you pulled.
- **Context windows:** `NUM_CTX`, `MAP_NUM_CTX`, `REDUCE_NUM_CTX`, etc. – adjust for your hardware.

---

## Important Notes

- **flashrank model download:** The cross‑encoder model is downloaded from Hugging Face on first rerank. Ensure internet access and sufficient disk space (~500 MB). If it fails, retrieval continues without reranking (a warning is printed).
- **Ollama must be running** before you start the assistant. The system expects the default `http://localhost:11434`.
- **Reindex after major updates:** If you update the code (especially ingestion or metadata extraction), run `reindex` from the prompt to rebuild the vector store with the new logic.
- **Citation format:** Currently style‑agnostic (`filename, Page N, ¶ 42 (Section; Jurisdiction; Year)`). You can change the format in `research_trail.py` (`format_citation`).
- **Memory usage:** The 14B synthesis model may be heavy. Reduce `SYNTHESIS_MODEL` to `qwen2.5:7b` (or smaller) if you encounter out‑of‑memory errors or slow generations.

---

## Troubleshooting

| Issue | Likely fix |
|-------|------------|
| `ModuleNotFoundError` | Check that you activated the virtual environment and installed requirements. |
| `Ollama connection refused` | Ensure Ollama is running (`ollama serve` or check system tray). |
| Embedding errors during indexing | Some PDFs may contain malformed characters. The system skips bad chunks; check the warning messages. |
| `Error finding id` during retrieval | This is a Chroma/HNSW issue when requesting more results than available. The system caps `k` automatically when a filter is active. If you still see it, reduce `SIMILARITY_K` or `BROAD_K`. |
| `flashrank` download fails | Check your internet connection and proxy settings. You can also download the model manually and point `RERANK_CACHE_DIR` to the local folder. |
| Out‑of‑memory (OOM) | Lower the context windows in `config.py` (e.g., `NUM_CTX=8192`), use smaller models, or enable CPU offloading in Ollama. |

---

## Files

| File | Responsibility |
|---|---|
| `config.py` | All tunable constants, grouped by pipeline stage |
| `query_understanding.py` | Intent classification (page/broad/summarize/compare/factual) |
| `filters.py` | Filename, year, and jurisdiction filter resolution |
| `heading_detection.py` | Section heading detection (font size/boldness) |
| `metadata_extraction.py` | Publication‑year extraction |
| `jurisdiction_extraction.py` | Jurisdiction/court extraction (keyword patterns) |
| `pinpoint_detection.py` | Paragraph‑marker (pinpoint citation) detection |
| `ingestion.py` | PDF loading, chunking, metadata tagging, vector store sync |
| `retrieval.py` | Hybrid retrieval, direct/whole‑doc fetch, cross‑encoder rerank |
| `generation.py` | Context assembly, per‑intent prompts (incl. compare), Ollama calls |
| `summarization.py` | Whole‑document summarization (stuff / map‑reduce) |
| `conversation.py` | Bounded, toggleable Q&A memory |
| `research_trail.py` | Auto‑logged research trail + citation export |
| `main.py` | CLI orchestration |

---

## Next Steps / Customisation

- **Citation style:** Edit `research_trail.py` → `format_citation` to implement Bluebook, OSCOLA, AGLC, or your faculty’s style.
- **Add more jurisdictions:** Extend the pattern table in `jurisdiction_extraction.py`.
- **Prompt tuning:** Adjust system prompts in `generation.py` and `summarization.py` to better suit your domain.
- **Frontend:** The CLI works, but you can wrap the logic in a Streamlit or Gradio UI.

---

## License
