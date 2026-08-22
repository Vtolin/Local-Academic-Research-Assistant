
# Academic Research Assistant with RAG

Local RAG assistant over your PDF/DOCX library, running entirely against a local Ollama instance. Hybrid BM25+vector retrieval, cross-encoder re-ranking, section/pinpoint/year/jurisdiction extraction, whole‑document summarization, cross‑source comparison with conflict detection, toggleable conversation memory, and an automatic research trail.

> **Disclaimer**
>
> This is a **research assistant, not legal advice**. It is not a substitute for a qualified legal professional, and no output of this tool should be treated as legal counsel or relied upon for any legal decision.
>
> AI-generated answers and summaries can be wrong, incomplete, or out of date. **Always verify every output against the original source documents** before using it for academic, legal, or professional work. The verbatim-extracted sections (statistics, tables, citations) are copied directly from your documents, but everything else is model-generated.

## Setup

### Requirements 
- **Python 3.9+** (tested with 3.13)
- **Ollama** installed and running locally (download from [ollama.com](https://ollama.com))
- **At least 6GB VRAM(8gb Preferred) with 16gb System RAM** (the more the better) – adjust models in `config.py` if you have limited resources.

### 1. Clone or download the repository
```bash
git clone https://github.com/Vtolin/Local-Academic-Research-Assistant
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
- **Note**: Consider using the Gemma 4 26B MoE model for synthesis if you have 32gb unified memory/32gb DDR5 Ram with 6-8gb VRAM. 

Pull them:
```bash
ollama pull nomic-embed-text
ollama pull qwen3.5:4b
ollama pull qwen2.5:7b
```

> Note: the config.py defaults at a 7b + 7b configuration.
---
> If you are on a GPU‑poor machine, you can reduce model sizes (e.g., `qwen2.5:7b` → `qwen3.5:4b`) but quality will drop noticeably if you use `qwen3.5:4b` for synthesis.

### 5. Prepare your document library
Create a folder named `pfolder` in the project root (or change `DOC_FOLDER` in `config.py`). Place your PDF or DOCX files inside (subdirectories are supported; DOCX files are converted via PyMuPDF, so heading/year/jurisdiction extraction and page metadata work for both formats).

```bash
mkdir pfolder
# copy your PDFs/DOCX files into pfolder/
```

### 6. Run the assistant
```bash
python main.py
```

On first run, it will:
- Detect supported documents (`.pdf`, `.docx`) in `pfolder`
- Chunk and embed them using `nomic-embed-text`
- Build a persistent Chroma vector store in `chroma_db/`
- Load a BM25 index (rebuilds on‑the‑fly)

Once ready, you’ll see a prompt. Type `reindex` if you add/remove documents later (the system also automatically syncs new files on startup).

---

## Usage

| Input | What happens |
|---|---|
| `What is the main methodology?` | Focused hybrid search |
| `What does page 5 say?` / `what do pages 5 and 7 say?` | Exact page(s), no search |
| `summarize journal5` | Auto‑detected file reference, answered as a normal question scoped to that file |
| `filter: filename.pdf \| your question` | Explicit file scope – also accepts a year (`filter: 2021 \| ...`) or jurisdiction (`filter: indonesia \| ...`) |
| `broad: your question` | Wide, diversity‑optimized sweep |
| `summarize: journal5` / `summarize: 2021` | Whole‑document summary (stuffing or map‑reduce) |
| `summarize: journal5 pdf` | Whole‑document summary, also exported as a PDF to `pdf_exports/` |
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
- **Table data:** `summarize:` now appends an "Extracted Table Data (Verbatim)" section - real tables are captured deterministically from the PDF layout (no model involvement, so cell values can't hallucinate) and rendered as markdown. One-row fragments, garbage tables, and oversized tables are filtered/capped with explicit notes.
- **Degraded layout safety net:** pages whose extraction alternates between two columns are indexed (keyword search still works) but tagged `layout_warning`; whole‑document summaries exclude those chunks, and Q&A context flags them, rather than feeding interleaved half‑sentences to the models. If a whole document is flagged, `summarize:` reports it instead of producing garbage - but verbatim tables from those pages are still extracted (find_tables clusters cells by position, recovering structure the raw line order lost).
- **Ollama must be running** before you start the assistant. The system expects the default `http://localhost:11434`.
- **Verify outputs:** this is a research assistant, not legal advice — AI answers can be wrong. Always check generated answers and summaries against the original source documents (see the disclaimer at the top).
- **Reindex after major updates:** If you update the code (especially ingestion or metadata extraction), run `reindex` from the prompt to rebuild the vector store with the new logic.
- **Citation format:** Currently style‑agnostic (`filename, Page N, ¶ 42 (Section; Jurisdiction; Year)`). You can change the format in `research_trail.py` (`format_citation`).
- **Memory usage:** The 14B synthesis model may be heavy. Reduce `SYNTHESIS_MODEL` to `qwen2.5:7b` (or smaller) if you encounter out‑of‑memory errors or slow generations.

---

## Troubleshooting

| Issue | Likely fix |
|-------|------------|
| `ModuleNotFoundError` | Check that you activated the virtual environment and installed requirements. |
| `Ollama connection refused` | Ensure Ollama is running (`ollama serve` or check system tray). |
| Embedding errors during indexing | Some documents may contain malformed characters. The system skips bad chunks; check the warning messages. |
| `[warning] ... dropped N segment(s) with suspected corrupted/garbage text` | A safety net caught corrupted-font garbage during ingestion. Those segments are excluded from the index - nothing to do unless you expected that text. |
| `[warning] ... page(s) N suspected interleaved multi-column layout` | Those pages' text alternates between columns. They stay searchable, but summaries exclude them (safety net) - see the "Degraded layout" note below. |
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
| `heading_detection.py` | Section heading detection (font size/boldness) + interleaved multi-column page detection |
| `content_quality.py` | Garbage-text scoring (corrupted-PDF safety net) |
| `table_extraction.py` | Deterministic verbatim table capture via PyMuPDF `find_tables` (no Java dependency) |
| `pdf_export.py` | Markdown→PDF export of summaries (markdown-it-py + PyMuPDF Story, no new heavy deps) |
| `metadata_extraction.py` | Publication‑year extraction |
| `jurisdiction_extraction.py` | Jurisdiction/court extraction (keyword patterns, incl. Indonesian courts) |
| `pinpoint_detection.py` | Pinpoint citation detection (¶ markers, Pasal/ayat/huruf, jo./juncto chains) |
| `ingestion.py` | Document (PDF/DOCX) loading, chunking, metadata tagging, vector store sync |
| `retrieval.py` | Hybrid retrieval, direct/whole‑doc fetch, cross‑encoder rerank |
| `generation.py` | Context assembly, per‑intent prompts (incl. compare), Ollama calls |
| `summarization.py` | Whole‑document summarization (stuff / map‑reduce) |
| `conversation.py` | Bounded, toggleable Q&A memory |
| `research_trail.py` | Auto‑logged research trail + citation export |
| `main.py` | CLI orchestration |

---


