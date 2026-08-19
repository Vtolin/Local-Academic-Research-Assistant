"""
Ingestion: loading PDFs from disk, chunking, and building/syncing the
persistent Chroma vector store.

Uses heading_detection.py (PyMuPDF-based) instead of PyPDFLoader for
extraction - PyPDFLoader gives back one flat text blob per page with no
structural information, which is enough for page-level metadata but
useless for section labels. heading_detection splits each page into
(section, text) segments up front, so every chunk downstream carries a
real 'section' tag alongside its page number.
"""
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import DOC_FOLDER, PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from heading_detection import build_page_texts_with_sections
from metadata_extraction import extract_year
from jurisdiction_extraction import extract_jurisdiction
from pinpoint_detection import tag_chunks_with_pinpoints


def _sanitize_text(text):
    """
    Strip characters that can make a chunk's text fail to round-trip as
    valid JSON/UTF-8 when sent to the embedding endpoint.

    PDFs with malformed or custom-mapped embedded fonts (common in
    scanned or oddly-produced legal documents) occasionally yield NUL
    bytes or lone/unpaired UTF-16 surrogate codepoints when their text is
    extracted. Both are valid Python str characters - they survive
    silently through chunking and storage - but neither is valid UTF-8,
    so a single one of these anywhere in a large batched embedding
    request is enough to make the *entire* request fail. The failure then
    surfaces far from its actual cause (a generic connection/HTTP error
    from the embedding client), which is exactly what made this bug hard
    to track down. Sanitizing here, at first extraction, is cheap and
    removes the failure mode at its source rather than downstream.
    """
    if not text:
        return text
    text = text.replace("\x00", "")
    return "".join(
        ch if not (0xD800 <= ord(ch) <= 0xDFFF) else "\ufffd"
        for ch in text
    )


def _relative_source(full_path):
    """Path relative to DOC_FOLDER, normalized to forward slashes so
    matching and display are consistent across Windows/Linux/Mac."""
    try:
        rel = os.path.relpath(full_path, DOC_FOLDER)
    except ValueError:
        rel = os.path.basename(full_path)
    return rel.replace(os.sep, "/")


def get_indexed_documents(vectorstore):
    """
    Return {relative_path: full_source_path} for every unique file in the
    index. Keyed by path relative to pfolder (not bare basename) so that
    two same-named files in different subfolders don't silently collide.
    """
    try:
        results = vectorstore.get(include=["metadatas"])
    except Exception as e:
        print(f"[warning] could not read index metadata: {e}")
        return {}

    doc_map = {}
    for metadata in results.get("metadatas", []):
        if metadata and "source" in metadata:
            full = metadata["source"]
            doc_map[_relative_source(full)] = full
    return doc_map


def get_indexed_years(vectorstore):
    """
    Return {relative_path: year_or_None} for every unique file in the
    index, reading the 'year' metadata tag attached at ingestion time (see
    metadata_extraction.extract_year). A document with no detected year
    maps to None rather than being omitted, so callers can distinguish
    "not indexed" from "indexed but year unknown".
    """
    try:
        results = vectorstore.get(include=["metadatas"])
    except Exception as e:
        print(f"[warning] could not read index metadata: {e}")
        return {}

    doc_years = {}
    for metadata in results.get("metadatas", []):
        if metadata and "source" in metadata:
            rel = _relative_source(metadata["source"])
            if rel not in doc_years or doc_years[rel] is None:
                doc_years[rel] = metadata.get("year")
    return doc_years


def get_indexed_jurisdictions(vectorstore):
    """
    Return {relative_path: jurisdiction_or_None} for every unique file in
    the index, mirroring get_indexed_years - see
    jurisdiction_extraction.extract_jurisdiction for how the value is
    determined and its limits (best-effort keyword matching, not real
    entity recognition).
    """
    try:
        results = vectorstore.get(include=["metadatas"])
    except Exception as e:
        print(f"[warning] could not read index metadata: {e}")
        return {}

    doc_jurisdictions = {}
    for metadata in results.get("metadatas", []):
        if metadata and "source" in metadata:
            rel = _relative_source(metadata["source"])
            if rel not in doc_jurisdictions or doc_jurisdictions[rel] is None:
                doc_jurisdictions[rel] = metadata.get("jurisdiction")
    return doc_jurisdictions


def _load_pdf_with_sections(path):
    """
    Load one PDF into a list of Documents - one per (page, detected
    section) segment - using layout-aware heading detection. If no
    headings are found anywhere in the document, this degrades to exactly
    one Document per page with an empty section label, which is no worse
    than the original flat page-blob extraction.

    Also runs best-effort publication-year and jurisdiction extraction
    once per PDF (not once per segment) and stamps every segment from this
    file with the same values, so year/jurisdiction filtering work at the
    document level regardless of which chunk ends up matching a query.
    """
    try:
        pages = build_page_texts_with_sections(path)
    except Exception as e:
        print(f"[warning] could not load {os.path.basename(path)}: {e}")
        return []

    year = extract_year(path)
    jurisdiction = extract_jurisdiction(path)

    docs = []
    for page_num in sorted(pages):
        for section, text in pages[page_num]:
            text = _sanitize_text(text)
            if not text.strip():
                continue
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": path, "page": page_num, "section": _sanitize_text(section),
                    "year": year, "jurisdiction": jurisdiction,
                },
            ))
    return docs


def split_and_load(paths):
    docs = []
    for p in paths:
        docs.extend(_load_pdf_with_sections(p))
    return docs


def _discover_pdfs():
    """Every .pdf under DOC_FOLDER, including subfolders."""
    paths = []
    for root, _, files in os.walk(DOC_FOLDER):
        for f in files:
            if f.lower().endswith(".pdf"):
                paths.append(os.path.join(root, f))
    return sorted(paths)


# How many chunks go into a single embedding API call. Bounds the blast
# radius of a bad chunk (see _sanitize_text) - if this were "every chunk
# in one call" (the original behavior), one rejected chunk out of
# thousands would fail the entire index build with no indication of
# which one was responsible.
_EMBED_BATCH_SIZE = 200


def _add_batch(vectorstore, docs, embeddings, persist_dir):
    """Embed and store one batch, creating the Chroma collection on the
    first successful batch and appending to it afterward."""
    if vectorstore is None:
        return Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)
    vectorstore.add_documents(docs)
    return vectorstore


def _embed_with_isolation(splits, embeddings, persist_dir):
    """
    Build/extend a Chroma index in fixed-size batches rather than one
    call covering every chunk, and isolate failures down to the
    individual chunk responsible.

    If a batch fails (most likely one rejected chunk - see
    _sanitize_text's docstring for the common cause even after
    sanitization, e.g. a chunk so degenerate the embedding model itself
    rejects it), that batch is retried one document at a time so the
    specific bad chunk(s) can be identified, reported by source/page, and
    skipped - every other chunk, in that batch and every other batch,
    still gets indexed. Only raises if EVERY chunk fails (nothing at all
    could be indexed), since that's the "embedding backend is actually
    unreachable" case main.py needs to detect and report distinctly.
    """
    vectorstore = None
    skipped = []
    for i in range(0, len(splits), _EMBED_BATCH_SIZE):
        batch = splits[i:i + _EMBED_BATCH_SIZE]
        try:
            vectorstore = _add_batch(vectorstore, batch, embeddings, persist_dir)
        except Exception:
            for doc in batch:
                try:
                    vectorstore = _add_batch(vectorstore, [doc], embeddings, persist_dir)
                except Exception as e:
                    src = os.path.basename(doc.metadata.get("source", "?"))
                    page = doc.metadata.get("page", 0) + 1
                    skipped.append((src, page, str(e)))

    if skipped:
        print(f"\n[warning] {len(skipped)} chunk(s) were rejected by the embedding backend and skipped:")
        for src, page, err in skipped[:10]:
            print(f"  - {src}, page {page}: {err}")
        if len(skipped) > 10:
            print(f"  ...and {len(skipped) - 10} more (see research_trail/logs for the full run).")

    if vectorstore is None:
        raise RuntimeError(
            f"Every one of {len(splits)} chunk(s) was rejected by the embedding backend - "
            "this points to the backend itself being unreachable/broken, not a bad chunk."
        )
    return vectorstore


def index_folder(embeddings, persist_dir=None):
    """
    persist_dir defaults to config.PERSIST_DIR, but callers (see main.py's
    'reindex' handling) can point this at a temporary directory instead,
    so a rebuild can be attempted WITHOUT first destroying the existing,
    working index. Only swap the temp directory into place once this
    returns a real vectorstore - if it raises or returns None, the caller
    still has the old index untouched.
    """
    persist_dir = persist_dir or PERSIST_DIR
    print(f"\n[1/3] Loading PDFs from '{DOC_FOLDER}'...")
    pdf_paths = _discover_pdfs()
    if not pdf_paths:
        return None

    docs = []
    for i, p in enumerate(pdf_paths, 1):
        print(f"  [{i}/{len(pdf_paths)}] {os.path.basename(p)}")
        docs.extend(_load_pdf_with_sections(p))
    if not docs:
        return None
    print(f"Loaded {len(docs)} page/section segment(s) from {len(pdf_paths)} file(s).")

    print("\n[2/3] Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = splitter.split_documents(docs)
    for i, chunk in enumerate(splits):
        chunk.metadata["chunk_index"] = i
    tag_chunks_with_pinpoints(splits)
    print(f"Created {len(splits)} text chunks.")

    print("\n[3/3] Building and saving persistent Vector Database...")
    try:
        vectorstore = _embed_with_isolation(splits, embeddings, persist_dir)
        print("Vector database built and saved to disk successfully.")
        return vectorstore
    except Exception as e:
        # Deliberately re-raised rather than returned as None: None is
        # also what this function returns when there are simply no PDFs
        # to index (see the early return above), and those are two very
        # different situations for a caller to report to the user - "add
        # some PDFs" vs. "your embedding backend errored out on
        # every chunk". Swallowing this into the same None made every
        # embedding failure print a misleading "No PDF files found" in
        # main.py even when PDFs were found and chunked successfully.
        print(f"Error generating embeddings: {e}")
        raise


def sync_new_files(vectorstore, doc_map):
    """
    Embed any PDFs dropped into pfolder (including subfolders) since the
    last run. Returns True if anything new was indexed - callers should
    refresh the BM25 side of hybrid retrieval (HybridIndex.refresh_bm25)
    whenever this happens, since BM25Retriever has no add_documents and
    would otherwise silently miss the new content.
    """
    on_disk = {_relative_source(p) for p in _discover_pdfs()}

    new_files = sorted(on_disk - set(doc_map.keys()))
    if not new_files:
        return False

    print(f"\nFound {len(new_files)} new PDF(s) not yet indexed: {', '.join(new_files)}")
    full_paths = [os.path.join(DOC_FOLDER, f) for f in new_files]
    docs = split_and_load(full_paths)
    if not docs:
        return False

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = splitter.split_documents(docs)
    for i, chunk in enumerate(splits):
        chunk.metadata["chunk_index"] = i
    tag_chunks_with_pinpoints(splits)

    skipped = []
    indexed = 0
    for i in range(0, len(splits), _EMBED_BATCH_SIZE):
        batch = splits[i:i + _EMBED_BATCH_SIZE]
        try:
            vectorstore.add_documents(batch)
            indexed += len(batch)
        except Exception:
            for doc in batch:
                try:
                    vectorstore.add_documents([doc])
                    indexed += 1
                except Exception as e:
                    src = os.path.basename(doc.metadata.get("source", "?"))
                    page = doc.metadata.get("page", 0) + 1
                    skipped.append((src, page, str(e)))

    if skipped:
        print(f"[warning] {len(skipped)} chunk(s) were rejected by the embedding backend and skipped:")
        for src, page, err in skipped[:10]:
            print(f"  - {src}, page {page}: {err}")
        if len(skipped) > 10:
            print(f"  ...and {len(skipped) - 10} more.")

    print(f"Indexed {indexed} new chunk(s).")
    return True
