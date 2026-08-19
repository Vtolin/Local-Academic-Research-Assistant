"""
Academic Thesis / Legal Research RAG Assistant - CLI entry point.

Pipeline (matches the flowchart, plus 'summarize:' and 'compare:' added as
parallel paths that bypass the standard fused-retrieval branch - see
below):
  User Query -> Query Understanding -> Intent?
    -> page-specific       -> Filter Resolution -> Direct Chunk Fetch
    -> factual/compare/... -> Filter Resolution -> Hybrid Retrieval (BM25+Vector, narrow)
    -> broad                -> Filter Resolution (wide) -> Hybrid Retrieval (BM25+MMR, broad)
    -> summarize             -> Filter Resolution -> whole-document fetch -> stuff/map-reduce
    -> compare                -> Filter Resolution PER TARGET -> separate per-source retrieval
  -> Cross-Encoder Re-ranking -> Context Assembly -> Prompt Selection (by intent)
  -> LLM Generation (config.LLM_MODEL / SYNTHESIS_MODEL) -> Citation Formatting
  -> Research Trail -> Final Response
"""
import gc
import logging
import os
import shutil
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# The ollama Python client logs every HTTP request/response via httpx at
# INFO level (e.g. "POST /api/chat 200 OK"). Harmless - just noisy in a
# CLI - so it's quieted to WARNING here rather than left to clutter every
# query. Leave this out (or set it back to INFO) if you want to see raw
# request/response logging for debugging network issues with Ollama.
logging.getLogger("httpx").setLevel(logging.WARNING)

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from config import (
    PERSIST_DIR, DOC_FOLDER, EMBED_MODEL, HYBRID_TOP_N, BROAD_TOP_N,
    MAX_HISTORY_TURNS, COMPARE_TOP_N_PER_SOURCE,
)
from ingestion import (
    index_folder, sync_new_files, get_indexed_documents,
    get_indexed_years, get_indexed_jurisdictions,
)
from query_understanding import parse_query, Intent
from filters import resolve_filter, build_source_filter
from retrieval import HybridIndex, get_chunks_by_pages, sort_docs, Chunk
from generation import format_docs, format_compare_docs, warn_if_context_too_large, generate_answer
from summarization import summarize_document
from conversation import ConversationMemory
from research_trail import ResearchTrail, format_citation

RESEARCH_TRAIL_PATH = "./research_trail.md"
CITATION_EXPORT_PATH = "./citations_export.txt"


def _safe_rmtree(path, retries=5, delay=0.5):
    """
    shutil.rmtree with retries, returning whether `path` is actually gone.

    On Windows, a Chroma/SQLite file that was just closed (e.g. because
    the Python object holding it went out of scope) can stay locked by
    the OS for a brief moment afterward. The original code called
    shutil.rmtree(path, ignore_errors=True) exactly once and moved on
    regardless of whether the delete actually succeeded - so a locked
    file could be silently left behind, and the *next* index build would
    write fresh data alongside those stale leftovers in the same
    directory. A vector index built from that kind of mixed old/new state
    is a very plausible cause of the unpredictable "Error finding id"
    HNSW errors seen elsewhere in this project - it's not just cosmetic.
    """
    for _ in range(retries):
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            return True
        time.sleep(delay)
    return not os.path.exists(path)


def _close_chroma_client(vectorstore):
    """
    Deterministically release a Chroma client's file handles so its
    persist directory can be deleted/moved immediately afterward.

    chromadb's Client.close() stops the shared System singleton - closing
    chroma.sqlite3 and the segment files - once the last client referencing
    it is closed. This is what makes a follow-up rmtree succeed. The
    previously used SharedSystemClient.clear_system_cache() only dropped
    the cache dictionaries WITHOUT stopping the System, so every file
    handle stayed open and the delete failed on Windows with "a file may
    still be locked" (the exact error reindex used to report).
    """
    try:
        client = getattr(vectorstore, "_client", None)
        if client is not None:
            client.close()
    except Exception:
        try:
            import chromadb
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except Exception:
            pass


def print_available_docs(doc_map):
    print("\nAvailable documents:")
    for name in sorted(doc_map):
        print(f" - {name}")


def print_sources(retrieved_docs):
    print("\nSources Referenced:")
    seen = set()
    for doc in retrieved_docs:
        citation = f"- {format_citation(doc.metadata)}"
        if citation not in seen:
            seen.add(citation)
            print(citation)


def print_compare_sources(docs_by_source):
    print("\nSources Referenced:")
    for display_name, docs in docs_by_source.items():
        print(f" {display_name}:")
        seen = set()
        for doc in docs:
            citation = f"  - {format_citation(doc.metadata)}"
            if citation not in seen:
                seen.add(citation)
                print(citation)


def handle_summarize(plan, vectorstore, doc_map, doc_years, doc_jurisdictions, doc_count, trail):
    """
    Whole-document summarization. Unlike the other intents, this doesn't
    go through hybrid_retrieve/rerank at all - it fetches every chunk for
    one target document and summarizes that directly (see
    summarization.py). The target is resolved with the exact same
    filters.py logic used everywhere else - a 'summarize:' target IS a
    filter target, there's just no separate question attached to it.
    """
    if not plan.filter_doc:
        if doc_count == 1:
            # Unambiguous - the only indexed document is the target.
            full_source = next(iter(doc_map.values()))
            display_name = next(iter(doc_map.keys()))
        else:
            print("\n Ambiguous Request Detected!")
            print(f"'summarize:' needs a target document, but there are {doc_count} documents indexed.")
            print("Example: 'summarize: journal5', 'summarize: filename.pdf', or 'summarize: 2021'")
            print_available_docs(doc_map)
            return
    else:
        source_filter, matched, err = resolve_filter(plan, doc_map, doc_years, doc_jurisdictions)
        if err:
            print(f"\n{err}")
            print_available_docs(doc_map)
            return
        if len(matched) > 1:
            print(f"\n'{plan.filter_doc}' matched multiple files: {', '.join(matched)}")
            print("Summarize needs exactly one target document - narrow it down.")
            print_available_docs(doc_map)
            return
        display_name = matched[0]
        full_source = doc_map[display_name]

    print(f"\nSummarizing {display_name}...")
    summary, stats = summarize_document(vectorstore, full_source, display_name)
    if summary is None:
        print("\nCouldn't find any indexed content for that document.")
        return

    print(f"\nSummary ({stats['method']}, {stats.get('doc_type', 'unknown')} document, {stats['chunk_count']} chunk(s) across {stats['page_count']} page(s)):")
    print(summary if summary else "The AI processed the document but returned a blank response.")
    print(f"\nSource: {display_name}")
    print("-" * 60)

    if summary:
        # Structured trail entry, mirroring the compare/broad entries:
        # a Details block records the run stats (the answer text doesn't
        # state method/doc-type/table counts), and the Sources section
        # cites the one document the summary covers.
        details = {
            "Method": stats["method"],
            "Document type": stats.get("doc_type", "unknown"),
            "Pages": stats["page_count"],
            "Chunks": stats["chunk_count"],
            "Verbatim facts extracted": stats.get("verbatim_fact_count", 0),
        }
        if stats.get("table_rows_filtered"):
            details["Table rows excluded"] = stats["table_rows_filtered"]
        total = stats.get("total_seconds")
        if total:
            details["Time"] = f"{int(total // 60)}m {int(total % 60):02d}s"

        source_meta = {"source": full_source}
        if doc_years.get(display_name):
            source_meta["year"] = doc_years[display_name]
        if doc_jurisdictions.get(display_name):
            source_meta["jurisdiction"] = doc_jurisdictions[display_name]

        trail.log(
            "summarize",
            f"summarize: {display_name}",
            summary,
            [Chunk(page_content="", metadata=source_meta)],
            details=details,
        )


def handle_compare(plan, vectorstore, hybrid_index, doc_map, doc_years, doc_jurisdictions, memory, trail):
    """
    Compare mode: resolves each comma-separated target independently (each
    must resolve to exactly ONE document - ambiguous or missing targets
    are reported by name rather than guessed at), retrieves for each
    target SEPARATELY (not fused into one pool - fusing would let one
    source's chunks dominate and silently crowd out the other), and hands
    the model clearly source-labeled context with an explicit instruction
    not to average disagreeing sources into one blended position.
    """
    resolved = {}  # display_name -> full_source
    errors = []
    for target in plan.compare_targets:
        source_filter, matched = build_source_filter(target, doc_map, doc_years, doc_jurisdictions)
        if not matched:
            errors.append(f"'{target}' matched no indexed document.")
        elif len(matched) > 1:
            errors.append(f"'{target}' matched multiple documents ({', '.join(matched)}) - be more specific.")
        else:
            resolved[matched[0]] = doc_map[matched[0]]

    if errors:
        print("\nCouldn't resolve all compare targets:")
        for e in errors:
            print(f" - {e}")
        print_available_docs(doc_map)
        return
    if len(resolved) < 2:
        print("\ncompare: needs at least two distinct, successfully-resolved documents.")
        return

    print(f"\nComparing: {', '.join(resolved.keys())}")
    print("Retrieving separately from each source (not fused)...")

    docs_by_source = {}
    for display_name, full_source in resolved.items():
        source_filter = {"source": full_source}
        fused = hybrid_index.hybrid_retrieve(plan.query, source_filter=source_filter, broad=False)
        docs_by_source[display_name] = sort_docs(hybrid_index.rerank(plan.query, fused, top_n=COMPARE_TOP_N_PER_SOURCE))

    formatted_context = format_compare_docs(docs_by_source)
    warn_if_context_too_large(formatted_context)

    if len(memory) > 0 and memory.enabled:
        print(f"(using {len(memory)} previous turn(s) as conversation context)")
    try:
        answer = generate_answer(plan.query, formatted_context, Intent.COMPARE, history_messages=memory.as_messages())
    except Exception as e:
        print(f"\nAn error occurred talking to Ollama: {e}")
        return

    print("\nAnswer:")
    print(answer if answer else "The AI processed the text but returned a blank response.")

    print_compare_sources(docs_by_source)
    print("-" * 60)

    if answer:
        memory.add(plan.query, answer)
        all_docs = [d for docs in docs_by_source.values() for d in docs]
        trail.log("compare", plan.raw_input, answer, all_docs)


def handle_query(user_input, vectorstore, hybrid_index, doc_map, doc_years, doc_jurisdictions, doc_count, memory, trail):
    # 1. Query Understanding + Intent
    try:
        plan = parse_query(user_input)
    except ValueError as e:
        print(str(e))
        return

    if plan.intent is Intent.SUMMARIZE:
        handle_summarize(plan, vectorstore, doc_map, doc_years, doc_jurisdictions, doc_count, trail)
        return
    if plan.intent is Intent.COMPARE:
        handle_compare(plan, vectorstore, hybrid_index, doc_map, doc_years, doc_jurisdictions, memory, trail)
        return
    if plan.filter_doc:
        print(f"Filter active: Searching only in file(s) matching '{plan.filter_doc}'")
    if plan.invalid_pages:
        print(f"[note] Ignoring page number(s) {plan.invalid_pages} - page numbering starts at 1.")

    # 2. Filter Resolution
    source_filter, matched, err = resolve_filter(plan, doc_map, doc_years, doc_jurisdictions)
    if err:
        print(f"\n{err}")
        print_available_docs(doc_map)
        return
    if plan.filter_doc and len(matched) > 1:
        print(f"Note: '{plan.filter_doc}' matched multiple files: {', '.join(matched)}")
    elif not plan.filter_doc and matched:
        print(f"Auto-filter active: Detected reference to file(s) -> {', '.join(matched)}")

    # 3. Retrieval, branched by intent
    if plan.intent is Intent.PAGE_SPECIFIC:
        if not source_filter and doc_count > 1:
            pages_display = ", ".join(str(p + 1) for p in plan.page_numbers)
            print("\n Ambiguous Request Detected!")
            print(f"You asked about page(s) {pages_display}, but there are {doc_count} documents in the database.")
            print("Please specify which document you mean using the filter syntax.")
            print("Example: filter: journal1.pdf | " + plan.query)
            print_available_docs(doc_map)
            return

        pages_display = ", ".join(str(p + 1) for p in plan.page_numbers)
        print(f"Page routing active: Targeting exactly page(s) {pages_display}")
        retrieved_docs = get_chunks_by_pages(vectorstore, plan.page_numbers, source_filter)
        # Reorder only - this set is already complete/exact, don't truncate it.
        retrieved_docs = hybrid_index.rerank(plan.query, retrieved_docs, top_n=None)

    else:
        broad = plan.intent is Intent.BROAD
        if broad:
            print("Broad mode: searching for a diverse spread across the document(s).")
        print("\nSearching context and generating response...")

        fused = hybrid_index.hybrid_retrieve(plan.query, source_filter=source_filter, broad=broad)
        top_n = BROAD_TOP_N if broad else HYBRID_TOP_N
        retrieved_docs = sort_docs(hybrid_index.rerank(plan.query, fused, top_n=top_n))

    if not retrieved_docs:
        print("\nAnswer:")
        print("I couldn't find any relevant text in the documents to answer that.")
        return

    # 4. Context Assembly
    formatted_context = format_docs(retrieved_docs)
    warn_if_context_too_large(formatted_context)

    # 5. Prompt Selection (by intent) + Generation
    if len(memory) > 0 and memory.enabled:
        print(f"(using {len(memory)} previous turn(s) as conversation context)")
    try:
        answer = generate_answer(plan.query, formatted_context, plan.intent, history_messages=memory.as_messages())
    except Exception as e:
        print(f"\nAn error occurred talking to Ollama: {e}")
        return

    print("\nAnswer:")
    print(answer if answer else "The AI processed the text but returned a blank response. Try rephrasing the question.")

    # 6. Citation Formatting
    print_sources(retrieved_docs)
    print("-" * 60)

    if answer:
        # plan.query (prefix-stripped) rather than the raw input, so a
        # 'filter: x | ...' pipe or 'broad:' prefix from this turn doesn't
        # show up as odd, non-conversational phrasing in a later prompt.
        memory.add(plan.query, answer)
        mode = {Intent.PAGE_SPECIFIC: "page", Intent.BROAD: "broad", Intent.FACTUAL: "ask"}[plan.intent]
        trail.log(mode, plan.query, answer, retrieved_docs)


def main():
    print("=" * 60)
    print(" Academic Thesis / Legal Research RAG Assistant")
    print(" (hybrid BM25+vector retrieval, cross-encoder re-ranked)")
    print("=" * 60)

    if not os.path.exists(DOC_FOLDER):
        os.makedirs(DOC_FOLDER)
        print(f"\nCreated folder '{DOC_FOLDER}'. Please place your document files (PDF or DOCX) there and restart.")
        return

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        try:
            vectorstore = index_folder(embeddings)
        except Exception as e:
            print(f"\nCouldn't build the index because embedding generation failed: {e}")
            print("This means Ollama/the embedding model isn't reachable, not that your documents are missing.")
            print("Check that Ollama is running and reachable, then run this again.")
            return
        if vectorstore is None:
            print(f"\nError: No documents found in '{DOC_FOLDER}'. Add PDF or DOCX files and run again.")
            return
    else:
        print("\nLoading existing Vector Database from disk...")
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        print("Vector Database loaded successfully.")
        doc_map = get_indexed_documents(vectorstore)
        try:
            if sync_new_files(vectorstore, doc_map):
                # New files were indexed – the BM25 index and the
                # doc_map/years/jurisdictions/count below are rebuilt
                # right after this block, so no refresh needed here.
                print("Synced new files into the existing index.")
        except Exception as e:
            print(f"\n[warning] Couldn't sync new files ({e}); continuing with the existing index as-is.")

    hybrid_index = HybridIndex(vectorstore)
    hybrid_index.refresh_bm25()
    memory = ConversationMemory(max_turns=MAX_HISTORY_TURNS)
    trail = ResearchTrail(RESEARCH_TRAIL_PATH)

    doc_map = get_indexed_documents(vectorstore)
    doc_years = get_indexed_years(vectorstore)
    doc_jurisdictions = get_indexed_jurisdictions(vectorstore)
    doc_count = len(doc_map)

    # Where the live index actually lives. Normally PERSIST_DIR, but if a
    # reindex swap can't remove the old directory (see below), the freshly
    # built temp index becomes the live one until the next successful swap
    # can restore the canonical location.
    live_dir = PERSIST_DIR

    print("\n" + "=" * 60)
    print(f"System Ready! ({doc_count} documents indexed)")
    print(" - Ask normal questions: 'What is the main methodology?'")
    print(" - Ask about specific pages: 'What does page 5 say?' or 'compare page 10 and page 23'")
    print(" - Mention a numbered file directly: 'summarize journal5' (auto-detected)")
    print(" - Or a publication year: 'what does the 2021 paper say' (extracted from each document's front matter)")
    print(" - Or target explicitly: 'filter: filename.pdf | your question'")
    print("   (also accepts a year or jurisdiction: 'filter: 2021 | ...', 'filter: indonesia | ...')")
    print(" - Prefix with 'broad:' for a wide, diverse sweep instead of a focused lookup")
    print(" - Prefix with 'summarize:' for a whole-document summary (not top-k retrieval)")
    print("   e.g. 'summarize: journal5', 'summarize: journal5.pdf', or 'summarize: 2021'")
    print(" - Prefix with 'compare:' to compare two+ sources without averaging them")
    print("   e.g. 'compare: caseA, caseB | how do they define proportionality'")
    print(f" - Follow-up questions remember the last {MAX_HISTORY_TURNS} answer(s).")
    print("   Type 'memory off'/'memory on' to toggle, 'forget' to clear what's remembered.")
    print(f" - Every answer is logged to {RESEARCH_TRAIL_PATH}. Type 'export citations' to")
    print(f"   write every source cited this session to {CITATION_EXPORT_PATH}.")
    print(" - Type 'reindex' to rebuild from scratch, or 'exit' to quit.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nAsk a question: ").strip()

            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("\nClosing session. Goodbye!")
                break
            if user_input.lower() in ["forget", "reset"]:
                memory.clear()
                print("Conversation memory cleared.")
                continue
            if user_input.lower() in ["memory off", "memory: off"]:
                memory.enabled = False
                print("Conversation memory OFF - stored history is kept but won't be used until turned back on.")
                continue
            if user_input.lower() in ["memory on", "memory: on"]:
                memory.enabled = True
                print(f"Conversation memory ON ({len(memory)} turn(s) currently remembered).")
                continue
            if user_input.lower() in ["export citations", "export: citations"]:
                count = trail.export_citations(CITATION_EXPORT_PATH)
                if count is None:
                    print("Nothing cited yet this session.")
                else:
                    print(f"Exported {count} unique citation(s) to {CITATION_EXPORT_PATH}.")
                continue
            if user_input.lower() == "reindex":
                # Build into a temp directory FIRST, and only touch the
                # live index once that build has actually succeeded.
                tmp_dir = PERSIST_DIR.rstrip("/\\") + "_rebuilding"
                if os.path.normcase(os.path.abspath(tmp_dir)) == os.path.normcase(os.path.abspath(live_dir)):
                    # The live index IS the temp directory (an earlier
                    # swap couldn't remove the old one and we've been
                    # using the temp index since). Building into that same
                    # directory would destroy the index currently
                    # answering questions - use a non-colliding name.
                    tmp_dir = PERSIST_DIR.rstrip("/\\") + "_rebuilding2"
                _safe_rmtree(tmp_dir)  # leftovers from a previously interrupted reindex, if any

                print(f"\nBuilding a fresh index at '{tmp_dir}'; your current index stays live until this succeeds...")
                try:
                    new_vectorstore = index_folder(embeddings, persist_dir=tmp_dir)
                except Exception as e:
                    print(f"\nReindex failed while generating embeddings: {e}")
                    print("Your existing index was NOT touched and is still usable.")
                    _safe_rmtree(tmp_dir)
                    continue

                if new_vectorstore is None:
                    print(f"\nNo documents found in '{DOC_FOLDER}' - nothing to index. Existing index left untouched.")
                    _safe_rmtree(tmp_dir)
                    continue

                # Build succeeded - now it's safe to retire the old index.
                # Close the old Chroma client BEFORE deleting anything:
                # Client.close() stops the shared System singleton and
                # releases chroma.sqlite3/segment file handles, so the
                # rmtree below actually succeeds (see _close_chroma_client).
                _close_chroma_client(vectorstore)
                del vectorstore
                del hybrid_index
                gc.collect()

                if not _safe_rmtree(live_dir):
                    print(f"\nCouldn't fully remove the old index at '{live_dir}' - a file may still be locked.")
                    print(f"The NEW index built fine and is sitting at '{tmp_dir}'. We'll use it for this session.")
                    print(f"To make it permanent, close the program, delete '{live_dir}' and rename")
                    print(f"'{tmp_dir}' to '{PERSIST_DIR}', then restart.")
                    vectorstore = new_vectorstore
                    live_dir = tmp_dir
                else:
                    final_dir = PERSIST_DIR
                    if os.path.exists(final_dir):
                        # Stale original directory from an earlier failed
                        # swap - no longer live, so try to clear it too
                        # and restore the canonical location.
                        _safe_rmtree(final_dir)
                    if os.path.exists(final_dir):
                        final_dir = live_dir
                        print(f"\nNote: '{PERSIST_DIR}' still couldn't be removed, so the rebuilt index")
                        print(f"was placed at '{final_dir}' instead.")
                    try:
                        # Close the temp-dir client BEFORE the move: on
                        # Windows, renaming a directory fails with
                        # "Access is denied" while any file inside it is
                        # still open (tested: shutil.move on a directory
                        # holding an open chroma.sqlite3/data_level0.bin).
                        _close_chroma_client(new_vectorstore)
                        shutil.move(tmp_dir, final_dir)
                        vectorstore = Chroma(persist_directory=final_dir, embedding_function=embeddings)
                        live_dir = final_dir
                    except Exception as e:
                        print(f"\n[warning] Couldn't move the rebuilt index into place ({e}); using it from '{tmp_dir}' for this session.")
                        print(f"To make it permanent, close the program, delete '{live_dir}' and rename")
                        print(f"'{tmp_dir}' to '{PERSIST_DIR}', then restart.")
                        try:
                            # The temp client was already closed above, so
                            # reopen it fresh at its current location.
                            vectorstore = Chroma(persist_directory=tmp_dir, embedding_function=embeddings)
                        except Exception:
                            vectorstore = new_vectorstore
                        live_dir = tmp_dir

                hybrid_index = HybridIndex(vectorstore)
                hybrid_index.refresh_bm25()
                memory.clear()  # old turns may reference content that no longer exists post-rebuild
                doc_map = get_indexed_documents(vectorstore)
                doc_years = get_indexed_years(vectorstore)
                doc_jurisdictions = get_indexed_jurisdictions(vectorstore)
                doc_count = len(doc_map)
                print(f"Reindexed. {doc_count} documents now indexed.")
                continue

            handle_query(user_input, vectorstore, hybrid_index, doc_map, doc_years, doc_jurisdictions, doc_count, memory, trail)

        except KeyboardInterrupt:
            print("\nSession interrupted. Bye!")
            break
        except Exception as e:
            print(f"\nAn error occurred during generation: {e}")


if __name__ == "__main__":
    main()
