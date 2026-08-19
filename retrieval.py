"""
Retrieval: hybrid (BM25 + vector) search for narrow/broad intents, direct
metadata fetch for page-specific intent, whole-document fetch for
summarization, and flashrank cross-encoder re-ranking shared across the
narrow/broad/page-specific paths - matching the diagram, where those three
branches converge into a single "Cross-Encoder Re-ranking" box before
context assembly.
"""
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

try:
    from langchain_classic.retrievers import EnsembleRetriever  # langchain >= 1.0
except ImportError:
    from langchain.retrievers import EnsembleRetriever  # langchain 0.x

from flashrank import Ranker, RerankRequest

from config import (
    SIMILARITY_K, BM25_K,
    BROAD_K, BROAD_FETCH_K, BROAD_BM25_K,
    BM25_WEIGHT, VECTOR_WEIGHT,
    RERANK_MODEL, RERANK_CACHE_DIR,
)


# Cap on the direct-metadata fallback fetch below, so a broken-retrieval
# recovery on a large document doesn't itself become a huge, unranked
# context dump - the reranker still sorts these by real relevance
# afterward, this just bounds worst-case size going in.
_FALLBACK_FETCH_LIMIT = 50


@dataclass
class Chunk:
    """Minimal stand-in for a langchain Document, used for chunks fetched
    directly via vectorstore.get() rather than through a retriever."""
    page_content: str
    metadata: dict = field(default_factory=dict)


def sort_docs(docs):
    return sorted(docs, key=lambda d: (
        d.metadata.get("source", ""),
        d.metadata.get("page", 0),
        d.metadata.get("chunk_index", 0)
    ))


def get_chunks_by_pages(vectorstore, pages, source_filter=None):
    """
    Direct Chunk Fetch (by page metadata): fetch every indexed chunk for
    the given (0-indexed) page numbers, bypassing similarity search
    entirely. When exact pages are named we already know precisely which
    chunks are wanted, so fetch them by metadata rather than risk a
    k/fetch_k cutoff or MMR diversity re-ranking dropping part of a page.
    """
    page_clause = {"page": pages[0]} if len(pages) == 1 else {"page": {"$in": pages}}
    where = page_clause if source_filter is None else {"$and": [source_filter, page_clause]}

    result = vectorstore.get(where=where, include=["documents", "metadatas"])
    chunks = [
        Chunk(page_content=doc, metadata=meta)
        for doc, meta in zip(result.get("documents", []), result.get("metadatas", []))
    ]
    return sort_docs(chunks)


def get_chunks_by_source(vectorstore, full_source):
    """
    Fetch every indexed chunk belonging to one document, in page order -
    the complete set, no relevance cutoff. Used for whole-document
    summarization ('summarize:'), where the point is "the whole document",
    not a top-k search result.
    """
    result = vectorstore.get(where={"source": full_source}, include=["documents", "metadatas"])
    chunks = [
        Chunk(page_content=doc, metadata=meta)
        for doc, meta in zip(result.get("documents", []), result.get("metadatas", []))
    ]
    return sort_docs(chunks)


class HybridIndex:
    """
    Owns the BM25 half of retrieval (langchain's Chroma wrapper only gives
    us vector search) and the cross-encoder reranker, and exposes hybrid
    fusion + re-ranking on top of a given vectorstore.

    BM25Retriever has no add_documents - it's built once from a fixed
    corpus - so refresh_bm25() must be called after indexing/syncing/
    reindexing to keep it in sync with the vector store.
    """

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self._all_docs = []
        self._bm25 = None
        self._reranker = None       # lazy: only pay flashrank's model-load cost if actually used
        self._reranker_unavailable = False  # sticky flag once init fails, so we don't retry a doomed download every query

    def refresh_bm25(self):
        """Rebuild the BM25 index from the vectorstore's current contents.
        Call after index_folder / sync_new_files / reindex."""
        result = self.vectorstore.get(include=["documents", "metadatas"])
        self._all_docs = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(result.get("documents", []), result.get("metadatas", []))
        ]
        self._bm25 = BM25Retriever.from_documents(self._all_docs) if self._all_docs else None

    def _allowed_sources(self, source_filter):
        """Extract the set of source paths a filter restricts to, or None
        for no restriction. Shared by _scoped_bm25 and
        _filtered_chunk_count so both interpret a filter identically."""
        if source_filter is None or "source" not in source_filter:
            return None
        val = source_filter["source"]
        return set(val.get("$in", [val])) if isinstance(val, dict) else {val}

    def _filtered_chunk_count(self, source_filter):
        """How many indexed chunks actually match `source_filter`, or None
        if there's no filter (meaning: don't cap anything, the full corpus
        is in play). Used to keep k/fetch_k from ever exceeding what a
        filtered query can actually satisfy - see hybrid_retrieve."""
        allowed = self._allowed_sources(source_filter)
        if allowed is None:
            return None
        return sum(1 for d in self._all_docs if d.metadata.get("source") in allowed)

    def _scoped_bm25(self, source_filter, k):
        """
        Build a BM25Retriever over just the filtered subset of the corpus.
        BM25Retriever doesn't support Chroma-style metadata filters
        natively, so when a document filter is active we pre-filter the
        doc list in Python before handing it to BM25 - the corpus sizes
        here (a handful of PDFs) make rebuilding per-query cheap. If your
        library grows into the hundreds of PDFs, cache these per-filter
        instead of rebuilding every call.
        """
        if source_filter is None:
            if self._bm25 is not None:
                self._bm25.k = k
            return self._bm25

        allowed = self._allowed_sources(source_filter)
        scoped_docs = [d for d in self._all_docs if d.metadata.get("source") in allowed]
        if not scoped_docs:
            return None
        retriever = BM25Retriever.from_documents(scoped_docs)
        retriever.k = k
        return retriever

    def hybrid_retrieve(self, query, source_filter=None, broad=False):
        """
        Fused BM25 + vector retrieval. Narrow (default) uses plain
        similarity; broad ('broad:' prefix) swaps the vector side for MMR
        to trade relevance for topic diversity. BM25 stays keyword-based
        either way - it's what catches exact terms (author names,
        technical vocabulary) that embeddings can blur across a diverse
        MMR sweep.

        Chroma's HNSW index can throw ("Error finding id" and similar
        messages, depending on version) when a query's requested k exceeds
        the number of candidates actually available under the active
        metadata filter - this is documented Chroma behavior, not a bug
        specific to this project (see
        https://docs.trychroma.com/docs/overview/troubleshooting: "This
        error happens when the HNSW index fails to retrieve the requested
        number of results for a query, given its structure and your
        data... decrease the number of results you request"). This matters
        most for anything that scopes retrieval to a single document -
        compare mode queries one document at a time, and a shorter paper
        can easily have fewer chunks than SIMILARITY_K/BROAD_K requests.

        So: whenever a filter is active, k and fetch_k are capped to the
        actual number of matching chunks *before* querying, which is
        exactly Chroma's own recommended fix and prevents most of these
        errors from happening at all. The try/except + _fallback_fetch
        below still exists as a safety net for whatever this cap doesn't
        catch (e.g. genuine index corruption rather than an over-large k).
        """
        available = self._filtered_chunk_count(source_filter)

        vector_kwargs = {"filter": source_filter} if source_filter else {}
        if broad:
            k = min(BROAD_K, available) if available is not None else BROAD_K
            fetch_k = min(BROAD_FETCH_K, available) if available is not None else BROAD_FETCH_K
            fetch_k = max(fetch_k, k)  # MMR requires fetch_k >= k
            vector_kwargs.update(k=k, fetch_k=fetch_k)
            vector_retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs=vector_kwargs)
            bm25_k = min(BROAD_BM25_K, available) if available is not None else BROAD_BM25_K
        else:
            k = min(SIMILARITY_K, available) if available is not None else SIMILARITY_K
            vector_kwargs.update(k=k)
            vector_retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs=vector_kwargs)
            bm25_k = min(BM25_K, available) if available is not None else BM25_K

        if available == 0:
            # Filter matches nothing at all - no point even querying.
            return []

        bm25 = self._scoped_bm25(source_filter, max(bm25_k, 1))

        try:
            if bm25 is None:
                # No BM25 corpus yet (fresh/empty index) - fall back to
                # vector-only rather than erroring out.
                return sort_docs(vector_retriever.invoke(query))

            ensemble = EnsembleRetriever(
                retrievers=[bm25, vector_retriever],
                weights=[BM25_WEIGHT, VECTOR_WEIGHT],
            )
            return ensemble.invoke(query)
        except Exception as e:
            if broad:
                print(f"[warning] broad (MMR) retrieval failed ({e}); retrying as a focused search instead.")
                return self.hybrid_retrieve(query, source_filter=source_filter, broad=False)
            print(f"[warning] hybrid retrieval failed ({e}); falling back to a direct fetch instead.")
            return self._fallback_fetch(source_filter)

    def _fallback_fetch(self, source_filter):
        """
        Last resort when both the fused retriever and (for broad mode)
        the focused retry have failed: fetch by metadata filter directly,
        bypassing Chroma's similarity/MMR query path entirely, since
        that's the part that's throwing. Only useful when `source_filter`
        actually narrows things down - with no filter at all this would
        mean pulling the entire corpus unranked, which defeats the
        purpose and could blow the context window, so in that case this
        just gives up and returns [] same as before.
        """
        if source_filter is None:
            return []
        try:
            result = self.vectorstore.get(where=source_filter, include=["documents", "metadatas"])
        except Exception as e:
            print(f"[warning] fallback fetch also failed ({e}); returning no results for this query.")
            return []

        chunks = [
            Chunk(page_content=doc, metadata=meta)
            for doc, meta in zip(result.get("documents", []), result.get("metadatas", []))
        ]
        return sort_docs(chunks)[:_FALLBACK_FETCH_LIMIT]

    def rerank(self, query, docs, top_n=None):
        """
        Cross-encoder re-ranking via flashrank. If top_n is None, reorder
        only and keep every doc - used for the page-specific path, where
        get_chunks_by_pages already fetched the complete, exact set and we
        don't want reranking to silently drop part of a requested page.
        For hybrid-retrieval paths, top_n trims the fused BM25+vector list
        down to what's worth spending context-window tokens on.
        """
        if not docs:
            return docs
        if self._reranker_unavailable:
            return docs[:top_n] if top_n else docs

        if self._reranker is None:
            try:
                self._reranker = Ranker(model_name=RERANK_MODEL, cache_dir=RERANK_CACHE_DIR)
            except Exception as e:
                # Most likely cause: first-run model download couldn't reach
                # Hugging Face (offline machine, restrictive firewall/proxy).
                # Degrade to un-reranked order rather than crashing the query.
                print(f"[warning] cross-encoder model unavailable ({e}); continuing without re-ranking.")
                self._reranker_unavailable = True
                return docs[:top_n] if top_n else docs

        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
        try:
            results = self._reranker.rerank(RerankRequest(query=query, passages=passages))
        except Exception as e:
            print(f"[warning] cross-encoder rerank failed ({e}); falling back to un-reranked order.")
            return docs[:top_n] if top_n else docs

        ordered = [docs[r["id"]] for r in results]
        return ordered[:top_n] if top_n else ordered
