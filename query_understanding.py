"""
Query Understanding + Intent classification.

Maps to the "Query Understanding" -> "Intent?" boxes in the pipeline
diagram, plus a fourth intent (SUMMARIZE) added on top of the original
three. Intent is resolved with cheap regex heuristics, NOT an extra LLM
call - a router model adds a full generation round-trip for something
regex already resolves correctly, and it keeps the explicit prefixes you
already rely on ('filter: x | q', 'broad: q', 'summarize: x').
"""
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# Matches every "page N" mention, not just the first - "correlation between
# page 10 and page 23" needs both, not just whichever comes first. Also
# matches the Indonesian equivalents "halaman N" / "hal. N" / "hal N",
# since a question typed in Indonesian ("apa isi halaman 5") should route
# the same way an English one does.
PAGE_MENTION_RE = re.compile(r"\b(?:page|halaman|hal\.?)\s+(\d+)\b", re.IGNORECASE)


class Intent(Enum):
    PAGE_SPECIFIC = auto()   # exact page(s) named -> direct chunk fetch
    BROAD = auto()           # 'broad:' prefix -> literature-review sweep
    FACTUAL = auto()         # default -> focused factual/compare/summarize lookup
    SUMMARIZE = auto()       # 'summarize:' prefix -> whole-document summary
    COMPARE = auto()         # 'compare:' prefix -> per-source retrieval, conflict-aware synthesis


@dataclass
class QueryPlan:
    raw_input: str
    query: str                              # cleaned query text (prefixes stripped)
    filter_doc: Optional[str] = None        # explicit doc target - from 'filter: X |' or 'summarize: X'
    compare_targets: list = field(default_factory=list)  # 'compare: X, Y, ... |' targets, unresolved
    broad_mode: bool = False
    page_numbers: list = field(default_factory=list)     # 0-indexed
    invalid_pages: list = field(default_factory=list)
    intent: Intent = Intent.FACTUAL


def parse_query(user_input: str) -> QueryPlan:
    """
    Classify intent and strip prefixes. 'summarize:' and 'compare:' are
    checked first and handled as self-contained commands - unlike
    'filter:'/'broad:', they don't compose with the rest of the pipeline,
    so they return immediately rather than falling through to page/broad
    detection.

    For the remaining three intents, order matters: 'filter:' is stripped
    first so 'broad:' detection and page detection never see it as literal
    text, and 'broad:' is stripped next so it can never leak into the
    implicit-filter scan or the final prompt.

    Raises ValueError on malformed 'filter:'/'compare:' syntax (missing
    '|', or fewer than two comma-separated targets for 'compare:').
    """
    query = user_input

    if query.lower().startswith("summarize:"):
        target = query[len("summarize:"):].strip()
        return QueryPlan(
            raw_input=user_input,
            query=target,
            filter_doc=target or None,
            intent=Intent.SUMMARIZE,
        )

    if query.lower().startswith("compare:"):
        remainder = query[len("compare:"):].strip()
        parts = remainder.split("|", 1)
        if len(parts) < 2:
            raise ValueError("Invalid compare syntax. Use: compare: doc1, doc2 | your question")
        targets_str, question = parts[0].strip(), parts[1].strip()
        targets = [t.strip() for t in targets_str.split(",") if t.strip()]
        if len(targets) < 2:
            raise ValueError("compare: needs at least two documents, separated by commas. Use: compare: doc1, doc2 | your question")
        return QueryPlan(
            raw_input=user_input,
            query=question,
            compare_targets=targets,
            intent=Intent.COMPARE,
        )

    filter_doc = None
    if query.lower().startswith("filter:"):
        parts = query[len("filter:"):].split("|", 1)
        if len(parts) < 2:
            raise ValueError("Invalid filter syntax. Use: filter: filename.pdf | your question")
        filter_doc, query = parts[0].strip(), parts[1].strip()

    broad_mode = query.lower().startswith("broad:")
    if broad_mode:
        query = query[len("broad:"):].strip()

    raw_page_nums = [int(n) for n in PAGE_MENTION_RE.findall(query)]
    invalid_pages = sorted({n for n in raw_page_nums if n < 1})
    page_numbers = sorted({n - 1 for n in raw_page_nums if n >= 1})  # 0-indexed for metadata

    if page_numbers:
        intent = Intent.PAGE_SPECIFIC
    elif broad_mode:
        intent = Intent.BROAD
    else:
        intent = Intent.FACTUAL

    return QueryPlan(
        raw_input=user_input,
        query=query,
        filter_doc=filter_doc,
        broad_mode=broad_mode,
        page_numbers=page_numbers,
        invalid_pages=invalid_pages,
        intent=intent,
    )
