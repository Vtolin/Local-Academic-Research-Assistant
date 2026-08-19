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


# Matches every page number named in a question, including lists and
# ranges: "page 5", "pages 5 and 7", "page 3, 5, and 7", "pages 5-7",
# and the Indonesian equivalents ("halaman 5 dan 7"). The number list is
# captured as one group and split/expanded by _parse_page_numbers, so a
# second bare number after "and"/"," is NOT silently dropped the way the
# old one-number-per-mention regex dropped it ("page 5 and 7" used to
# retrieve only page 5).
PAGE_MENTION_RE = re.compile(
    r"\b(?:pages?|halaman|hal\.?)\s+"
    r"(\d+(?:\s*[-–]\s*\d+)?"
    r"(?:(?:\s*,\s*(?:(?:and|dan)\s+)?|\s+(?:and|dan)\s+)"
    r"\d+(?:\s*[-–]\s*\d+)?)*)",
    re.IGNORECASE,
)

# Ranges spanning more pages than this are collapsed to their two
# endpoints instead of expanded - 'pages 1-10000' should not become a
# 10,000-entry page fetch.
_PAGE_RANGE_MAX_SPAN = 200

# Splits a captured number list ("5 and 7", "3, 5-7, dan 9") on
# commas/"and"/"dan" into its individual entries.
_PAGE_LIST_SPLIT_RE = re.compile(r"\s*(?:,|and|dan)\s*", re.IGNORECASE)

_PAGE_RANGE_RE = re.compile(r"(\d+)\s*[-–]\s*(\d+)")


class Intent(Enum):
    PAGE_SPECIFIC = auto()   # exact page(s) named -> direct chunk fetch
    BROAD = auto()           # 'broad:' prefix -> literature-review sweep
    FACTUAL = auto()         # default -> focused factual/compare/summarize lookup
    SUMMARIZE = auto()       # 'summarize:' prefix -> whole-document summary
    COMPARE = auto()         # 'compare:' prefix -> per-source retrieval, conflict-aware synthesis


def _parse_page_numbers(query: str) -> list:
    """
    All page numbers named in `query`, in mention order (duplicates
    removed by the caller). Supports 'page 5', 'pages 5 and 7',
    'page 3, 5, and 7', 'halaman 5 dan 7', and ranges ('pages 5-7'
    expands to 5, 6, 7; a range wider than _PAGE_RANGE_MAX_SPAN is
    collapsed to its two endpoints rather than expanded).
    """
    numbers = []
    for m in PAGE_MENTION_RE.finditer(query):
        # Normalize ', and N' / ', dan N' to ', N' first so a bare
        # re.split doesn't emit an empty part between the ',' and 'and'.
        body = re.sub(r",\s+(?:and|dan)(?=\s+\d)", ",", m.group(1), flags=re.IGNORECASE)
        for part in _PAGE_LIST_SPLIT_RE.split(body):
            part = part.strip()
            if not part:
                continue
            rng = _PAGE_RANGE_RE.fullmatch(part)
            if rng:
                lo, hi = sorted((int(rng.group(1)), int(rng.group(2))))
                if hi - lo > _PAGE_RANGE_MAX_SPAN:
                    numbers.extend((lo, hi))
                else:
                    numbers.extend(range(lo, hi + 1))
            else:
                numbers.append(int(part))
    return numbers


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

    raw_page_nums = _parse_page_numbers(query)
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
