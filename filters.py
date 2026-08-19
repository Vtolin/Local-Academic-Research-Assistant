"""
Filter Resolution: turning a user-named file reference (explicit
'filter: x | q' or an implicit mention like "summarize journal5") into a
Chroma metadata filter. Also resolves references by extracted publication
year (see metadata_extraction.py) when filenames don't encode it.
"""
import os
import re

from metadata_extraction import YEAR_TOKEN_RE


def _normalize(s):
    """Collapse whitespace/underscore/hyphen so 'journal 4', 'journal_4',
    and 'journal-4' all compare equal to 'journal4'. Used on both sides of
    a match (the typed needle and each candidate filename) so explicit
    'filter:'/'summarize:' targets are at least as forgiving as the
    implicit auto-detect path below, which already did this."""
    return re.sub(r"[\s_\-]+", "", s.lower())


def build_source_filter(filter_doc, doc_map, doc_years=None, doc_jurisdictions=None):
    """
    Chroma's metadata `where` filter does NOT support `$contains`, so
    substring matching is done in Python, then turned into an exact-match
    $in list for Chroma.

    A naive substring check has a sneaky bug for numbered filenames:
    "journal1" IS a plain substring of "journal10.pdf", "journal11.pdf",
    etc. Typing `filter: journal1 | ...` would silently pull in every
    journal1X file too. The fix only applies when the search term itself
    ends in a digit (that's the exact situation where a longer number could
    be mistaken for it) - a purely alphabetic term like "journal" is left
    free to broadly match "journal1.pdf", "journal55.pdf", and so on, since
    that broad-prefix behavior is intentional and useful.

    Matching is done on normalized text (see _normalize) so "journal 4",
    "journal_4", and "journal4" are all treated as the same target.

    If nothing matches by filename, falls back in order to:
      1. Publication year (e.g. "2021"), if `doc_years` is given and the
         term looks like a bare 4-digit year.
      2. Jurisdiction (e.g. "australia" matching "Australia (High Court)"),
         if `doc_jurisdictions` is given - substring match, case-insensitive.

    Jurisdiction matching is EXPLICIT-only (this function, not the implicit
    auto-detect below) - see detect_implicit_doc_filter's docstring for why
    jurisdiction names aren't auto-scanned from free-form question text.

    Returns (filter_dict_or_None, matched_relative_paths).
    """
    needle = _normalize(filter_doc)
    if not needle:
        return None, []

    escaped = re.escape(needle)
    pattern = re.compile(escaped + r"(?!\d)") if needle[-1].isdigit() else re.compile(escaped)

    matches = [name for name in doc_map if pattern.search(_normalize(name))]
    if not matches and doc_years and YEAR_TOKEN_RE.match(filter_doc.strip()):
        matches = _match_by_year(int(filter_doc.strip()), doc_years)
    if not matches and doc_jurisdictions:
        matches = _match_by_jurisdiction(filter_doc.strip(), doc_jurisdictions)
    if not matches:
        return None, []
    sources = [doc_map[name] for name in matches]
    if len(sources) == 1:
        return {"source": sources[0]}, matches
    return {"source": {"$in": sources}}, matches


def _match_by_year(year, doc_years):
    return [name for name, y in doc_years.items() if y == year]


def _match_by_jurisdiction(term, doc_jurisdictions):
    """Word-boundary substring match, not a naive 'in' check - "US" must
    not match inside "Australia" the way a plain substring test would
    ("aUStralia" literally contains "us"). Same class of bug as the
    filename journal1-inside-journal10 issue elsewhere in this file."""
    pattern = re.compile(rf"\b{re.escape(term.lower())}\b")
    return [name for name, j in doc_jurisdictions.items() if j and pattern.search(j.lower())]


def detect_implicit_doc_filter(query, doc_map, doc_years=None):
    """
    Scans the query for natural mentions of indexed document names - e.g.
    "journal5", "journal 5", or "journal5.pdf" - without requiring the
    explicit 'filter:' syntax. Also recognizes a bare publication year
    (e.g. "what does the 2021 paper say") when `doc_years` is given, since
    sequential filenames don't encode year at all.

    Deliberately does NOT auto-scan for jurisdiction mentions the way it
    does for years - "what happened in Australia" as an ordinary question
    would incorrectly narrow retrieval to Australian-sourced documents
    only. A year is an unambiguous 4-digit token; a jurisdiction name is
    an ordinary word/phrase that appears in all kinds of unrelated
    sentences, so jurisdiction filtering is explicit-only (build_source_filter,
    via 'filter: australia | ...' or 'compare:').

    Two safeguards on the filename side:

    1. Restricted to filenames containing at least one digit (journal5,
       paper12, study_2023, ...). Without this, a document named something
       like "study.pdf" or "review.pdf" would silently hijack every
       question that happens to use the ordinary English word "study" or
       "review" - which, in academic writing, is most of them. Descriptive,
       non-numbered filenames can still be targeted explicitly with
       'filter: <name> | ...'.

    2. All matching uses \\b word boundaries, so "journal1" can never
       accidentally match inside "journal10" (a plain substring check
       would, since "journal1" literally IS a substring of "journal10").

    Collects every distinct document mentioned rather than stopping at the
    first hit, so "compare journal5 and journal9" retrieves both - and a
    year match is unioned in alongside any filename matches rather than
    replacing them.
    """
    query_clean = query.lower()
    matched_names, matched_sources = [], []

    for rel_path, full_path in doc_map.items():
        filename = os.path.basename(rel_path).lower()
        name_no_ext = os.path.splitext(filename)[0]

        if not any(ch.isdigit() for ch in name_no_ext):
            continue  # plain-word filename - too high a false-positive risk to auto-match

        spaced_name = re.sub(r"([a-zA-Z]+)(\d+)", r"\1 \2", name_no_ext)   # "journal5" -> "journal 5"
        normalized = re.sub(r"[_\-]+", " ", name_no_ext)                   # "study_2023" -> "study 2023"

        candidates = {filename, name_no_ext, spaced_name, normalized}
        if any(re.search(rf"\b{re.escape(c)}\b", query_clean) for c in candidates if c):
            matched_names.append(rel_path)
            matched_sources.append(full_path)

    if doc_years:
        for year_str in re.findall(r"\b(19\d{2}|20\d{2})\b", query_clean):
            for name in _match_by_year(int(year_str), doc_years):
                if name not in matched_names:
                    matched_names.append(name)
                    matched_sources.append(doc_map[name])

    if not matched_sources:
        return None, []
    if len(matched_sources) == 1:
        return {"source": matched_sources[0]}, matched_names
    return {"source": {"$in": matched_sources}}, matched_names


def resolve_filter(plan, doc_map, doc_years=None, doc_jurisdictions=None):
    """
    Resolve a QueryPlan's document filter: try the explicit 'filter:'
    target first, falling back to implicit detection when none was given.
    Returns (source_filter_or_None, matched_names, error_message_or_None).
    """
    if plan.filter_doc:
        source_filter, matched = build_source_filter(plan.filter_doc, doc_map, doc_years, doc_jurisdictions)
        if source_filter is None:
            return None, [], f"No indexed file matches '{plan.filter_doc}'."
        return source_filter, matched, None

    source_filter, matched = detect_implicit_doc_filter(plan.query, doc_map, doc_years)
    return source_filter, matched, None
