"""
Publication-year extraction (best-effort metadata for filtering).

Sequential filenames like journal4.pdf don't encode publication year, but
an academic PDF's front matter usually does - a copyright line, a
"Received/Accepted" date, a DOI stamp, volume/issue info. This scans the
first couple of pages for 4-digit years in a plausible range and scores
each occurrence by proximity to keywords that typically accompany a real
publication date, so a copyright-line year outranks an arbitrary year
mentioned in body text or buried in a references list.

Best-effort, not bibliographic metadata extraction - it can pick the wrong
year on documents with unusual front matter, or find nothing at all
(returns None). Callers treat None as "year unknown" and just skip
year-based filtering for that document - the same graceful-degradation
pattern as heading_detection.py's empty section label.
"""
import re
from datetime import datetime

import pymupdf as fitz  # `fitz` is the old import name; PyMuPDF now ships
                         # it as `pymupdf` and warns on the old name, but
                         # the API is identical, so aliasing it back to
                         # `fitz` avoids renaming every call in this file

YEAR_TOKEN_RE = re.compile(r"^(19|20)\d{2}$")
_YEAR_RE = re.compile(r"\b(19[0-9]{2}|20[0-9]{2})\b")

# Keywords whose nearby years are far more likely to be an actual
# publication date than an arbitrary year mentioned in running text or a
# reference-list entry.
_STRONG_SIGNALS = ("copyright", "©", "published", "publication date")
_MEDIUM_SIGNALS = ("received", "accepted", "doi", "issn", "issue", "volume")

_PAGES_TO_SCAN = 2
_PROXIMITY_WINDOW = 60  # chars on each side of a year to look for a signal word


def _current_max_year():
    return datetime.now().year + 1  # tolerate "in press" / forthcoming dates


def extract_year(pdf_path):
    """Best-effort publication year for one PDF, or None if nothing
    plausible was found (empty document, no 4-digit years on the scanned
    pages, or a PDF that fails to open)."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc[:_PAGES_TO_SCAN]:
            text += page.get_text() + "\n"
        doc.close()
    except Exception:
        return None

    max_year = _current_max_year()
    text_lower = text.lower()
    candidates = []  # (score, position, year) - lower position = earlier on the page

    for m in _YEAR_RE.finditer(text):
        year = int(m.group(0))
        if year < 1900 or year > max_year:
            continue
        window = text_lower[max(0, m.start() - _PROXIMITY_WINDOW): m.start() + _PROXIMITY_WINDOW]
        if any(sig in window for sig in _STRONG_SIGNALS):
            score = 2
        elif any(sig in window for sig in _MEDIUM_SIGNALS):
            score = 1
        else:
            score = 0
        candidates.append((score, m.start(), year))

    if not candidates:
        return None

    best_score = max(c[0] for c in candidates)
    top = sorted((c for c in candidates if c[0] == best_score), key=lambda c: c[1])
    return top[0][2]
