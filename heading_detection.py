"""
Layout-based section heading detection.

PDFs don't carry a "this is a heading" flag - PyPDFLoader (used in the
original pipeline) just gives back a flat text blob per page with no
structural information. This module uses PyMuPDF (fitz) instead, which
exposes each line's font size and bold/italic flags, and applies a
heuristic: headings are lines that are meaningfully larger and/or bolder
than the document's own body text, are short, and don't end like a
sentence.

This is heuristic, not perfect - a PDF with unconventional styling (e.g.
headings in the same size as body text, distinguished only by color) will
under-detect. It degrades gracefully in that case: chunks just fall back to
an empty section label, exactly like before this feature existed.
"""
import re
from collections import Counter
from dataclasses import dataclass

import pymupdf as fitz  # `fitz` is the old import name; PyMuPDF now ships
                         # it as `pymupdf` and warns on the old name, but
                         # the API is identical, so aliasing it back to
                         # `fitz` avoids renaming every call in this file

# PyMuPDF span flag bit for bold (see mupdf's text_flags: bit 4 = bold).
_BOLD_FLAG = 1 << 4

# A heading candidate must be at least this many x larger than body text,
# UNLESS it's also bold (bold headings can be the same size as body text).
_SIZE_RATIO_FOR_PLAIN = 1.15

# Headings are short lines, not paragraphs. A line longer than this is
# almost certainly body text even if it happens to be bold/large (e.g. a
# bolded key term at the start of a sentence).
_MAX_HEADING_WORDS = 14

# Lines ending in these are essentially never headings - they read like
# sentences, not titles.
_SENTENCE_ENDINGS = (".", ",", ";", ":")

# Common numbered/lettered heading prefixes: "1.", "1.2", "IV.", "A.", "Chapter 3"
_HEADING_PREFIX_RE = re.compile(
    r"^\s*(chapter\s+\d+|appendix\s+[a-z]|\d+(\.\d+)*\.?|[ivxlc]+\.|[a-z]\))\s+",
    re.IGNORECASE,
)


@dataclass
class Line:
    text: str
    size: float
    bold: bool
    page: int  # 0-indexed


def _extract_lines(pdf_path):
    """Flatten a PDF into an ordered list of Lines with font metadata,
    reading-order preserved (PyMuPDF's block/line order follows the page's
    natural top-to-bottom, left-to-right layout for standard documents)."""
    lines = []
    doc = fitz.open(pdf_path)
    try:
        for page_num, page in enumerate(doc):
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in spans).strip()
                    if not text:
                        continue
                    size = max(s["size"] for s in spans)
                    bold = any(
                        (s["flags"] & _BOLD_FLAG) or "bold" in s.get("font", "").lower()
                        for s in spans
                    )
                    lines.append(Line(text=text, size=round(size, 1), bold=bold, page=page_num))
    finally:
        doc.close()
    return lines


def _body_font_size(lines):
    """The document's dominant font size, weighted by character count so a
    handful of large headings don't skew the baseline. Falls back to 11pt
    (a common default) if the PDF has no extractable text at all."""
    counts = Counter()
    for ln in lines:
        counts[ln.size] += len(ln.text)
    return counts.most_common(1)[0][0] if counts else 11.0


def _looks_like_heading(line, body_size):
    words = line.text.split()
    if not words or len(words) > _MAX_HEADING_WORDS:
        return False
    if line.text.rstrip().endswith(_SENTENCE_ENDINGS) and not line.text.rstrip().endswith(":"):
        # Trailing ':' is fine ("Results:") - it's the sentence-style
        # punctuation (. , ;) that rules a line out.
        return False

    has_numbering = bool(_HEADING_PREFIX_RE.match(line.text))
    big_enough = line.size >= body_size * _SIZE_RATIO_FOR_PLAIN
    bold_and_not_smaller = line.bold and line.size >= body_size

    return big_enough or bold_and_not_smaller or (has_numbering and (line.bold or big_enough))


def detect_sections(pdf_path):
    """
    Returns a list of (page_num, section_title) marking every detected
    heading in reading order, e.g. [(0, "1. Introduction"), (1, "2.1 Data
    Collection"), (1, "3. Results"), ...]. Empty list if nothing looked
    like a heading (caller should treat every chunk as having no section).
    """
    lines = _extract_lines(pdf_path)
    if not lines:
        return []
    body_size = _body_font_size(lines)
    return [(ln.page, ln.text) for ln in lines if _looks_like_heading(ln, body_size)]


def build_page_texts_with_sections(pdf_path):
    """
    Re-walks the PDF and returns, per page, a list of (section_title, text)
    segments - text is everything on that page belonging to that section
    (section_title == "" for any text before the first heading on the
    page, which inherits the last section carried over from a prior page).

    Returns: {page_num: [(section_title, text), ...]}
    """
    lines = _extract_lines(pdf_path)
    if not lines:
        return {}

    body_size = _body_font_size(lines)
    pages = {}
    current_section = ""
    current_page = None
    buffer = []
    has_body_since_heading = False

    def flush():
        if current_page is not None and buffer:
            pages.setdefault(current_page, []).append((current_section, "\n".join(buffer)))

    for ln in lines:
        if current_page is None:
            current_page = ln.page
        if ln.page != current_page:
            flush()
            buffer.clear()
            current_page = ln.page
            # has_body_since_heading intentionally NOT reset here - a
            # section that started on the previous page and is still
            # accumulating body text on this one shouldn't be treated as
            # "heading-only" just because the page turned.

        if _looks_like_heading(ln, body_size):
            if has_body_since_heading or not buffer:
                # Real content under the current section, or nothing
                # buffered yet (first heading in the doc) - close it out.
                flush()
            # else: previous buffer was only an unbroken run of headings
            # (e.g. a title immediately followed by "1. Introduction")
            # with no body text yet - drop it rather than emit a
            # near-empty section, and let the more specific heading win.
            buffer.clear()
            current_section = ln.text.strip()
            buffer.append(ln.text)
            has_body_since_heading = False
        else:
            buffer.append(ln.text)
            has_body_since_heading = True

    flush()
    return pages
