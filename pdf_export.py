"""
Markdown -> PDF export for whole-document summaries.

Zero new dependencies: markdown-it-py (already installed) converts the
summary markdown to HTML, and PyMuPDF's Story + DocumentWriter paginate
that HTML into a proper PDF - headings, bold text, bullet lists, and
tables all render, so the exported file is a real human-readable summary
document rather than a raw dump of terminal output.

Deliberately simple and defensive:
- A4 pages with margins and one stylesheet.
- A hard page cap so a pathological document can't loop forever.
- Returns the written path, or None on failure - callers print a warning
  and carry on (same graceful-degradation pattern as the rest of the
  pipeline).
"""
import os
import re
from datetime import datetime

import markdown_it
import pymupdf as fitz  # `fitz` is the old import name; PyMuPDF now ships
                         # it as `pymupdf` and warns on the old name, but
                         # the API is identical, so aliasing it back to
                         # `fitz` avoids renaming every call in this file

from config import PDF_EXPORT_DIR

# A4 (595x842pt) with 1cm margins.
_MEDIA = fitz.paper_rect("a4")
_MARGIN = 36

# Safety net: if pagination ever misbehaves (e.g. a PyMuPDF regression
# like the one found while writing this module, where Story.place() must
# be paired with Story.draw() or the loop never advances), bail instead
# of hanging or writing pages forever.
_PAGE_CAP = 100

_CSS = (
    "* { font-family: sans-serif; font-size: 10.5px; line-height: 1.45; }"
    "h1 { font-size: 17px; margin: 0 0 8px 0; }"
    "h2 { font-size: 14px; } h3 { font-size: 12px; }"
    "table { border-collapse: collapse; }"
    "td, th { border: 0.5px solid #888; padding: 2px 5px; font-size: 9.5px; }"
)

_UNSAFE_FILENAME_RE = re.compile(r"[^\w\-. ]+")

# MuPDF's HTML shaper applies ligature substitution to f-f/f-i/f-l
# sequences (U+FB00-U+FB06: f-f ligature, f-i, f-l, ...), which makes
# copy-paste and text extraction from the exported PDF carry ligature
# codepoints instead of plain 'ff'/'fi'. An empty span between the
# letters resets the shaping run, so the same visual result renders
# without ligatures and the PDF's text layer stays plain ASCII.
_LIGATURE_BREAK_RE = re.compile(r"(f)(?=f|i|l)")


def _break_ligatures(html):
    return _LIGATURE_BREAK_RE.sub(r"\1<span></span>", html)


def _markdown_to_html(markdown_text):
    """markdown-it-py renders the markdown this pipeline emits (headings,
    bold, italics, lists, pipe tables). The 'commonmark' preset plus the
    table rule is used instead of 'gfm-like' because gfm-like enables
    linkify, which needs the linkify-it-py plugin that isn't a dependency."""
    return (
        markdown_it.MarkdownIt("commonmark", {"linkify": False})
        .enable("table")
        .render(markdown_text)
    )


def _build_document_markdown(document_name, summary_markdown, stats):
    """Prepend a title and a small metadata block to the summary text so
    the exported PDF stands alone as a document."""
    stats = stats or {}
    lines = [f"# Summary: {document_name}", ""]

    meta = []
    if stats.get("method"):
        meta.append(f"- **Method:** {stats['method']}")
    if stats.get("doc_type"):
        meta.append(f"- **Document type:** {stats['doc_type']}")
    if stats.get("page_count"):
        meta.append(f"- **Pages:** {stats['page_count']}")
    if stats.get("chunk_count"):
        meta.append(f"- **Chunks:** {stats['chunk_count']}")
    if stats.get("table_count"):
        meta.append(f"- **Tables extracted:** {stats['table_count']}")
    if meta:
        lines.append("\n".join(meta))
        lines.append("")

    lines.append(f"*Exported {datetime.now().strftime('%Y-%m-%d %H:%M')} by the research assistant.*")
    lines.append("")
    lines.append(summary_markdown)
    return "\n\n".join(lines)


def export_summary_pdf(document_name, summary_markdown, stats=None, export_dir=None):
    """
    Write `summary_markdown` (plus a title/metadata block) to
    '<export_dir>/<document_name>_summary_<timestamp>.pdf'.

    Returns the written path, or None on failure (bad HTML, unwritable
    directory, pagination trouble, ...) - the caller keeps the terminal
    output either way.
    """
    export_dir = export_dir or PDF_EXPORT_DIR
    writer = None
    path = None
    try:
        md_doc = _build_document_markdown(document_name, summary_markdown, stats)
        html = _break_ligatures(_markdown_to_html(md_doc))

        os.makedirs(export_dir, exist_ok=True)
        safe_name = _UNSAFE_FILENAME_RE.sub("_", document_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(export_dir, f"{safe_name}_summary_{timestamp}.pdf")

        writer = fitz.DocumentWriter(path)
        story = fitz.Story(html=f"<html><body>{html}</body></html>", user_css=_CSS)
        where = _MEDIA + (_MARGIN, _MARGIN, -_MARGIN, -_MARGIN)

        more, pages = 1, 0
        while more and pages < _PAGE_CAP:
            device = writer.begin_page(_MEDIA)
            more, _filled = story.place(where)
            story.draw(device)
            writer.end_page()
            pages += 1

        writer.close()
        writer = None
        return path
    except Exception:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if path is not None:
            try:
                os.remove(path)
            except OSError:
                pass
        return None
