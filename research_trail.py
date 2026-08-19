"""
Research trail + citation export.

Two related but distinct things:

  - Research trail: an append-only markdown log of every question asked
    and answer given (with its citations), written automatically as you
    work. This is a session audit trail, not a formatted bibliography -
    it's meant to be read back later to reconstruct what you found and
    where, not handed in as-is.

  - Citation export: a deduplicated list of just the sources actually
    cited across the session (or read back from an existing trail file),
    written as a flat list suitable for turning into a table of
    authorities/bibliography. Citation FORMAT (Bluebook/OSCOLA/AGLC/...)
    is deliberately not hardcoded here - see format_citation's `style`
    parameter - because getting that wrong is worse than not offering it;
    ask before assuming a convention.

  Security note: question and answer text (including quotes from the
  source documents) is written VERBATIM by design - an audit trail that
  sanitizes content would no longer be verbatim. That means the file is
  untrusted content: if you ever render research_trail.md as HTML or
  feed it to another tool, embedded document text could carry malicious
  links or markup. Treat it as data, not as a document you authored.
"""
import os
from datetime import datetime


def format_citation(doc_meta, style="plain"):
    """
    One citation line for a single chunk's metadata. `style` controls
    formatting:
      - "plain" (default): "filename.pdf, Page 3, ¶ 42 (Section)" - style-
        agnostic, always available, safe default until a real citation
        style is configured.
      - Add more styles here once you've picked one (Bluebook/OSCOLA/
        AGLC/...) - each is just a different arrangement of the same
        underlying fields (source, page, pinpoint, section, year,
        jurisdiction), no new detection work needed.
    """
    src = os.path.basename(doc_meta.get("source", "Unknown file"))
    page = doc_meta.get("page", 0) + 1
    pinpoint = (doc_meta.get("pinpoint") or "").strip()
    section = (doc_meta.get("section") or "").strip()
    year = doc_meta.get("year")
    jurisdiction = (doc_meta.get("jurisdiction") or "").strip()

    if style == "plain":
        parts = [src, f"Page {page}"]
        if pinpoint:
            parts.append(pinpoint)
        line = ", ".join(parts)
        extra = []
        if section:
            extra.append(section)
        if jurisdiction:
            extra.append(jurisdiction)
        if year:
            extra.append(str(year))
        if extra:
            line += f" ({'; '.join(extra)})"
        return line

    # Unrecognized style falls back to plain rather than raising - a typo
    # in a style name shouldn't break logging/export.
    return format_citation(doc_meta, style="plain")


class ResearchTrail:
    """
    Appends one markdown entry per answered question to `path`. Nothing is
    held in memory beyond the current session's citation set (for
    export_citations) - each entry is written to disk immediately, so a
    crash mid-session doesn't lose earlier entries.
    """

    def __init__(self, path):
        self.path = path
        self._seen_citations = {}  # citation line -> doc_meta, for export_citations

    def log(self, mode, question, answer, retrieved_docs):
        """Append one entry. `mode` is a short label ('ask', 'broad',
        'summarize', 'compare') so the trail reads clearly when skimmed."""
        citations = []
        for doc in retrieved_docs:
            meta = doc.metadata
            line = format_citation(meta)
            if line not in self._seen_citations:
                self._seen_citations[line] = meta
            citations.append(line)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry_lines = [
            f"## [{mode}] {timestamp}",
            "",
            f"**Q:** {question}",
            "",
            f"**A:** {answer}",
            "",
        ]
        if citations:
            entry_lines.append("**Sources:**")
            for c in dict.fromkeys(citations):  # dedupe, keep first-seen order
                entry_lines.append(f"- {c}")
            entry_lines.append("")
        entry_lines.append("---")
        entry_lines.append("")

        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write("\n".join(entry_lines) + "\n")
        except OSError as e:
            print(f"[warning] could not write to research trail ({e}); continuing without logging this entry.")

    def export_citations(self, export_path, style="plain"):
        """Write every distinct citation seen so far this session, one per
        line, to `export_path`. Returns the count written, or None if
        nothing has been cited yet."""
        if not self._seen_citations:
            return None
        lines = [format_citation(meta, style=style) for meta in self._seen_citations.values()]
        try:
            with open(export_path, "w", encoding="utf-8") as f:
                f.write("\n".join(sorted(set(lines))) + "\n")
        except OSError as e:
            print(f"[warning] could not write citation export ({e}).")
            return None
        return len(set(lines))
