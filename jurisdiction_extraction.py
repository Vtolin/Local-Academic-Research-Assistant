"""
Jurisdiction extraction (best-effort metadata for filtering).

Unlike publication year (a clean 4-digit pattern) or section headings
(a font-size/boldness signal), jurisdiction is genuinely an open-ended
language-understanding problem - "which court decided this" isn't a fixed
format the way a date is. This module takes the same approach as the rest
of this pipeline's metadata extraction anyway: a deterministic, ingestion-
time keyword/pattern pass over the first page, rather than an LLM call.

That's a real tradeoff, not a free lunch - a keyword table will always be
incomplete, and this one only covers commonly-cited English-speaking and
international courts. It will miss courts not in the table, and a
document with none of these signals just gets jurisdiction=None (the same
graceful-degradation pattern as year/section/pinpoint). But it keeps
ingestion fast, deterministic, and independent of Ollama being reachable
at index time - consistent with how heading/year detection were designed
- and it's straightforward to extend: add a (pattern, label) pair to
_JURISDICTION_PATTERNS below for any court missing from the list.

Worth knowing: plain keyword search (the BM25 half of hybrid retrieval)
already surfaces court-name mentions to some extent, since if a chunk's
text literally contains "Court of Appeal", a query mentioning "Court of
Appeal" will tend to rank it higher. What this module adds on top is the
ability to definitively FILTER to only one jurisdiction's documents
(rather than just rank them higher), and to show jurisdiction in
citations/exports.
"""
import re

_PAGES_TO_SCAN = 1  # jurisdiction/court name is almost always on the cover/first page

# (pattern, canonical label). Order matters somewhat - more specific
# patterns are listed before broader ones they could otherwise be
# shadowed by (e.g. "England and Wales" before a generic "United Kingdom").
_JURISDICTION_PATTERNS = [
    # Indonesia (civil law system - courts are organized by subject-matter
    # and instance/appellate tier, not the common-law tiers most of this
    # table covers). Order is critical here in exactly the way it bit the
    # US-state fallback below during testing: a more specific multi-word
    # name MUST come before a shorter name it contains as a substring, or
    # the generic pattern matches first and shadows the specific one -
    # e.g. "pengadilan tinggi" would incorrectly match text that actually
    # says "pengadilan tinggi agama" if the generic entry came first.
    (r"\bpengadilan tinggi tata usaha negara\b", "Indonesia (PTTUN / High Administrative Court)"),
    (r"\bpengadilan tata usaha negara\b", "Indonesia (PTUN / Administrative Court)"),
    (r"\bpengadilan tinggi agama\b", "Indonesia (High Religious Court)"),
    (r"\bpengadilan agama\b", "Indonesia (Religious Court)"),
    (r"\bpengadilan hubungan industrial\b", "Indonesia (Industrial Relations Court)"),
    (r"\bpengadilan niaga\b", "Indonesia (Commercial Court)"),
    (r"\bpengadilan pajak\b", "Indonesia (Tax Court)"),
    (r"\bpengadilan militer\b", "Indonesia (Military Court)"),

    # Specialized courts and dispute-resolution fora
    (r"\bmahkamah syar'?iyah\b", "Indonesia (Mahkamah Syar'iyah / Sharia Court, Aceh)"),
    (r"\bpengadilan syar'?iyah\b", "Indonesia (Pengadilan Syar'iyah / Sharia Court, Aceh)"),
    (r"\bpengadilan tindak pidana korupsi\b|\bpengadilan tipikor\b", "Indonesia (Corruption Court / Tipikor)"),
    (r"\bpengadilan hak asasi manusia\b", "Indonesia (Human Rights Court)"),
    (r"\bpengadilan perikanan\b", "Indonesia (Fisheries Court)"),
    (r"\bpengadilan anak\b", "Indonesia (Juvenile Court)"),
    (r"\bbadan penyelesaian sengketa konsumen\b", "Indonesia (BPSK / Consumer Dispute Settlement Body)"),
    (r"\bkomisi pengawas persaingan usaha\b", "Indonesia (KPPU / Competition Commission)"),
    (r"\bbadan arbitrase nasional indonesia\b", "Indonesia (BANI / National Arbitration Board)"),

    (r"\bmahkamah agung\b", "Indonesia (Mahkamah Agung / Supreme Court)"),
    (r"\bmahkamah konstitusi\b", "Indonesia (Mahkamah Konstitusi / Constitutional Court)"),
    (r"\bpengadilan tinggi\b", "Indonesia (High Court)"),   # generic - after all "pengadilan tinggi X" above
    (r"\bpengadilan negeri\b", "Indonesia (District Court)"),

    # United States
    (r"\bsupreme court of the united states\b", "US Supreme Court"),
    (r"\bu\.?\s?s\.?\s*supreme court\b", "US Supreme Court"),
    (r"\bunited states court of appeals for the (\w+) circuit\b", "US Court of Appeals ({0} Cir.)"),
    (r"\bunited states district court\b", "US District Court"),

    # United Kingdom / England & Wales
    (r"\bsupreme court of the united kingdom\b", "UK Supreme Court"),
    (r"\bcourt of appeal.{0,40}england and wales\b", "England & Wales (Court of Appeal)"),
    (r"\bhigh court of justice\b", "England & Wales (High Court)"),
    (r"\bengland and wales\b", "England & Wales"),
    (r"\bhouse of lords\b", "UK (House of Lords, pre-2009)"),

    # Australia
    (r"\bhigh court of australia\b", "Australia (High Court)"),
    (r"\bfederal court of australia\b", "Australia (Federal Court)"),
    (r"\bsupreme court of (new south wales|victoria|queensland|western australia|south australia|tasmania)\b",
     "Australia ({0} Supreme Court)"),

    # Canada
    (r"\bsupreme court of canada\b", "Canada (Supreme Court)"),
    (r"\bfederal court of appeal\b.{0,40}canada|\bcanada\b.{0,40}federal court of appeal", "Canada (Federal Court of Appeal)"),

    # New Zealand
    (r"\bsupreme court of new zealand\b", "New Zealand (Supreme Court)"),
    (r"\bhigh court of new zealand\b", "New Zealand (High Court)"),

    # Ireland
    (r"\bsupreme court of ireland\b", "Ireland (Supreme Court)"),

    # International / supranational
    (r"\binternational court of justice\b", "ICJ"),
    (r"\binternational criminal court\b", "ICC"),
    (r"\beuropean court of human rights\b", "ECtHR"),
    (r"\bcourt of justice of the european union\b", "EU (CJEU)"),
    (r"\bworld trade organization\b.{0,40}(panel|appellate body)", "WTO"),

    # Other common-law jurisdictions
    (r"\bhigh court of singapore\b", "Singapore (High Court)"),
    (r"\bcourt of final appeal\b.{0,40}hong kong|\bhong kong\b.{0,40}court of final appeal", "Hong Kong (Court of Final Appeal)"),
    (r"\bconstitutional court of south africa\b", "South Africa (Constitutional Court)"),
    (r"\bsupreme court of india\b", "India (Supreme Court)"),

    # Generic "Supreme Court of <X>" catch-all for US states (Ohio, Texas,
    # etc.) - deliberately LAST, so it never shadows a more specific
    # country-level match listed above (this exact bug happened during
    # testing: "Supreme Court of Canada" was matching this pattern before
    # reaching the dedicated Canada entry, when this line was placed near
    # the top of the list).
    (r"\bsupreme court of (\w+)\b", "{0} Supreme Court (US state)"),
]

_COMPILED = [(re.compile(pat, re.IGNORECASE), label) for pat, label in _JURISDICTION_PATTERNS]


def extract_jurisdiction(pdf_path):
    """Best-effort jurisdiction label for one PDF, or None if nothing in
    _JURISDICTION_PATTERNS matched the first page. Import is local to
    avoid a hard PyMuPDF dependency for callers that only need the
    pattern table (e.g. tests)."""
    import pymupdf as fitz

    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc[:_PAGES_TO_SCAN]:
            text += page.get_text() + "\n"
        doc.close()
    except Exception:
        return None

    return match_jurisdiction(text)


def match_jurisdiction(text):
    """Pattern-matching logic split out from extract_jurisdiction so it
    can be tested directly against plain text, without needing a real PDF
    file on disk for every test case."""
    for pattern, label in _COMPILED:
        m = pattern.search(text)
        if m and m.groups():
            try:
                return label.format(*[g.title() if g else g for g in m.groups()])
            except (IndexError, KeyError):
                return label
        if m:
            return label
    return None
