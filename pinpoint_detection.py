"""
Pinpoint citation detection: finding the specific-provision markers legal
writing uses to cite a precise point in a source, so citations can point
past the page number to the actual locator - PDF page numbers are a
rendering artifact, not a real citation target in either tradition below.

Two different conventions are detected, kept deliberately separate rather
than merged into one numbering system (an article number and a judgment
paragraph number are different things - collapsing them could produce a
nonsensical range like "paragraphs 5-42" out of an unrelated Pasal 5 and
paragraph [42]):

  - Common-law judgment paragraphs: bracketed ([42], the dominant UK/
    Commonwealth/international convention) or pilcrow/spelled-out
    (¶ 42, para. 42, paragraph 42).
  - Civil-law statute articles (Indonesia and other civil-law systems):
    "Pasal N" (Article N), optionally with "ayat (M)" (paragraph/clause M),
    "huruf X" (letter X - e.g. "Pasal 5 ayat (2) huruf a"), and article
    chains joined with "jo."/"juncto" ("Pasal 5 jo. Pasal 7").

Both are deliberately conservative. A bare leading number ("42. The
claimant submits...") is NOT treated as a judgment-paragraph marker, even
though some judgments use it - on its own it's indistinguishable from an
ordinary numbered list, a statute subsection, or heading_detection.py's
own numbered-heading convention without more context than a single chunk
provides. Multiple distinct Pasal numbers found in one chunk are listed
rather than paired with a specific ayat, since associating the right ayat
to the right Pasal reliably would need positional analysis this
chunk-level heuristic doesn't attempt (a huruf is only appended when the
chunk contains exactly one Pasal, for the same reason).

Every chunk gets tagged with a 'pinpoint' metadata string if either
convention was found - Pasal/ayat takes priority if both somehow appear
in the same chunk, since the other convention is far less likely to occur
in the same document. No markers found -> no 'pinpoint' key at all,
falling back to the page number, same graceful-degradation pattern as
section/year/jurisdiction detection elsewhere in this pipeline.
"""
import re

_BRACKET_RE = re.compile(r"\[(\d{1,4})\]")
_PILCROW_RE = re.compile(r"(?:¶|para(?:graph)?\.?)\s*(\d{1,4})", re.IGNORECASE)
_PASAL_RE = re.compile(r"\bpasal\s+(\d{1,4}[a-zA-Z]?)\b", re.IGNORECASE)
_AYAT_RE = re.compile(r"\bayat\s*\(?(\d{1,3})\)?", re.IGNORECASE)

# "huruf a" / "huruf (a)" - a bare letter token (or parenthesized letter)
# after "huruf". The (?!\w) guard keeps ordinary phrases like "huruf
# kapital" (capital letter) from matching a stray "k".
_HURUF_RE = re.compile(
    r"\bhuruf\s*(?:\(([a-zA-Z])\)|([a-zA-Z])(?!\w))", re.IGNORECASE
)

# Article chains: "Pasal 5 jo. Pasal 7" / "Pasal 5 juncto Pasal 7" - the
# standard Indonesian way of citing a provision read together with
# another. Only the first chain in a chunk is used (see module docstring
# on positional analysis).
_PASAL_JO_RE = re.compile(
    r"\bpasal\s+(\d{1,4}[a-zA-Z]?)\s*(?:jo\.|juncto)\s*pasal\s+(\d{1,4}[a-zA-Z]?)",
    re.IGNORECASE,
)


def find_paragraph_markers(text):
    """All common-law judgment-paragraph numbers found in `text`, in the
    order they appear."""
    numbers = []
    for pattern in (_BRACKET_RE, _PILCROW_RE):
        numbers.extend(int(m.group(1)) for m in pattern.finditer(text))
    return numbers


def find_pasal_markers(text):
    """Civil-law statute markers: (pasal_numbers, ayat_numbers), each a
    list of strings in the order they appear. Pasal numbers keep any
    trailing letter (Indonesian statutes amend with suffixes like "5A")."""
    pasal = _PASAL_RE.findall(text)
    ayat = _AYAT_RE.findall(text)
    return pasal, ayat


def find_huruf_markers(text):
    """Sub-article letter markers ("huruf a"), lowercase, in the order
    they appear. See module docstring for when they're actually used."""
    return [m.group(1) or m.group(2) for m in _HURUF_RE.finditer(text)]


def find_pasal_juncto(text):
    """A "Pasal X jo. Pasal Y" chain, or None if the chunk contains none.
    Takes priority over the plain Pasal list in tag_chunks_with_pinpoints,
    since "Pasal 5, 7" would lose the jo. relationship."""
    m = _PASAL_JO_RE.search(text)
    if not m:
        return None
    return f"Pasal {m.group(1)} jo. Pasal {m.group(2)}"


def format_pinpoint(numbers):
    """Turn found judgment-paragraph numbers into a citation-ready
    pinpoint string. One number -> '¶ 42'. Several -> the min-max range
    spanning them, '¶¶ 40-45', even if every number in between didn't
    literally appear (a chunk covering paragraphs 40-45 might only show
    markers at 40 and 45 if the ones between run on without restating a
    bracket each time)."""
    if not numbers:
        return None
    lo, hi = min(numbers), max(numbers)
    return f"¶ {lo}" if lo == hi else f"¶¶ {lo}-{hi}"


def format_pasal_pinpoint(pasal_numbers, ayat_numbers, huruf_letters=None):
    """Turn found Pasal/ayat markers into a citation-ready string. One
    Pasal -> 'Pasal 5' (+ ' ayat (2)' / ' ayat (2, 3)' if ayat markers
    were also found, + ' huruf a' if exactly one Pasal and one or more
    huruf letters were found). Several distinct Pasal numbers -> just
    listed ('Pasal 5, 8'), without attempting to pair specific ayat or
    huruf markers to specific Pasal numbers - see module docstring for
    why."""
    if not pasal_numbers:
        return None
    unique_pasal = list(dict.fromkeys(pasal_numbers))  # dedupe, keep first-seen order
    if len(unique_pasal) > 1:
        return f"Pasal {', '.join(unique_pasal)}"

    base = f"Pasal {unique_pasal[0]}"
    unique_ayat = list(dict.fromkeys(ayat_numbers))
    if unique_ayat:
        base += f" ayat ({', '.join(unique_ayat)})"
    if huruf_letters:
        unique_huruf = list(dict.fromkeys(huruf_letters))
        base += f" huruf {', '.join(unique_huruf)}"
    return base


def tag_chunks_with_pinpoints(chunks):
    """Mutates `chunks` in place, adding metadata['pinpoint'] to any chunk
    whose text contains a detected marker (a jo./juncto Pasal chain first,
    then Pasal/ayat/huruf, then bracket/pilcrow). Returns the same list
    for convenience/chaining."""
    for chunk in chunks:
        pinpoint = find_pasal_juncto(chunk.page_content)
        if not pinpoint:
            pasal, ayat = find_pasal_markers(chunk.page_content)
            pinpoint = format_pasal_pinpoint(pasal, ayat, find_huruf_markers(chunk.page_content))
        if not pinpoint:
            pinpoint = format_pinpoint(find_paragraph_markers(chunk.page_content))
        if pinpoint:
            chunk.metadata["pinpoint"] = pinpoint
    return chunks
