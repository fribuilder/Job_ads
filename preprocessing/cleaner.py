"""
preprocessing/cleaner.py
========================
Lightweight cleaner for raw USAJOBS job announcement text.

Goal
----
This is preprocessing, not perfect information extraction.

The cleaner aims to:
  1. Remove USAJOBS webpage chrome, icons, URLs, contact info, and repeated templates.
  2. Keep mostly job-related sections:
      Summary, Duties, Requirements, Conditions of Employment,
      Qualifications, Education, Additional Information.
  3. Drop application-process sections:
      Benefits, Required Documents, How to Apply, Agency Contact Information,
      Next Steps, Fair and Transparent, How You Will Be Evaluated.
  4. Avoid over-cleaning short but meaningful requirement phrases.

Public API
----------
    from preprocessing.cleaner import clean_description

    cleaned = clean_description(raw_text)
"""

from __future__ import annotations

from email import header
import re
import unicodedata

# ────────────────────────────────────────────────────────────────────────────
# Encoding / Normalization
# ─────────────────────────────────────────────────────────────────────────── 

_MOJIBAKE: dict[str, str] = {
    "\u00e2\u20ac\u201d": "-",
    "\u00e2\u20ac\u201c": "-",
    "\u00e2\u20ac\u02dc": "'",
    "\u00e2\u20ac\u2122": "'",
    "\u00e2\u20ac\u0153": '"',
    "\u00e2\u20ac\u009d": '"',
    "\u00e2\u20ac\u00a6": "...",
    "\u00c2": " ",
    "\u2014": "-",
    "\u2013": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\ufeff": "",
    "\ufffd": "",
    "\x91": "'",
    "\x92": "'",
    "\x93": '"',
    "\x94": '"',
    "\x96": "-",
    "\x97": "-",
}

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

def _repair_encoding(text: str) -> str:
    for bad, good in _MOJIBAKE.items():
        text = text.replace(bad, good)
    return unicodedata.normalize("NFKC", text)

def _basic_normalize(text: str) -> str:
    text = text.replace('""', '"')
    text = _repair_encoding(text)
    text = _CONTROL_RE.sub("", text)
    text= text.replace("\r"," ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _final_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([\])}])", r"\1", text)
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"\.{2}", ".", text)
    return text.strip(" .")

# ─────────────────────────────────────────────────────────────────────────────
# Section definitions
# ─────────────────────────────────────────────────────────────────────────────

KEEP_SECTIONS = {
    "summary",
    "duties",
    "requirements",
    "conditions of employment",
    "qualifications",
    "education",
    "additional information",
}

DROP_SECTIONS = {
    "overview",
    "this job is open to",
    "benefits",
    "how you will be evaluated",
    "required documents",
    "how to apply",
    "agency contact information",
    "next steps",
    "fair and transparent",
    "fair & transparent",
}

SECTION_HEADERS = [
    "This job is open to",
    "Conditions of Employment",
    "How You Will Be Evaluated",
    "Required Documents",
    "How to Apply",
    "Agency contact information",
    "Fair and Transparent",
    "Fair & Transparent",
    "Additional information",
    "Next steps",
    "Summary",
    "Overview",
    "Duties",
    "Requirements",
    "Qualifications",
    "Education",
    "Benefits",
]

ALL_SECTIONS_NAMES = sorted(
    KEEP_SECTIONS | DROP_SECTIONS,
    key = len,
    reverse = True, 
)

def _norm_header(s: str) -> str:
    """Normalize section header names."""
    s = s.lower().strip()
    s = s.replace("&", "and")
    s = re.sub(r"\s+", " ", s)
    return s

def _strip_usajobs_chrome(text: str) -> str:
    """
    Remove top-level USAJOBS page chrome and icon slugs.
    Handles things like:
        USAJOBS - Job Announcement
        family-of-overseas-employees-icon
        peace-corps-and-americorps-icons
        Created with Sketch.
        se-other
    """
    text = re.sub(r"USAJOBS\s*-\s*Job Announcement", "", text, flags=re.I)
    text = re.sub(r"\S+-icons?\b", " ", text, flags=re.I)
    text = re.sub(r"\bCreated with Sketch\.?", " ", text, flags=re.I)
    text = re.sub(r"\bse-other\b", " ", text, flags=re.I)
    return text 

def _mark_sections(text: str) -> str:
    """
    Insert explicit section markers before USAJOBS headers.
    
    Newer USAJOBS samples ofen look like:
        Help Summary ... Help Overview ... Help Duties ...
    
    Old samples often look like:
        Summary ... Overview ... Duties ...

    This function normalizes both forms into marker-separated blocks
    """
    headers_alt = "|".join(
        re.escape(h) for h in sorted(SECTION_HEADERS, key=len, reverse = True)
    )

    def repl(m: re.Match) -> str:
        header = _norm_header(m.group(1))
        return f"||SECTION|| {header} ||"

    #  New USAJOBS format:
    # "Help Summary", "Help Overview", "Help Duties", ...
    text = re.sub(
        rf"\bHelp\s+({headers_alt})\b",
        repl,
        text,
    )

    # Old .dat format:
    # "Summary. ...", "Duties. ...", "Requirements. ..."
    # Require a period after the header, so normal words are not matched.
    text = re.sub(
        rf"\b({headers_alt})\.\s+",
        repl,
        text,
    )

    text = re.sub(
        rf"(?:(?<=[.!?])\s+|^)({headers_alt})\s+(?=[A-Z])",
        repl,
        text,
    )

    return text

def _extract_kept_sections(text: str) -> str:
    """
    Keep body text from useful job-content sections and drop application/webpage sections.
    """
    text = _mark_sections(text)
    parts = re.split(r"\|\|SECTION\|\|", text)

    kept_blocks = []

    for part in parts:
        part = part.strip(" |")
        if not part:
            continue
        matched: str | None = None
        body = ""

        for section in ALL_SECTIONS_NAMES:
            section_norm = _norm_header(section)

            if part.startswith(section_norm):
                matched = section_norm
                body = part[len(section_norm):].strip(" |")
                break
        
        if matched in KEEP_SECTIONS:
            kept_blocks.append(body)
        
    return " ".join(kept_blocks)

# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitting
# ─────────────────────────────────────────────────────────────────────────────

_ABBREVIATIONS = [
    "U.S.",
    "U.S.C.",
    "C.F.R.",
    "F.R.",
    "Ph.D.",
    "M.D.",
    "D.C.",
    "i.e.",
    "e.g.",
    "Mr.",
    "Ms.",
    "Mrs.",
    "Dr.",
    "St.",
    "No.",
]


def _split_sentences(text: str) -> list[str]:
    """
    Simple sentence splitter with abbreviation protection.

    This is intentionally lightweight. It is good enough for preprocessing,
    but it is not a full NLP sentence segmenter.
    """
    placeholders: dict[str, str] = {}

    for i, abbr in enumerate(sorted(_ABBREVIATIONS, key=len, reverse=True)):
        token = f"__ABBR_{i}__"
        placeholders[token] = abbr
        text = text.replace(abbr, token)

    parts = re.split(r"(?<=[.!?])\s+", text)

    restored: list[str] = []
    for part in parts:
        for token, abbr in placeholders.items():
            part = part.replace(token, abbr)

        part = part.strip()
        if part:
            restored.append(part)

    return restored

# ─────────────────────────────────────────────────────────────────────────────
# Fragment-level cleanup
# ─────────────────────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)

_PHONE_RE = re.compile(
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)

_SALARY_RE = re.compile(
    r"(?:salary\s*)?"
    r"(\$[\d,]+(?:\.\d+)?)"
    r"\s*[-–]\s*"
    r"(\$[\d,]+(?:\.\d+)?)"
    r"(?:\s*(per\s+(?:year|hour|annum)))?",
    re.I,
)

_OBVIOUS_NOISE = {
    "help",
    "print",
    "learn more",
    "learn more about this agency",
    "review our benefits",
    "previous result",
    "next result",
    "job closed",
    "hiring complete",
    "reviewing applications",
    "this job announcement has closed",
    "this job originated on",
    "only resumes submitted according to the instructions on the job announcement listed at usajobs will be considered",
}


def _normalize_salary(text: str) -> str:
    """
    Normalize salary expressions if they appear inside kept sections.

    Note: Overview is dropped, so salary from Overview is usually not kept.
    If you need salary, it is better to extract it separately as a structured
    field rather than mix it into description text.
    """

    def repl(m: re.Match) -> str:
        lo = m.group(1)
        hi = m.group(2)
        period = (m.group(3) or "per year").lower()
        return f"The salary range is {lo} - {hi} {period}."

    return _SALARY_RE.sub(repl, text)


def _remove_urls_emails_phones(text: str) -> str:
    """Remove URLs, emails, and phone numbers."""
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _PHONE_RE.sub(" ", text)
    return text


def _is_contact_or_address(sentence: str) -> bool:
    """Detect obvious contact/address fragments."""
    s = sentence.strip()

    if _EMAIL_RE.search(s) or _PHONE_RE.search(s):
        return True

    if re.search(
        r"\b(P\.?O\.?\s*Box|Street|St\.|Avenue|Ave\.|Road|Rd\.|Building|Suite|Room|DO NOT MAIL)\b",
        s,
        re.I,
    ):
        return True

    if re.search(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b", s):
        return True

    return False


def _is_obvious_noise_sentence(sentence: str) -> bool:
    """
    Remove only clear noise.

    Important: Do NOT remove all short sentences. Short phrases such as
    "Valid driver's license required" or "CPR certification required" may be
    meaningful requirements.
    """
    s = sentence.strip(" .")
    sl = s.lower()
    sl = re.sub(r"\s+", " ", sl)

    if not sl:
        return True

    if sl in _OBVIOUS_NOISE:
        return True

    if sl in {"yes", "no", "none", "not required", "not applicable"}:
        return True

    # Pure punctuation / numbers.
    if re.fullmatch(r"[\d\s\-–$.,/():]+", s):
        return True

    return False


def _dedup_sentences(sentences: list[str]) -> list[str]:
    """
    Remove exact duplicate sentences after light normalization.

    This is intentionally conservative. It avoids fuzzy deletion that may remove
    legitimate repeated but different requirements.
    """
    seen: set[str] = set()
    out: list[str] = []

    for s in sentences:
        key = s.lower()
        key = re.sub(r"[^a-z0-9]+", " ", key)
        key = re.sub(r"\s+", " ", key).strip()

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)
        out.append(s)

    return out

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def clean_description(text: str) -> str:
    """
    Clean a raw USAJOBS description/text field.

    Parameters
    ----------
    text:
        Raw USAJOBS description text. Works for both:
          - old .dat-style descriptions
          - newer JSONL-style ``text`` fields

    Returns
    -------
    str
        Mostly job-related plain text.

        Returns an empty string for obvious placeholder/spam records.
    """
    if not text or not text.strip():
        return ""

    text = _basic_normalize(text)

    # Strong placeholder/spam filter.
    # A normal job may contain one "do not apply" in a contact email or note,
    # but repeated occurrences indicate a fake/placeholder announcement.
    if text.lower().count("do not apply") >= 5:
        return ""

    text = _strip_usajobs_chrome(text)

    # Normalize salary before section extraction only for readability if salary
    # happens to appear in a kept section.
    text = _normalize_salary(text)

    # Main step: keep only job-content sections.
    text = _extract_kept_sections(text)

    # Light cleanup inside kept sections.
    text = _remove_urls_emails_phones(text)

    sentences = _split_sentences(text)

    kept: list[str] = []
    for sent in sentences:
        sent = sent.strip(" .")
        if not sent:
            continue

        if _is_obvious_noise_sentence(sent):
            continue

        if _is_contact_or_address(sent):
            continue

        kept.append(sent)

    kept = _dedup_sentences(kept)

    cleaned = ". ".join(kept)
    cleaned = _final_spacing(cleaned)

    if cleaned:
        cleaned += "."

    return cleaned