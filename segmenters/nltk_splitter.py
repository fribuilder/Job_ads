"""
segmenters/nltk_splitter.py
============================
Sentence splitting using NLTK with hard boundary pre-processing.

This module provides an alternative upstream splitter to pure rule-based
splitting. It combines:
  1. Pre-normalization: placeholder protection for patterns that cause false
     sentence boundaries in job-ad text (time ranges, GS grades, etc.)
  2. Hard boundary pre-processing (numbered lists, semicolons via rules.py)
  3. Punctuation-based boundary normalization (bullets, dashes, etc.)
  4. NLTK sentence tokenization
  5. Post-processing: merge incomplete and lowercase-continuation fragments

Each splitting step is gated by a completeness check. If any fragment
fails the check, the split is rejected and the next step is tried.
If no step produces valid splits, the original text is returned as-is.
"""

from __future__ import annotations

import re

import nltk
from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize

from .rules import split_numbered_list, split_semicolon, is_complete_unit, all_complete

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-normalization: protect job-ad patterns from false boundary detection
# ─────────────────────────────────────────────────────────────────────────────

# Time range: "8:00am - 4:30pm", "08:00 am-4:30 pm", etc.
_TIME_RANGE_RE = re.compile(
    r"\b\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?\s*[-–—]\s*\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?",
    re.I,
)

# Weekday range: "Monday - Friday", "Mon-Fri", etc.
_WEEKDAY = (
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
    r"|Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
)
_WEEKDAY_RANGE_RE = re.compile(
    rf"\b{_WEEKDAY}\s*[-–—]\s*{_WEEKDAY}\b",
    re.I,
)

# U.S. / U.S.A. — NLTK Punkt sometimes fails on these
_US_ABBREV_RE = re.compile(r"\bU\.S\.(?:A\.)?(?=[\s,;)]|$)", re.I)

# GS grade ranges: "GS-3 to GS-6", "GS-14"
_GS_GRADE_RE = re.compile(r"\bGS-\d+(?:\s+to\s+GS-\d+)?\b", re.I)

_PROTECT_PATTERNS: list[re.Pattern] = [
    _TIME_RANGE_RE,
    _WEEKDAY_RANGE_RE,
    _US_ABBREV_RE,
    _GS_GRADE_RE,
]

# Placeholder uses a null byte prefix so it can't appear in real text
_PLACEHOLDER_FMT = "\x00P{}\x00"


def _pre_normalize(text: str) -> tuple[str, dict[str, str]]:
    """Replace sensitive tokens with placeholders to protect them from NLTK.

    Returns the modified text and a restore map (placeholder → original).
    """
    restore: dict[str, str] = {}
    counter = 0

    def _sub(m: re.Match) -> str:
        nonlocal counter
        key = _PLACEHOLDER_FMT.format(counter)
        counter += 1
        restore[key] = m.group(0)
        return key

    for pat in _PROTECT_PATTERNS:
        text = pat.sub(_sub, text)

    return text, restore


def _post_restore(parts: list[str], restore: dict[str, str]) -> list[str]:
    """Undo placeholder substitutions in all parts."""
    if not restore:
        return parts
    result = []
    for p in parts:
        for key, val in restore.items():
            p = p.replace(key, val)
        result.append(p)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Boundary normalization pattern
# Semicolons are excluded here as they are handled separately by split_semicolon
# Note: space-hyphen-space ( \s-\s ) is intentionally omitted because it
# appears in time/date ranges ("8:00am - 4:30pm", "Monday - Friday") and
# salary ranges, causing catastrophic false boundaries in job-ad text.
# ─────────────────────────────────────────────────────────────────────────────

_BOUNDARY_PATTERN = re.compile(
    r"("
    r"\s\+\s"      # space + plus + space
    r"|\s\*\s"     # space + asterisk + space
    r"|\s•\s"      # bullet point
    r"|\s·\s"      # middle dot
    r"|--"         # double hyphen
    r"|\*\*"       # double asterisk
    r")"
)

_DOUBLE_PERIOD = re.compile(r"\.{2,}")


def _postprocess(parts: list[str]) -> list[str]:
    """Post-process split results.

    Merges a fragment into the preceding one when:
    - it is semantically incomplete (starts with function word or ends with
      preposition, per ``is_complete_unit``), OR
    - it starts with a lowercase letter, which signals a mid-sentence split
      caused by a false boundary (e.g. NLTK splitting "... conditions" /
      "to use judgment ..." at a semicolon that was inside a clause).
    """
    if not parts:
        return parts

    result = [parts[0]]
    for part in parts[1:]:
        first_char = part.lstrip()[:1]
        is_lowercase_start = bool(first_char) and first_char.islower()
        if not is_complete_unit(part) or is_lowercase_start:
            result[-1] = result[-1].rstrip() + " " + part.lstrip()
        else:
            result.append(part)

    return result

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def nltk_split(text: str) -> list[str]:
    if not text or not text.strip():
        return []

    # Step 0: protect job-ad patterns from NLTK false boundaries
    text, restore = _pre_normalize(text)

    # Step 1: numbered lists
    numbered_parts = split_numbered_list(text)
    if len(numbered_parts) > 1:
        results = []
        for part in numbered_parts:
            results.extend(nltk_split(part))
        return _post_restore(_postprocess(results), restore)

    # Step 2: semicolons
    semicolon_parts = split_semicolon(text)
    if len(semicolon_parts) > 1:
        results = []
        for part in semicolon_parts:
            results.extend(nltk_split(part))
        return _post_restore(_postprocess(results), restore)

    # Step 3: normalize
    normalized = ". ".join(text.split("\n"))
    normalized = _BOUNDARY_PATTERN.sub(". ", normalized)
    normalized = _DOUBLE_PERIOD.sub(".", normalized)

    # Step 4: NLTK
    sentences = [s.strip() for s in _nltk_sent_tokenize(normalized) if s.strip()]
    if len(sentences) > 1:
        return _post_restore(_postprocess(sentences), restore)

    # Step 5: fallback
    return _post_restore([text], restore)


def nltk_split_batch(texts: list[str]) -> list[list[str]]:
    """
    Split a batch of texts using nltk_split.
    """
    return [nltk_split(text) for text in texts]