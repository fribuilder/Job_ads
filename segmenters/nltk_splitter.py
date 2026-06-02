"""
segmenters/nltk_splitter.py
============================
Sentence splitting using NLTK with hard boundary pre-processing.

This module provides an alternative upstream splitter to pure rule-based
splitting. It combines:
  1. Hard boundary pre-processing (numbered lists, semicolons via rules.py)
  2. Punctuation-based boundary normalization (bullets, dashes, etc.)
  3. NLTK sentence tokenization

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
# Boundary normalization pattern
# Semicolons are excluded here as they are handled separately by split_semicolon
# ─────────────────────────────────────────────────────────────────────────────

_BOUNDARY_PATTERN = re.compile(
    r"("
    r"\s\+\s"      # space + plus + space
    r"|\s\*\s"     # space + asterisk + space
    r"|\s\-\s"     # space + hyphen + space
    r"|\s•\s"      # bullet point
    r"|\s·\s"      # middle dot
    r"|--"         # double hyphen
    r"|\*\*"       # double asterisk
    r")"
)

_DOUBLE_PERIOD = re.compile(r"\.{2,}")

def _postprocess(parts: list[str]) -> list[str]:
    """
    Post-process split results.
    
    For each part:
    - If complete: keep as-is
    - If incomplete: merge with previous part
    
    If only one part remains after merging, return original text.
    """
    if not parts:
        return parts
    
    result = [parts[0]]
    for part in parts[1:]:
        if not is_complete_unit(part):
            # 合并到前一个片段
            result[-1] = result[-1] + " " + part
        else:
            result.append(part)
    
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def nltk_split(text: str) -> list[str]:
    if not text or not text.strip():
        return []

    # Step 1: numbered lists
    numbered_parts = split_numbered_list(text)
    if len(numbered_parts) > 1:
        results = []
        for part in numbered_parts:
            results.extend(nltk_split(part))
        return _postprocess(results)

    # Step 2: semicolons
    semicolon_parts = split_semicolon(text)
    if len(semicolon_parts) > 1:
        results = []
        for part in semicolon_parts:
            results.extend(nltk_split(part))
        return _postprocess(results)

    # Step 3: normalize
    normalized = ". ".join(text.split("\n"))
    normalized = _BOUNDARY_PATTERN.sub(". ", normalized)
    normalized = _DOUBLE_PERIOD.sub(".", normalized)

    # Step 4: NLTK
    sentences = [s.strip() for s in _nltk_sent_tokenize(normalized) if s.strip()]
    if len(sentences) > 1:
        return _postprocess(sentences)

    # Step 5: fallback
    return [text]


def nltk_split_batch(texts: list[str]) -> list[list[str]]:
    """
    Split a batch of texts using nltk_split.
    """
    return [nltk_split(text) for text in texts]