"""
segmenters.rules
=================
Rule-based segementation utilities.

These functions are upure text operations and do not depend on spaCy.
They handle strong boundary signals:
  - Semicolon-separated lists
  - Numbered / lettered lists
  - Semantic completeness validation

All functions are stateless and can be used independetly. 
"""

from __future__ import annotations
import re

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Words that indicate an incomplete fragment when appering at the start
_INCOMPLETE_START: frozenset[str] = frozenset({
    "and", "or",       
    "of", "with",       
    "including", "such", "per", 
})

_INCOMPLETE_END: frozenset[str] = frozenset({
    "and", "or", "with", "of", "in", "for", "to", "by", "on", "at",
    "from", "as", "including", "such", "the", "a", "an",
})

# Numbered list patterns, ordered from most specific to leaset specific
# Each pattern captures the delimiter and the content after it
_NUMBERED_LIST_PATTERNS: list[re.Pattern[str]] = [
    # (1) ... (2) ... style
    re.compile(r"\(\d+\)\s+"),
    # 1. ... 2. ... style
    re.compile(r"(?:(?<=\s)|^)\d+\.\s+"),
    # 1) ... 2) ... style
    re.compile(r"(?:(?<=\s)|^)\d+\)\s+"),
    # a) ... b) ... style (lowercase)
    re.compile(r"(?:(?<=\s)|^)[a-z]\)\s+"),
    # A) ... B) ... style (uppercase)
    re.compile(r"(?:(?<=\s)|^)[A-Z]\)\s+"),
    # a. ... b. ... style
    re.compile(r"(?:(?<=\s)|^)[a-z]\.\s+"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Completeness validation
# ─────────────────────────────────────────────────────────────────────────────

def is_complete_unit(text: str) -> bool:
    """
    Checks if a text fragment is semantically complete

    A fragment is considered incomplete if:
    - It has fewer than 2 tokens
    - It starts with a function word (e.g. "with", "and", "or")
    - It ends with a conjunction or preposition (e.g. "and", "or", "with")
    """
    text = text.strip()
    if not text:
        return False
    
    tokens = text.split()

    # Too short
    if len(tokens) < 2:
        return False
    
    # Only flag as incomplete if the word is lowercase
    # "OR Education" (uppercase) is a complete alternative condition
    # "or equivalent" (lowercase) is an incomplete fragment
    if tokens[0].lower() in _INCOMPLETE_START and tokens[0][0].islower():
        return False

    if tokens[-1].lower().rstrip(".,;:") in _INCOMPLETE_END:
        return False

    return True

def all_complete(parts: list[str]) -> bool:
    """
    Check if all parts pass is_complete_unit.
    """
    return all(is_complete_unit(p) for p in parts)

# ─────────────────────────────────────────────────────────────────────────────
# Semicolon splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_semicolon(text: str) -> list[str]:
    """
    Split text on semicolons if there are two or more semicolons.
    """

    parts = [p.strip() for p in text.split(";")]
    parts = [p for p in parts if p]  # Remove empty parts

    # Only split if there are 2 or more semicolons
    if len(parts) < 3:
        return [text]

    return parts

# ─────────────────────────────────────────────────────────────────────────────
# Numbered / lettered list splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_numbered_list(text: str) -> list[str]:
    """
    Split text that follows a numbered or lettered list pattern.
 
    Recognized formats:
      (1) item one (2) item two (3) item three
      1. item one 2. item two 3. item three
      1) item one 2) item two 3) item three
      a) item one b) item two c) item three
      A) item one B) item two C) item three
      a. item one b. item two c. item three
    """
    for pattern in _NUMBERED_LIST_PATTERNS:
        parts = _try_split_numbered_list(text, pattern)
        if parts is not None:
            return parts
        
    return [text]

def _try_split_numbered_list(text: str, pattern: re.Pattern[str]) -> list[str] | None:
    """
    Attempts to split text using the provided pattern. Returns None if the pattern does not match.
    """
    # Find all delimiter positions
    matches = list(pattern.finditer(text))

    # Not enough matches to split
    if len(matches) < 2:
        return None 
    
    # Extract items between delimiters
    parts = []
    for i, match in enumerate(matches):
        if i == 0:
            before = text[:match.start()].strip()
            if before:
                parts.append(before)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        part = text[start:end].strip()
        if part:
            parts.append(part)

    # Need at least 2 items
    if len(parts) < 2:
        return None

    return parts