"""
segmenters/spacy_utils.py
==========================
spaCy-based dependency parsing utilities for coordination splitting.

This module provides functions to detect and split coordination structures
in sentences using spaCy dependency relations.

All functions take spaCy Doc or Span objects as input, assuming the caller
manages model loading.
"""

from __future__ import annotations
import re
import spacy

# ──────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────

_PARENTHETICAL = re.compile(
    r"\b(for example|i\.e\.?|e\.g\.?|for instance|namely)$", re.I
)

def _conj_root_dep(token) -> str:
    """Follow conj chain to find ultimate head's dep."""
    current = token
    while current.dep_ == "conj":
        current = current.head
    return current.dep_


def _is_inside_parentheses(sent, comma_token) -> bool:
    """Check if a comma is inside parentheses."""
    depth = 0
    for token in sent:
        if token.text in ("(", "[", "{"):
            depth += 1
        elif token.text in (")", "]", "}"):
            depth -= 1
        if token.i == comma_token.i and depth > 0:
            return True
    return False

_SHARED_CONTEXT_DEPS = frozenset({
    "pobj",   # prepositional object - shares prep context
    "dobj",   # direct object - shares verb context
    "attr",   # attribute - shares copula
    "acomp",  # adjective complement
    "xcomp",  # open clausal complement
    "appos",  # appositive
    "relcl",  # relative clause - shares head noun
    "acl",    # adjectival clause - shares head noun
    "advcl",  # adverbial clause
    "ccomp",  # clausal complement
    "amod",   # adjectival modifier
    "nmod",   # nominal modifier
})

# ──────────────────────────────────────────────────────────
# Coordination detection
# ─────────────────────────────────────────────────────────

def has_splittable_coordination(sent) -> bool:
    conj_tokens = [t for t in sent if t.dep_ == "conj"]
    if not conj_tokens:
        return False

    for token in conj_tokens:
        if token.pos_ in ("ADJ", "ADV"):
            continue
        ultimate_dep = _conj_root_dep(token)
        if ultimate_dep in _SHARED_CONTEXT_DEPS:
            continue
        return True

    return False


def _find_valid_commas(sent) -> list[int]:
    """Return doc-relative token indices of valid coordination-split commas.

    A comma is valid when the token immediately following it (skipping CCONJ/AUX)
    is a ``conj`` dependent with its own syntactic structure and not inside a
    shared-context dependency chain.
    """
    comma_tokens = [t for t in sent if t.text == ","]
    if not comma_tokens:
        return []

    valid_commas: list[int] = []
    step2_prev = 0  # span-relative token index of the start of the current segment

    for comma in comma_tokens:
        if _is_inside_parentheses(sent, comma):
            step2_prev = comma.i + 1
            continue

        if comma.i > 0 and sent[comma.i - 1].pos_ == "ADP":
            step2_prev = comma.i + 1
            continue

        if _PARENTHETICAL.search(sent[step2_prev:comma.i].text):
            step2_prev = comma.i + 1
            continue

        right_i = comma.i + 1
        while right_i < len(sent) and sent[right_i].pos_ in ("CCONJ", "AUX"):
            right_i += 1
        if right_i >= len(sent):
            continue
        right = sent[right_i]

        if right.dep_ != "conj":
            step2_prev = comma.i + 1
            continue

        if right.pos_ in ("ADJ", "ADV", "NUM"):
            step2_prev = comma.i + 1
            continue

        ultimate_dep = _conj_root_dep(right)
        if ultimate_dep in _SHARED_CONTEXT_DEPS:
            step2_prev = comma.i + 1
            continue

        has_own_structure = any(
            t.dep_ in (
                "dobj", "nsubj", "prep", "attr", "ccomp",
                "relcl", "advmod", "acl", "xcomp", "oprd"
            )
            for t in right.children
        )
        if not has_own_structure:
            step2_prev = comma.i + 1
            continue

        valid_commas.append(comma.i)
        step2_prev = comma.i + 1

    return valid_commas


# ──────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────

def split_by_coordination(text: str, nlp) -> list[str]:
    """
    Split a sentence on comma/conjunction boundaries using spaCy
    dependency relations to validate each split point.
    """
    if ":" in text:
        return [text]

    doc = nlp(text)
    sent = list(doc.sents)[0]

    if not has_splittable_coordination(sent):
        return [text]

    valid_commas = _find_valid_commas(sent)
    if not valid_commas:
        return [text]

    parts = []
    prev = 0
    for comma_i in valid_commas:
        part = sent[prev:comma_i].text.strip()
        part = re.sub(r"^(and|or|but)\s+", "", part, flags=re.I)
        if part:
            parts.append(part)
        prev = comma_i + 1

    last = sent[prev:].text.strip()
    last = re.sub(r"^(and|or|but)\s+", "", last, flags=re.I)
    if last:
        parts.append(last)

    if len(parts) < 2:
        return [text]

    return parts


def split_by_coordination_detailed(text: str, nlp) -> list[dict]:
    """
    Like ``split_by_coordination`` but returns each part with char offsets.

    Offsets are relative to *text* (the sentence string passed in).

    Returns
    -------
    list[dict]
        Each item has keys ``text``, ``start`` (int), ``end`` (int), and
        ``type`` (always ``"weak"`` — coordination splits are weak boundaries).
        Returns a single-element list when no split is made, so the caller
        always gets a non-empty list.
    """
    if ":" in text:
        return [{"text": text, "start": 0, "end": len(text), "type": "weak"}]

    doc = nlp(text)
    sent = list(doc.sents)[0]

    if not has_splittable_coordination(sent):
        return [{"text": text, "start": 0, "end": len(text), "type": "weak"}]

    valid_commas = _find_valid_commas(sent)
    if not valid_commas:
        return [{"text": text, "start": 0, "end": len(text), "type": "weak"}]

    parts: list[dict] = []
    prev = 0
    for comma_i in valid_commas:
        span = sent[prev:comma_i]
        raw = span.text
        stripped = raw.strip()
        cleaned = re.sub(r"^(and|or|but)\s+", "", stripped, flags=re.I)
        if cleaned:
            lead = len(raw) - len(raw.lstrip())
            conj_skip = len(stripped) - len(cleaned)
            trail = len(raw) - len(raw.rstrip())
            parts.append({
                "text": cleaned,
                "start": span.start_char + lead + conj_skip,
                "end": span.end_char - trail,
                "type": "weak",
            })
        prev = comma_i + 1

    span = sent[prev:]
    raw = span.text
    stripped = raw.strip()
    cleaned = re.sub(r"^(and|or|but)\s+", "", stripped, flags=re.I)
    if cleaned:
        lead = len(raw) - len(raw.lstrip())
        conj_skip = len(stripped) - len(cleaned)
        trail = len(raw) - len(raw.rstrip())
        parts.append({
            "text": cleaned,
            "start": span.start_char + lead + conj_skip,
            "end": span.end_char - trail,
            "type": "weak",
        })

    if len(parts) < 2:
        return [{"text": text, "start": 0, "end": len(text), "type": "weak"}]

    return parts