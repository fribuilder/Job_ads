"""
segmenters
==========
Segmenter package for the USAJOBS → O*NET pipeline.

Registry
--------
``SEGMENTER_REGISTRY`` maps string keys to segmenter metadata.
``eval_only=True`` means the segmenter must not be wired into full-corpus
pipeline runs (it is reserved for sampled evaluation subsets).
"""
from __future__ import annotations

from .spacy_segmenter import SpacySegmenter
from .sat_segmenter import SatSegmenter

SEGMENTER_REGISTRY: dict[str, dict] = {
    "spacy": {"cls": SpacySegmenter, "eval_only": False},
    "sat":   {"cls": SatSegmenter,   "eval_only": True},
}

__all__ = ["SpacySegmenter", "SatSegmenter", "SEGMENTER_REGISTRY"]
