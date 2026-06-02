"""
segmenters/hybrid_segmenter.py
==============================
Hybrid segmenter: spaCy primary with SaT fallback for long segments.

Strategy
--------
1. Run the full SpacySegmenter pipeline (NLTK + coordination splitting).
2. Any segment longer than ``long_seg_threshold`` characters is re-segmented
   by SaT, which handles HTML-stripped bullet lists better than rule-based
   or dependency-based methods.
3. Short segments pass through unchanged.

The threshold defaults to 300 chars.  In this corpus, legitimate single
sentences peak at ~240 chars; merged list blobs typically exceed 400 chars,
so 300 is a conservative cut that avoids sending normal prose to SaT while
catching the problematic run-ons.

Usage
-----
    # Reuse already-loaded instances to avoid double model loading:
    hybrid = HybridSegmenter(spacy_segmenter=spacy_seg, sat_segmenter=sat_seg)

    # Or let the class load its own models:
    hybrid = HybridSegmenter()
"""
from __future__ import annotations

from .spacy_segmenter import SpacySegmenter
from .sat_segmenter import SatSegmenter


class HybridSegmenter:
    """
    Segmenter combining spaCy (primary) with SaT fallback for long segments.

    Parameters
    ----------
    spacy_segmenter : SpacySegmenter | None
        Pre-loaded SpacySegmenter instance.  If None, a new one is created
        using ``spacy_model``.
    sat_segmenter : SatSegmenter | None
        Pre-loaded SatSegmenter instance.  If None, a new one is created
        using ``sat_model``.
    spacy_model : str
        spaCy model name (used only when ``spacy_segmenter`` is None).
    sat_model : str
        wtpsplit model identifier (used only when ``sat_segmenter`` is None).
    long_seg_threshold : int
        Segments longer than this many characters are handed to SaT.
    """

    def __init__(
        self,
        spacy_segmenter: SpacySegmenter | None = None,
        sat_segmenter: SatSegmenter | None = None,
        spacy_model: str = "en_core_web_sm",
        sat_model: str = "sat-3l-sm",
        long_seg_threshold: int = 300,
    ) -> None:
        self._spacy = spacy_segmenter or SpacySegmenter(model=spacy_model)
        self._sat   = sat_segmenter   or SatSegmenter(model=sat_model)
        self.threshold = long_seg_threshold

    def segment(self, text: str) -> list[str]:
        """Segment text using spaCy, falling back to SaT for long segments."""
        if not text or not text.strip():
            return []

        result: list[str] = []
        for seg in self._spacy.segment(text):
            if len(seg) > self.threshold:
                result.extend(self._sat.segment(seg))
            else:
                result.append(seg)

        return result

    def segment_batch(self, texts: list[str]) -> list[list[str]]:
        return [self.segment(t) for t in texts]
