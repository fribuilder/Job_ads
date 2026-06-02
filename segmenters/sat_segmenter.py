"""
segmenters/sat_segmenter.py
============================
Segment-any-Text (SaT) segmenter backed by the ``wtpsplit`` library.

**For evaluation / sampled subsets only.**  SaT models are 200–400 MB and
inference is significantly slower than rule-based or spaCy segmenters.  Do
*not* wire this segmenter into a full-corpus loop in ``pipeline.py``.  It is
registered in ``segmenters.SEGMENTER_REGISTRY`` under the key ``"sat"`` with
``eval_only=True`` so that automated full-corpus runs skip it by default.
"""
from __future__ import annotations

import numpy as np


class SatSegmenter:
    """
    Clause / sentence segmenter backed by wtpsplit's Segment-any-Text models.

    Parameters
    ----------
    model : str
        wtpsplit model identifier.  Defaults to ``"sat-3l-sm"`` (fastest /
        smallest; ~200 MB download on first use).
    threshold : float | None
        Boundary-probability cut-off forwarded to ``SaT.split()``.
        ``None`` uses the model's baked-in default (recommended).
        Lower values → finer splits; higher values → coarser splits.

    Notes
    -----
    ``wtpsplit`` is imported lazily inside ``__init__`` so that importing
    this module does not hard-require the library.
    """

    def __init__(
        self,
        model: str = "sat-3l-sm",
        threshold: float | None = None,
    ) -> None:
        try:
            from wtpsplit import SaT
        except ImportError as exc:
            raise ImportError(
                "wtpsplit is required for SatSegmenter. "
                "Install it with: pip install wtpsplit"
            ) from exc
        self._sat = SaT(model)
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Public interface (mirrors SpacySegmenter / base convention)
    # ------------------------------------------------------------------

    def segment(self, text: str) -> list[str]:
        """Segment *text* into clauses using SaT."""
        if not text or not text.strip():
            return []
        kwargs: dict = {}
        if self.threshold is not None:
            kwargs["threshold"] = self.threshold
        return [s for s in self._sat.split(text, **kwargs) if s.strip()]

    def segment_batch(self, texts: list[str]) -> list[list[str]]:
        """
        Segment a list of texts.

        Iterates individually rather than using SaT's native batching so that
        empty inputs are handled consistently with ``segment()``.
        """
        return [self.segment(t) for t in texts]

    def boundary_proba(self, text: str) -> np.ndarray:
        """
        Return per-character sentence-boundary probabilities for *text*.

        Wraps ``SaT.predict_proba``; useful for multi-granularity thresholding
        and cross-segmenter analysis without committing to a hard split.

        Returns
        -------
        np.ndarray
            1-D float array of length ``len(text)`` where each value is the
            model's estimated probability that the corresponding character is a
            sentence boundary.
        """
        result = self._sat.predict_proba(text)
        if isinstance(result, np.ndarray):
            return result
        return np.asarray(result)
