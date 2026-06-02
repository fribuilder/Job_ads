"""
segmenters/spacy_segmenter.py
==============================
Segmenter combining NLTK upstream splitting with spaCy-based
coordination splitting.
"""
from __future__ import annotations

import spacy


from .nltk_splitter import nltk_split
from .spacy_utils import split_by_coordination, split_by_coordination_detailed


class SpacySegmenter:
    """
    Segmenter using NLTK for sentence splitting and spaCy for
    coordination structure splitting.

    Pipeline
    --------
    1. nltk_split(): sentence splitting with numbered list and
       semicolon handling
    2. split_by_coordination(): coordination structure splitting
       using spaCy dependency relations
    """

    def __init__(
        self,
        model: str = "en_core_web_sm",
        use_coordination_split: bool = True,
    ) -> None:
        self.nlp = spacy.load(model)
        self.use_coordination_split = use_coordination_split

    def segment(self, text: str) -> list[str]:
        """
        Segment a cleaned job description into requirement-bearing clauses.
        """
        if not text or not text.strip():
            return []

        # Step 1: NLTK upstream splitting
        sentences = nltk_split(text)

        if not self.use_coordination_split:
            return sentences

        # Step 2: spaCy coordination splitting
        results = []
        for sent in sentences:
            parts = split_by_coordination(sent, self.nlp)
            results.extend(parts)

        return results

    def segment_detailed(self, text: str) -> list[dict]:
        """
        Segment with a two-level granularity hierarchy.

        Returns Level-1 (NLTK) segments as the top-level list, each annotated
        with char offsets and a ``"strong"`` boundary type (sentence end,
        semicolon, or numbered-list break).  Each top-level item carries a
        ``subsegments`` list of the finer coordination splits (``"weak"``
        boundary type) produced by the spaCy layer.  If coordination splitting
        does not further divide a clause the subsegments list contains that
        clause as a single entry, so no information is lost.

        All char offsets are relative to the original *text* argument.

        Returns
        -------
        list[dict]
            ``[{"text": str, "start": int, "end": int, "type": "strong",
               "subsegments": [{"text": str, "start": int, "end": int,
                                "type": "weak"}, ...]}, ...]``
        """
        if not text or not text.strip():
            return []

        sentences = nltk_split(text)
        result: list[dict] = []
        cursor = 0

        for sent in sentences:
            # Locate the sentence in the original text; nltk_split may normalise
            # whitespace/bullets so we fall back to the cursor if find() misses.
            idx = text.find(sent, cursor)
            if idx == -1:
                start, end = cursor, cursor + len(sent)
            else:
                start, end = idx, idx + len(sent)
                cursor = end

            if self.use_coordination_split:
                subs_raw = split_by_coordination_detailed(sent, self.nlp)
            else:
                subs_raw = [{"text": sent, "start": 0, "end": len(sent), "type": "weak"}]

            # Re-base subsegment offsets onto the original text
            subsegments = [
                {
                    "text": s["text"],
                    "start": start + s["start"],
                    "end": start + s["end"],
                    "type": s["type"],
                }
                for s in subs_raw
            ]

            result.append({
                "text": sent,
                "start": start,
                "end": end,
                "type": "strong",
                "subsegments": subsegments,
            })

        return result

    def segment_batch(self, texts: list[str]) -> list[list[str]]:
        """
        Segment a batch of texts.

        Uses spaCy's nlp.pipe() for efficient batch processing.
        """
        if not texts:
            return []

        # Step 1: NLTK upstream splitting for all texts
        all_sentences = [nltk_split(text) for text in texts]

        if not self.use_coordination_split:
            return all_sentences

        # Step 2: Flatten all sentences for batch spaCy processing
        flat_sentences = []
        text_indices = []  # Track which text each sentence belongs to
        for i, sentences in enumerate(all_sentences):
            for sent in sentences:
                flat_sentences.append(sent)
                text_indices.append(i)

        # Step 3: Batch process with spaCy
        results: list[list[str]] = [[] for _ in texts]
        for doc, text_i in zip(
            self.nlp.pipe(flat_sentences, batch_size=32),
            text_indices
        ):
            sent = list(doc.sents)[0]
            text = sent.text
            parts = split_by_coordination(text, self.nlp)
            results[text_i].extend(parts)

        return results