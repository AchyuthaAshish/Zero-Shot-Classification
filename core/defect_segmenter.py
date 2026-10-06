"""Defect Segmentation Layer for Industrial Defect Classification.

Conforms to Phase 2 Step 2.2:
1. Converts multi-defect reports into clean, independent defect text segments.
2. Reuses Step 2.1 MultiDefectDetector output and evidence groups rather than duplicating detection logic.
3. Preserves exact character offsets (start_char, end_char) within original text.
4. Preserves connected causal defect chains as single segments (e.g. "because", "due to", "valla", "వల్ల").
5. Preserves same-equipment symptom clusters and compound subjects as single segments.
6. Handles single-defect inputs (1 segment covering original text).
7. Handles unknown / non-defect inputs (0 segments, preserving Step 1.3 Unknown behavior).
8. Handles ambiguous alternative hypotheses ("or") as single segments.
9. Supports English, Telugu-English code-switched, and native Telugu script.
"""

from typing import Dict, List, Optional, Tuple, Any

from core.schemas import (
    DefectSegment,
    DefectSegmentationResult,
    MultiDefectAssessment,
    DefectSignal
)
from core.multi_defect_detector import (
    get_multi_defect_detector,
    MultiDefectDetector
)


class DefectSegmenter:
    """Extracts clean, independent defect text segments from single and multi-defect descriptions."""

    def __init__(self, multi_defect_detector: Optional[MultiDefectDetector] = None):
        self.detector = multi_defect_detector or get_multi_defect_detector()

    def segment_defects(
        self,
        raw_or_preprocessed: Any,
        multi_defect_assessment: Optional[MultiDefectAssessment] = None
    ) -> DefectSegmentationResult:
        """Extracts defect text segments from the input report.

        Returns a DefectSegmentationResult.
        """
        # 1. Resolve raw text
        if hasattr(raw_or_preprocessed, "raw_text"):
            original_text = raw_or_preprocessed.raw_text
        elif hasattr(raw_or_preprocessed, "normalized_text"):
            original_text = raw_or_preprocessed.normalized_text
        else:
            original_text = str(raw_or_preprocessed or "")

        clean_text = original_text.strip()
        if not clean_text:
            return DefectSegmentationResult(
                original_text=original_text,
                is_multi_defect=False,
                segment_count=0,
                segments=[],
                method="rule_based_evidence_span_extraction",
                segmentation_reason="Empty input text contains no defect segments."
            )

        # 2. Leverage Step 2.1 MultiDefectAssessment
        assessment = multi_defect_assessment
        if assessment is None:
            assessment = self.detector.detect_multi_defect(raw_or_preprocessed)

        # -------------------------------------------------------------------
        # Case A: Unknown / Non-Defect / Vague Input (0 Defect Signals)
        # -------------------------------------------------------------------
        if assessment.defect_signal_count == 0 or not assessment.signals:
            return DefectSegmentationResult(
                original_text=original_text,
                is_multi_defect=False,
                segment_count=0,
                segments=[],
                method="rule_based_evidence_span_extraction",
                segmentation_reason=assessment.detection_reason
            )

        # -------------------------------------------------------------------
        # Case B: Single Defect (Single Signal, Causal Chain, or Compound)
        # -------------------------------------------------------------------
        if not assessment.is_multi_defect or assessment.defect_signal_count == 1:
            primary_signal = assessment.signals[0] if assessment.signals else None
            start_char, end_char, actual_span = self._find_exact_span(original_text, clean_text)
            single_segment = DefectSegment(
                segment_id=1,
                text=actual_span,
                start_char=start_char,
                end_char=end_char,
                source_clause=actual_span,
                evidence_group=primary_signal,
                linked_equipment=primary_signal.equipment if primary_signal else None,
                segmentation_reason=assessment.detection_reason
            )
            return DefectSegmentationResult(
                original_text=original_text,
                is_multi_defect=False,
                segment_count=1,
                segments=[single_segment],
                method="rule_based_evidence_span_extraction",
                segmentation_reason=assessment.detection_reason
            )

        # -------------------------------------------------------------------
        # Case C: True Multi-Defect (2 or More Distinct Defect Signals)
        # -------------------------------------------------------------------
        segments: List[DefectSegment] = []
        search_from = 0

        for idx, signal in enumerate(assessment.signals, start=1):
            start_char, end_char, actual_span = self._find_exact_span(
                original_text=original_text,
                span_text=signal.text_span,
                search_from=search_from
            )
            search_from = max(search_from, end_char)

            seg = DefectSegment(
                segment_id=idx,
                text=actual_span,
                start_char=start_char,
                end_char=end_char,
                source_clause=signal.text_span,
                evidence_group=signal,
                linked_equipment=signal.equipment,
                segmentation_reason=f"Segment {idx}: {signal.category} - {signal.symptom or 'reported anomaly'}"
            )
            segments.append(seg)

        categories_summary = ", ".join(s.category for s in assessment.signals)
        reason = f"Extracted {len(segments)} independent defect segments across: {categories_summary}."

        return DefectSegmentationResult(
            original_text=original_text,
            is_multi_defect=True,
            segment_count=len(segments),
            segments=segments,
            method="rule_based_evidence_span_extraction",
            segmentation_reason=reason
        )

    # -----------------------------------------------------------------------
    # Span Extraction Helpers
    # -----------------------------------------------------------------------

    def _find_exact_span(
        self,
        original_text: str,
        span_text: str,
        search_from: int = 0
    ) -> Tuple[int, int, str]:
        """Locates exact start_char and end_char of span_text in original_text."""
        clean_span = span_text.strip()
        if not clean_span:
            return 0, 0, ""

        # 1. Exact case-sensitive match after search_from
        idx = original_text.find(clean_span, search_from)
        if idx != -1:
            return idx, idx + len(clean_span), original_text[idx:idx + len(clean_span)]

        # 2. Case-insensitive match after search_from
        lower_orig = original_text.lower()
        lower_span = clean_span.lower()
        idx = lower_orig.find(lower_span, search_from)
        if idx != -1:
            return idx, idx + len(clean_span), original_text[idx:idx + len(clean_span)]

        # 3. Match from beginning of text if not found after search_from
        idx = original_text.find(clean_span)
        if idx != -1:
            return idx, idx + len(clean_span), original_text[idx:idx + len(clean_span)]

        idx = lower_orig.find(lower_span)
        if idx != -1:
            return idx, idx + len(clean_span), original_text[idx:idx + len(clean_span)]

        # 4. Fallback: return full clean text bounds
        return 0, len(original_text), clean_span


# ---------------------------------------------------------------------------
# Global Singleton & Public Service Function
# ---------------------------------------------------------------------------

_defect_segmenter_instance: Optional[DefectSegmenter] = None


def get_defect_segmenter() -> DefectSegmenter:
    """Returns singleton DefectSegmenter instance."""
    global _defect_segmenter_instance
    if _defect_segmenter_instance is None:
        _defect_segmenter_instance = DefectSegmenter()
    return _defect_segmenter_instance


def segment_defects(
    raw_or_preprocessed: Any,
    multi_defect_assessment: Optional[MultiDefectAssessment] = None
) -> DefectSegmentationResult:
    """Public convenience function to extract defect segments from input text."""
    segmenter = get_defect_segmenter()
    return segmenter.segment_defects(raw_or_preprocessed, multi_defect_assessment=multi_defect_assessment)
