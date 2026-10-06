"""Per-Defect Classification Layer for Industrial Defect Intelligence.

Conforms to Phase 2 Step 2.3:
1. Independently classifies EACH extracted defect segment using existing classification infrastructure.
2. Reuses existing Local ML, Gemini zero-shot, and Hybrid classification pathways without duplicating logic.
3. Assigns each defect its own:
   - category (strictly adhering to canonical 8-category taxonomy)
   - confidence assessment (preserving calibration for local, qualitative for Gemini)
   - reliability (High, Medium, Low)
   - natural language explanation (grounded in its own segment, zero cross-segment contamination)
   - source segment character boundaries (source_start_char, source_end_char)
   - ambiguity assessment
4. Preserves 100% single-defect backward compatibility.
5. Preserves causal single-defect chains (1 defect record).
6. Preserves Unknown non-defect / vague inputs (0 defect records).
7. Preserves ambiguous disjunctive alternative hypotheses (1 defect record with ambiguity flag).
8. Supports same-category multiple defects and different-category multiple defects.
"""

from typing import Dict, List, Optional, Any
from core.schemas import (
    ClassificationResult,
    ConfidenceAssessment,
    AmbiguityAssessment,
    DefectSegment,
    DefectSegmentationResult,
    MultiDefectAssessment,
    PerDefectClassification,
    MultiDefectClassificationResult
)
from core.defect_segmenter import get_defect_segmenter, DefectSegmenter
from core.multi_defect_detector import get_multi_defect_detector, MultiDefectDetector
from taxonomy.repository import REQUIRED_APPROVED_CATEGORIES, get_taxonomy_repository, TaxonomyRepository


class PerDefectClassifier:
    """Orchestrates independent classification of extracted defect segments."""

    def __init__(
        self,
        defect_segmenter: Optional[DefectSegmenter] = None,
        multi_defect_detector: Optional[MultiDefectDetector] = None,
        taxonomy_repo: Optional[TaxonomyRepository] = None
    ):
        self.segmenter = defect_segmenter or get_defect_segmenter()
        self.detector = multi_defect_detector or get_multi_defect_detector()
        self.taxonomy_repo = taxonomy_repo or get_taxonomy_repository()

    def classify_report(
        self,
        raw_or_preprocessed: Any,
        mode: str = "local",
        multi_defect_assessment: Optional[MultiDefectAssessment] = None,
        segmentation_result: Optional[DefectSegmentationResult] = None,
        local_classifier: Optional[Any] = None,
        defect_classifier: Optional[Any] = None
    ) -> MultiDefectClassificationResult:
        """Classifies a defect report, segmenting and independently classifying each defect.

        Returns a structured MultiDefectClassificationResult.
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
            md_res = MultiDefectClassificationResult(
                original_text=original_text,
                is_multi_defect=False,
                defect_count=0,
                defects=[],
                overall_status="unknown",
                method="empty_input_handling",
                segmentation_result=None,
                multi_defect_assessment=None
            )
            from core.multi_defect_validator import validate_multi_defect_result
            md_res.validation_result = validate_multi_defect_result(md_res)
            return md_res

        # 2. Multi-defect assessment (Step 2.1)
        assessment = multi_defect_assessment
        if assessment is None:
            assessment = self.detector.detect_multi_defect(raw_or_preprocessed)

        # 3. Defect segmentation (Step 2.2)
        seg_result = segmentation_result
        if seg_result is None:
            seg_result = self.segmenter.segment_defects(raw_or_preprocessed, multi_defect_assessment=assessment)

        # -------------------------------------------------------------------
        # Case A: Unknown / Non-Defect / Vague Input (0 Segments)
        # -------------------------------------------------------------------
        if seg_result.segment_count == 0 or not seg_result.segments:
            md_res = MultiDefectClassificationResult(
                original_text=original_text,
                is_multi_defect=False,
                defect_count=0,
                defects=[],
                overall_status="unknown",
                method="unknown_evidence_safeguard",
                segmentation_result=seg_result,
                multi_defect_assessment=assessment
            )
            from core.multi_defect_validator import validate_multi_defect_result
            md_res.validation_result = validate_multi_defect_result(md_res)
            return md_res

        # -------------------------------------------------------------------
        # Case B: Single Defect (Single Signal, Causal Chain, or Ambiguous 'or')
        # -------------------------------------------------------------------
        if not seg_result.is_multi_defect or seg_result.segment_count == 1:
            single_seg = seg_result.segments[0]
            single_res = self._classify_single_segment(
                text=single_seg.text,
                mode=mode,
                local_classifier=local_classifier,
                defect_classifier=defect_classifier
            )

            defect = self._build_per_defect(
                defect_id=1,
                segment=single_seg,
                single_res=single_res,
                mode=mode,
                local_classifier=local_classifier,
                defect_classifier=defect_classifier
            )

            md_res = MultiDefectClassificationResult(
                original_text=original_text,
                is_multi_defect=False,
                defect_count=1,
                defects=[defect],
                overall_status=single_res.status,
                method="single_defect_per_defect_classification",
                segmentation_result=seg_result,
                multi_defect_assessment=assessment
            )
            from core.multi_defect_validator import validate_multi_defect_result
            md_res.validation_result = validate_multi_defect_result(md_res)
            return md_res

        # -------------------------------------------------------------------
        # Case C: True Multi-Defect (2 or More Distinct Defect Segments)
        # -------------------------------------------------------------------
        return self.classify_segments(
            segments=seg_result.segments,
            original_text=original_text,
            mode=mode,
            multi_defect_assessment=assessment,
            segmentation_result=seg_result,
            local_classifier=local_classifier,
            defect_classifier=defect_classifier
        )

    def classify_segments(
        self,
        segments: List[DefectSegment],
        original_text: str,
        mode: str = "local",
        multi_defect_assessment: Optional[MultiDefectAssessment] = None,
        segmentation_result: Optional[DefectSegmentationResult] = None,
        local_classifier: Optional[Any] = None,
        defect_classifier: Optional[Any] = None
    ) -> MultiDefectClassificationResult:
        """Independently classifies each segment in the provided list.

        Returns an aggregate MultiDefectClassificationResult.
        """
        defects: List[PerDefectClassification] = []

        for idx, seg in enumerate(segments, start=1):
            single_res = self._classify_single_segment(
                text=seg.text,
                mode=mode,
                local_classifier=local_classifier,
                defect_classifier=defect_classifier
            )

            defect = self._build_per_defect(
                defect_id=idx,
                segment=seg,
                single_res=single_res,
                mode=mode,
                local_classifier=local_classifier,
                defect_classifier=defect_classifier
            )
            defects.append(defect)

        # Determine overall aggregate status
        statuses = [d.status for d in defects]
        if all(s == "success" for s in statuses):
            overall_status = "success"
        elif all(s in ("unknown", "model_error", "validation_error", "system_error") for s in statuses):
            overall_status = "error" if any("error" in s for s in statuses) else "unknown"
        elif any("error" in s for s in statuses):
            overall_status = "partial_error"
        else:
            overall_status = "success"

        md_res = MultiDefectClassificationResult(
            original_text=original_text,
            is_multi_defect=len(defects) > 1,
            defect_count=len(defects),
            defects=defects,
            overall_status=overall_status,
            method="per_segment_independent_classification",
            segmentation_result=segmentation_result,
            multi_defect_assessment=multi_defect_assessment
        )
        from core.multi_defect_validator import validate_multi_defect_result
        md_res.validation_result = validate_multi_defect_result(md_res)
        return md_res

    def _classify_single_segment(
        self,
        text: str,
        mode: str = "local",
        local_classifier: Optional[Any] = None,
        defect_classifier: Optional[Any] = None
    ) -> ClassificationResult:
        """Classifies an individual defect text segment through the appropriate pathway."""
        clean_text = text.strip()
        active_mode = (mode or "local").lower()

        # Pathway 1: Pure Local ML
        if active_mode == "local":
            from ml.local_classifier import get_local_classifier
            local_clf = local_classifier or get_local_classifier()
            return local_clf.classify(clean_text, _is_subsegment=True)

        # Pathway 2: External LLM / Gemini Zero-Shot
        if active_mode in ("gemini", "external", "aimlapi"):
            from classification.classifier import get_defect_classifier
            clf = defect_classifier or get_defect_classifier(mode=active_mode)
            return clf._classify_gemini(clean_text, _is_subsegment=True)

        # Pathway 3: HYBRID (Local ML first with external LLM fallback)
        from classification.classifier import get_defect_classifier
        clf = defect_classifier or get_defect_classifier(mode="hybrid")
        return clf.classify(clean_text, mode_override="hybrid", _is_subsegment=True)

    def _build_per_defect(
        self,
        defect_id: int,
        segment: DefectSegment,
        single_res: ClassificationResult,
        mode: str,
        local_classifier: Optional[Any] = None,
        defect_classifier: Optional[Any] = None
    ) -> PerDefectClassification:
        """Constructs a validated PerDefectClassification record from a ClassificationResult."""
        # Enforce canonical taxonomy validation
        cat = single_res.category
        status = single_res.status
        reliability = single_res.reliability

        if not self.taxonomy_repo.is_valid_category(cat):
            cat = "Unknown"
            status = "validation_error"
            reliability = "Low"

        # Resolve provider and model
        provider = single_res.model_source
        model_name = None
        if mode == "local":
            from ml.local_classifier import get_local_classifier
            local_clf = local_classifier or get_local_classifier()
            model_name = getattr(local_clf, "model_type", "local_ml")
        else:
            model_name = single_res.model_source

        # Extract confidence metrics
        ca = single_res.confidence_assessment
        raw_score = ca.raw_score if ca else single_res.confidence
        calibrated_prob = ca.calibrated_prob if ca else None
        top2_margin = ca.top2_margin if ca else None

        return PerDefectClassification(
            defect_id=defect_id,
            segment_id=segment.segment_id,
            text=segment.text,
            category=cat,
            confidence_assessment=ca or ConfidenceAssessment(
                level=reliability,
                approximate_range="Qualitative / N/A" if mode != "local" else "Uncalibrated",
                raw_score=raw_score,
                calibrated_prob=calibrated_prob,
                top2_margin=top2_margin,
                is_calibrated=bool(ca and ca.is_calibrated),
                calibration_method=ca.calibration_method if ca else None,
                is_ambiguous=bool(single_res.ambiguity_assessment and single_res.ambiguity_assessment.is_ambiguous)
            ),
            reliability=reliability,
            explanation=single_res.reason,
            classification_mode=mode,
            provider=provider,
            model=model_name,
            ambiguity_assessment=single_res.ambiguity_assessment,
            source_start_char=segment.start_char,
            source_end_char=segment.end_char,
            status=status,
            confidence=single_res.confidence,
            raw_score=raw_score,
            calibrated_prob=calibrated_prob,
            top2_margin=top2_margin
        )


# ---------------------------------------------------------------------------
# Global Singleton & Public Service Function
# ---------------------------------------------------------------------------

_per_defect_classifier_instance: Optional[PerDefectClassifier] = None


def get_per_defect_classifier() -> PerDefectClassifier:
    """Returns singleton PerDefectClassifier instance."""
    global _per_defect_classifier_instance
    if _per_defect_classifier_instance is None:
        _per_defect_classifier_instance = PerDefectClassifier()
    return _per_defect_classifier_instance


def classify_per_defect(
    raw_or_preprocessed: Any,
    mode: str = "local",
    multi_defect_assessment: Optional[MultiDefectAssessment] = None,
    segmentation_result: Optional[DefectSegmentationResult] = None,
    local_classifier: Optional[Any] = None,
    defect_classifier: Optional[Any] = None
) -> MultiDefectClassificationResult:
    """Public service function to classify each defect segment in a report independently."""
    classifier = get_per_defect_classifier()
    return classifier.classify_report(
        raw_or_preprocessed=raw_or_preprocessed,
        mode=mode,
        multi_defect_assessment=multi_defect_assessment,
        segmentation_result=segmentation_result,
        local_classifier=local_classifier,
        defect_classifier=defect_classifier
    )
