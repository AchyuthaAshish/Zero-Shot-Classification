"""Multi-Defect Validation Layer for Industrial Defect Classification.

Conforms to Phase 2 Step 2.4:
1. Validates that multi-defect detection, defect segmentation, and per-defect classification
   outputs are internally consistent before future database persistence (Step 2.5).
2. Purely observational: does NOT alter predictions, confidence values, explanations,
   or source text.
3. Implements deterministic validation checks for:
   - A. Segment count consistency
   - B. Segment ID consistency
   - C. Defect ID consistency
   - D. Source span validity & non-overlapping boundaries
   - E. Order consistency
   - F. Taxonomy compliance (8 canonical categories)
   - G. Confidence structure validity (calibrated [0, 1] vs qualitative)
   - H. Explanation validity
   - I. Multi-defect flag consistency across components
   - J. Unknown non-defect consistency
   - K. Ambiguity consistency
4. Produces structured ValidationIssue and MultiDefectValidationResult.
5. Standard status model: VALID, PARTIAL, INVALID, UNKNOWN.
"""

from typing import Dict, List, Optional, Any, Union
from core.schemas import (
    ClassificationResult,
    ConfidenceAssessment,
    AmbiguityAssessment,
    DefectSegment,
    DefectSegmentationResult,
    MultiDefectAssessment,
    PerDefectClassification,
    MultiDefectClassificationResult,
    ValidationIssue,
    MultiDefectValidationResult
)
from taxonomy.repository import REQUIRED_APPROVED_CATEGORIES, get_taxonomy_repository, TaxonomyRepository


class MultiDefectValidator:
    """Dedicated validation engine for single- and multi-defect classification results."""

    def __init__(self, taxonomy_repo: Optional[TaxonomyRepository] = None):
        self.taxonomy_repo = taxonomy_repo or get_taxonomy_repository()

    def validate(
        self,
        classification_result: Optional[Union[MultiDefectClassificationResult, ClassificationResult]] = None,
        segmentation_result: Optional[DefectSegmentationResult] = None,
        multi_defect_assessment: Optional[MultiDefectAssessment] = None,
        original_text: Optional[str] = None,
        defects: Optional[List[PerDefectClassification]] = None,
        segments: Optional[List[DefectSegment]] = None
    ) -> MultiDefectValidationResult:
        """Deterministically validates consistency across detection, segmentation, and classification.

        Does NOT alter predictions or classifier outputs.
        Returns a structured MultiDefectValidationResult.
        """
        # 1. Unpack and resolve inputs
        target_md_result: Optional[MultiDefectClassificationResult] = None
        target_single_result: Optional[ClassificationResult] = None

        if classification_result is not None:
            if isinstance(classification_result, MultiDefectClassificationResult):
                target_md_result = classification_result
                if original_text is None:
                    original_text = classification_result.original_text
                if segmentation_result is None:
                    segmentation_result = classification_result.segmentation_result
                if multi_defect_assessment is None:
                    multi_defect_assessment = classification_result.multi_defect_assessment
                if defects is None:
                    defects = classification_result.defects
            elif isinstance(classification_result, ClassificationResult):
                target_single_result = classification_result
                if original_text is None:
                    original_text = classification_result.original_description
                if multi_defect_assessment is None:
                    multi_defect_assessment = classification_result.multi_defect_assessment
                if segmentation_result is None:
                    segmentation_result = classification_result.segmentation_result
                if classification_result.multi_defect_classification is not None:
                    target_md_result = classification_result.multi_defect_classification
                    if segmentation_result is None:
                        segmentation_result = target_md_result.segmentation_result
                    if multi_defect_assessment is None:
                        multi_defect_assessment = target_md_result.multi_defect_assessment
                    if defects is None:
                        defects = target_md_result.defects

        if original_text is None and segmentation_result is not None:
            original_text = segmentation_result.original_text

        text_str = str(original_text or "")
        text_len = len(text_str)

        resolved_segments: List[DefectSegment] = []
        if segments is not None:
            resolved_segments = list(segments)
        elif segmentation_result is not None and segmentation_result.segments is not None:
            resolved_segments = list(segmentation_result.segments)

        resolved_defects: List[PerDefectClassification] = []
        if defects is not None:
            resolved_defects = list(defects)
        elif target_md_result is not None and target_md_result.defects is not None:
            resolved_defects = list(target_md_result.defects)

        errors: List[ValidationIssue] = []
        warnings: List[ValidationIssue] = []

        checks: Dict[str, bool] = {
            "segment_count_consistency": True,
            "segment_id_consistency": True,
            "defect_id_consistency": True,
            "source_span_validity": True,
            "span_non_overlapping": True,
            "order_consistency": True,
            "taxonomy_validity": True,
            "confidence_structure_validity": True,
            "explanation_validity": True,
            "multi_defect_flag_consistency": True,
            "unknown_consistency": True,
            "ambiguity_consistency": True,
        }

        # -------------------------------------------------------------------
        # Check J & Non-Defect Safeguards: Unknown / Empty Inputs
        # -------------------------------------------------------------------
        is_unknown_case = False
        if not text_str.strip():
            is_unknown_case = True
        elif (
            target_single_result is not None
            and target_single_result.category == "Unknown"
            and target_single_result.status in ("unknown", "model_error", "system_error")
            and len(resolved_segments) == 0
        ):
            is_unknown_case = True
        elif (
            target_md_result is not None
            and target_md_result.overall_status in ("unknown", "model_error", "system_error")
            and len(resolved_segments) == 0
            and len(resolved_defects) == 0
        ):
            is_unknown_case = True

        if is_unknown_case:
            if len(resolved_defects) > 0:
                errors.append(ValidationIssue(
                    code="FABRICATED_DEFECTS_FOR_UNKNOWN",
                    severity="error",
                    message=f"Defects ({len(resolved_defects)}) were fabricated for an Unknown non-defect report."
                ))
                checks["unknown_consistency"] = False

            if target_md_result and target_md_result.is_multi_defect:
                errors.append(ValidationIssue(
                    code="UNKNOWN_MARKED_MULTI_DEFECT",
                    severity="error",
                    message="Unknown non-defect report marked as multi-defect."
                ))
                checks["unknown_consistency"] = False

            if segmentation_result and segmentation_result.is_multi_defect:
                errors.append(ValidationIssue(
                    code="UNKNOWN_MARKED_MULTI_DEFECT",
                    severity="error",
                    message="Unknown non-defect segmentation marked as multi-defect."
                ))
                checks["unknown_consistency"] = False

            if len(resolved_segments) > 0:
                errors.append(ValidationIssue(
                    code="SEGMENTS_CREATED_FOR_UNKNOWN",
                    severity="error",
                    message=f"Created {len(resolved_segments)} segments for an Unknown non-defect report."
                ))
                checks["unknown_consistency"] = False

            is_valid = len(errors) == 0
            return MultiDefectValidationResult(
                is_valid=is_valid,
                status="UNKNOWN" if is_valid else "INVALID",
                errors=errors,
                warnings=warnings,
                checks=checks,
                validated_defect_count=len(resolved_defects),
                expected_segment_count=len(resolved_segments),
                validation_method="deterministic_rule_based_validator"
            )

        # -------------------------------------------------------------------
        # Check A: Segment Count Consistency
        # -------------------------------------------------------------------
        expected_seg_count = (
            segmentation_result.segment_count
            if segmentation_result is not None
            else len(resolved_segments)
        )

        if len(resolved_segments) != expected_seg_count:
            errors.append(ValidationIssue(
                code="SEGMENT_COUNT_MISMATCH",
                severity="error",
                message=f"SegmentationResult.segment_count ({expected_seg_count}) does not match actual segments count ({len(resolved_segments)})."
            ))
            checks["segment_count_consistency"] = False

        if len(resolved_segments) != len(resolved_defects):
            errors.append(ValidationIssue(
                code="SEGMENT_CLASSIFICATION_COUNT_MISMATCH",
                severity="error",
                message=f"Segment count ({len(resolved_segments)}) does not match defect classification count ({len(resolved_defects)})."
            ))
            checks["segment_count_consistency"] = False

        # -------------------------------------------------------------------
        # Check B: Segment ID Consistency
        # -------------------------------------------------------------------
        seen_seg_ids = set()
        for seg in resolved_segments:
            if seg.segment_id is None or seg.segment_id <= 0:
                errors.append(ValidationIssue(
                    code="MISSING_SEGMENT_ID",
                    severity="error",
                    message=f"Segment has missing or non-positive segment_id: {seg.segment_id}",
                    segment_id=seg.segment_id
                ))
                checks["segment_id_consistency"] = False
            elif seg.segment_id in seen_seg_ids:
                errors.append(ValidationIssue(
                    code="DUPLICATE_SEGMENT_ID",
                    severity="error",
                    message=f"Duplicate segment_id {seg.segment_id} found in segmentation result.",
                    segment_id=seg.segment_id
                ))
                checks["segment_id_consistency"] = False
            else:
                seen_seg_ids.add(seg.segment_id)

        seen_defect_seg_ids = set()
        for d in resolved_defects:
            if d.segment_id is None or d.segment_id <= 0:
                errors.append(ValidationIssue(
                    code="MISSING_SEGMENT_ID",
                    severity="error",
                    message=f"Defect {d.defect_id} is missing a valid segment_id: {d.segment_id}",
                    defect_id=d.defect_id
                ))
                checks["segment_id_consistency"] = False
            elif seen_seg_ids and d.segment_id not in seen_seg_ids:
                errors.append(ValidationIssue(
                    code="UNKNOWN_SEGMENT_ID",
                    severity="error",
                    message=f"Defect {d.defect_id} references unknown segment_id {d.segment_id}.",
                    defect_id=d.defect_id,
                    segment_id=d.segment_id
                ))
                checks["segment_id_consistency"] = False
            elif d.segment_id in seen_defect_seg_ids:
                errors.append(ValidationIssue(
                    code="DUPLICATE_SEGMENT_ID",
                    severity="error",
                    message=f"Multiple classifications reference the same segment_id {d.segment_id}.",
                    defect_id=d.defect_id,
                    segment_id=d.segment_id
                ))
                checks["segment_id_consistency"] = False
            else:
                seen_defect_seg_ids.add(d.segment_id)

        # -------------------------------------------------------------------
        # Check C: Defect ID Consistency
        # -------------------------------------------------------------------
        seen_defect_ids = set()
        for d in resolved_defects:
            if d.defect_id is None or d.defect_id <= 0:
                errors.append(ValidationIssue(
                    code="MISSING_DEFECT_ID",
                    severity="error",
                    message=f"Defect has missing or non-positive defect_id: {d.defect_id}",
                    defect_id=d.defect_id
                ))
                checks["defect_id_consistency"] = False
            elif d.defect_id in seen_defect_ids:
                errors.append(ValidationIssue(
                    code="DUPLICATE_DEFECT_ID",
                    severity="error",
                    message=f"Duplicate defect_id {d.defect_id} found across classifications.",
                    defect_id=d.defect_id
                ))
                checks["defect_id_consistency"] = False
            else:
                seen_defect_ids.add(d.defect_id)

        # -------------------------------------------------------------------
        # Check D: Source Span Validation & Non-Overlapping Spans
        # -------------------------------------------------------------------
        for d in resolved_defects:
            s_start = d.source_start_char
            s_end = d.source_end_char
            if s_start < 0 or s_end <= s_start or (text_str and s_end > text_len):
                errors.append(ValidationIssue(
                    code="SPAN_OUT_OF_BOUNDS",
                    severity="error",
                    message=f"Defect {d.defect_id} source span [{s_start}:{s_end}] is out of bounds for text length {text_len}.",
                    defect_id=d.defect_id,
                    segment_id=d.segment_id
                ))
                checks["source_span_validity"] = False
            elif text_str and text_str[s_start:s_end] != d.text:
                errors.append(ValidationIssue(
                    code="SPAN_TEXT_MISMATCH",
                    severity="error",
                    message=f"Defect {d.defect_id} text '{d.text}' does not match original text span '{text_str[s_start:s_end]}' at [{s_start}:{s_end}].",
                    defect_id=d.defect_id,
                    segment_id=d.segment_id
                ))
                checks["source_span_validity"] = False

        for seg in resolved_segments:
            if seg.start_char < 0 or seg.end_char <= seg.start_char or (text_str and seg.end_char > text_len):
                errors.append(ValidationIssue(
                    code="SPAN_OUT_OF_BOUNDS",
                    severity="error",
                    message=f"Segment {seg.segment_id} span [{seg.start_char}:{seg.end_char}] is out of bounds for text length {text_len}.",
                    segment_id=seg.segment_id
                ))
                checks["source_span_validity"] = False
            elif text_str and text_str[seg.start_char:seg.end_char] != seg.text:
                errors.append(ValidationIssue(
                    code="SPAN_TEXT_MISMATCH",
                    severity="error",
                    message=f"Segment {seg.segment_id} text '{seg.text}' does not match original text slice at [{seg.start_char}:{seg.end_char}].",
                    segment_id=seg.segment_id
                ))
                checks["source_span_validity"] = False

        # Non-overlapping checks for segments
        for i in range(len(resolved_segments)):
            for j in range(i + 1, len(resolved_segments)):
                s1 = resolved_segments[i]
                s2 = resolved_segments[j]
                if max(s1.start_char, s2.start_char) < min(s1.end_char, s2.end_char):
                    errors.append(ValidationIssue(
                        code="OVERLAPPING_SPANS",
                        severity="error",
                        message=f"Segments {s1.segment_id} [{s1.start_char}:{s1.end_char}] and {s2.segment_id} [{s2.start_char}:{s2.end_char}] overlap.",
                        segment_id=s1.segment_id
                    ))
                    checks["span_non_overlapping"] = False

        # Non-overlapping checks for defects
        for i in range(len(resolved_defects)):
            for j in range(i + 1, len(resolved_defects)):
                d1 = resolved_defects[i]
                d2 = resolved_defects[j]
                if max(d1.source_start_char, d2.source_start_char) < min(d1.source_end_char, d2.source_end_char):
                    errors.append(ValidationIssue(
                        code="OVERLAPPING_SPANS",
                        severity="error",
                        message=f"Defects {d1.defect_id} [{d1.source_start_char}:{d1.source_end_char}] and {d2.defect_id} [{d2.source_start_char}:{d2.source_end_char}] overlap.",
                        defect_id=d1.defect_id
                    ))
                    checks["span_non_overlapping"] = False

        # -------------------------------------------------------------------
        # Check E: Order Consistency
        # -------------------------------------------------------------------
        if len(resolved_defects) == len(resolved_segments):
            for idx in range(len(resolved_defects)):
                d = resolved_defects[idx]
                s = resolved_segments[idx]
                if d.segment_id != s.segment_id:
                    errors.append(ValidationIssue(
                        code="ORDER_MISMATCH",
                        severity="error",
                        message=f"Defect at index {idx} has segment_id {d.segment_id}, but segment at index {idx} has segment_id {s.segment_id}.",
                        defect_id=d.defect_id,
                        segment_id=d.segment_id
                    ))
                    checks["order_consistency"] = False
                if d.source_start_char != s.start_char or d.source_end_char != s.end_char:
                    errors.append(ValidationIssue(
                        code="ORDER_MISMATCH",
                        severity="error",
                        message=f"Defect at index {idx} span [{d.source_start_char}:{d.source_end_char}] does not match segment span [{s.start_char}:{s.end_char}].",
                        defect_id=d.defect_id,
                        segment_id=d.segment_id
                    ))
                    checks["order_consistency"] = False

        for i in range(len(resolved_segments) - 1):
            if resolved_segments[i].start_char > resolved_segments[i + 1].start_char:
                errors.append(ValidationIssue(
                    code="ORDER_MISMATCH",
                    severity="error",
                    message="Segments are not in order of appearance in original text.",
                    segment_id=resolved_segments[i].segment_id
                ))
                checks["order_consistency"] = False

        # -------------------------------------------------------------------
        # Check F: Taxonomy Validation
        # -------------------------------------------------------------------
        for d in resolved_defects:
            if not self.taxonomy_repo.is_valid_category(d.category):
                errors.append(ValidationIssue(
                    code="INVALID_TAXONOMY_CATEGORY",
                    severity="error",
                    message=f"Defect {d.defect_id} category '{d.category}' is not an approved taxonomy category.",
                    defect_id=d.defect_id,
                    segment_id=d.segment_id
                ))
                checks["taxonomy_validity"] = False

        if target_single_result and not self.taxonomy_repo.is_valid_category(target_single_result.category):
            errors.append(ValidationIssue(
                code="INVALID_TAXONOMY_CATEGORY",
                severity="error",
                message=f"Overall classification category '{target_single_result.category}' is not an approved taxonomy category."
            ))
            checks["taxonomy_validity"] = False

        # -------------------------------------------------------------------
        # Check G: Confidence Structure Validation
        # -------------------------------------------------------------------
        for d in resolved_defects:
            ca = d.confidence_assessment
            if ca is None:
                errors.append(ValidationIssue(
                    code="MISSING_CONFIDENCE_ASSESSMENT",
                    severity="error",
                    message=f"Defect {d.defect_id} is missing ConfidenceAssessment.",
                    defect_id=d.defect_id,
                    segment_id=d.segment_id
                ))
                checks["confidence_structure_validity"] = False
            else:
                if ca.level not in ("High", "Medium", "Low", "Uncertain"):
                    errors.append(ValidationIssue(
                        code="INVALID_CONFIDENCE_LEVEL",
                        severity="error",
                        message=f"Defect {d.defect_id} has invalid confidence level '{ca.level}'.",
                        defect_id=d.defect_id,
                        segment_id=d.segment_id
                    ))
                    checks["confidence_structure_validity"] = False

                if ca.calibrated_prob is not None:
                    if (
                        not isinstance(ca.calibrated_prob, (int, float))
                        or ca.calibrated_prob < 0.0
                        or ca.calibrated_prob > 1.0
                    ):
                        errors.append(ValidationIssue(
                            code="INVALID_CALIBRATED_PROBABILITY",
                            severity="error",
                            message=f"Defect {d.defect_id} calibrated probability ({ca.calibrated_prob}) is outside [0.0, 1.0].",
                            defect_id=d.defect_id,
                            segment_id=d.segment_id
                        ))
                        checks["confidence_structure_validity"] = False

                if d.calibrated_prob is not None:
                    if (
                        not isinstance(d.calibrated_prob, (int, float))
                        or d.calibrated_prob < 0.0
                        or d.calibrated_prob > 1.0
                    ):
                        errors.append(ValidationIssue(
                            code="INVALID_CALIBRATED_PROBABILITY",
                            severity="error",
                            message=f"Defect {d.defect_id} calibrated_prob attribute ({d.calibrated_prob}) is outside [0.0, 1.0].",
                            defect_id=d.defect_id,
                            segment_id=d.segment_id
                        ))
                        checks["confidence_structure_validity"] = False

                if ca.raw_score is not None:
                    if not isinstance(ca.raw_score, (int, float)) or (
                        isinstance(ca.raw_score, float) and ca.raw_score != ca.raw_score
                    ):
                        errors.append(ValidationIssue(
                            code="INVALID_RAW_SCORE",
                            severity="error",
                            message=f"Defect {d.defect_id} raw_score is not a valid number.",
                            defect_id=d.defect_id,
                            segment_id=d.segment_id
                        ))
                        checks["confidence_structure_validity"] = False

                if ca.top2_margin is not None:
                    if (
                        not isinstance(ca.top2_margin, (int, float))
                        or ca.top2_margin < 0.0
                        or ca.top2_margin > 1.0
                    ):
                        errors.append(ValidationIssue(
                            code="INVALID_TOP2_MARGIN",
                            severity="error",
                            message=f"Defect {d.defect_id} top2_margin ({ca.top2_margin}) is outside [0.0, 1.0].",
                            defect_id=d.defect_id,
                            segment_id=d.segment_id
                        ))
                        checks["confidence_structure_validity"] = False

                # Warnings for low confidence
                if ca.level == "Low" or (ca.calibrated_prob is not None and ca.calibrated_prob < 0.70):
                    warnings.append(ValidationIssue(
                        code="LOW_CONFIDENCE",
                        severity="warning",
                        message=f"Defect {d.defect_id} has low confidence (level={ca.level}, prob={ca.calibrated_prob}).",
                        defect_id=d.defect_id,
                        segment_id=d.segment_id
                    ))

                # Warning for missing model/provider metadata
                if d.provider is None or d.model is None:
                    warnings.append(ValidationIssue(
                        code="MISSING_MODEL_METADATA",
                        severity="warning",
                        message=f"Defect {d.defect_id} is missing provider or model metadata.",
                        defect_id=d.defect_id,
                        segment_id=d.segment_id
                    ))

        # -------------------------------------------------------------------
        # Check H: Explanation Validation
        # -------------------------------------------------------------------
        for d in resolved_defects:
            if not d.explanation or not str(d.explanation).strip():
                errors.append(ValidationIssue(
                    code="MISSING_EXPLANATION",
                    severity="error",
                    message=f"Defect {d.defect_id} is missing an explanation.",
                    defect_id=d.defect_id,
                    segment_id=d.segment_id
                ))
                checks["explanation_validity"] = False
            else:
                words = str(d.explanation).strip().split()
                if len(words) < 3:
                    warnings.append(ValidationIssue(
                        code="SHORT_EXPLANATION",
                        severity="warning",
                        message=f"Defect {d.defect_id} explanation is very short ({len(words)} words).",
                        defect_id=d.defect_id,
                        segment_id=d.segment_id
                    ))

                # Cross-segment contamination check
                if len(resolved_segments) > 1:
                    for other_seg in resolved_segments:
                        if other_seg.segment_id != d.segment_id and other_seg.linked_equipment:
                            other_eq = other_seg.linked_equipment.lower()
                            if other_eq in d.explanation.lower():
                                own_eq = getattr(d, "linked_equipment", None)
                                if not own_eq or own_eq.lower() not in d.explanation.lower():
                                    if d.text.lower() not in d.explanation.lower():
                                        warnings.append(ValidationIssue(
                                            code="CROSS_SEGMENT_EXPLANATION_MISMATCH",
                                            severity="warning",
                                            message=f"Defect {d.defect_id} explanation references equipment from segment {other_seg.segment_id} ('{other_seg.linked_equipment}').",
                                            defect_id=d.defect_id,
                                            segment_id=d.segment_id
                                        ))

        # -------------------------------------------------------------------
        # Check I: Multi-Defect Flag Consistency
        # -------------------------------------------------------------------
        num_segs = len(resolved_segments)
        if num_segs >= 2:
            if target_md_result and not target_md_result.is_multi_defect:
                errors.append(ValidationIssue(
                    code="INCONSISTENT_MULTI_DEFECT_FLAG",
                    severity="error",
                    message=f"MultiDefectClassificationResult.is_multi_defect is False, but {num_segs} segments exist."
                ))
                checks["multi_defect_flag_consistency"] = False
            if segmentation_result and not segmentation_result.is_multi_defect:
                errors.append(ValidationIssue(
                    code="INCONSISTENT_MULTI_DEFECT_FLAG",
                    severity="error",
                    message=f"DefectSegmentationResult.is_multi_defect is False, but {num_segs} segments exist."
                ))
                checks["multi_defect_flag_consistency"] = False
            if multi_defect_assessment and not multi_defect_assessment.is_multi_defect:
                errors.append(ValidationIssue(
                    code="INCONSISTENT_MULTI_DEFECT_FLAG",
                    severity="error",
                    message=f"MultiDefectAssessment.is_multi_defect is False, but {num_segs} segments exist."
                ))
                checks["multi_defect_flag_consistency"] = False
        elif num_segs == 1:
            if target_md_result and target_md_result.is_multi_defect:
                errors.append(ValidationIssue(
                    code="INCONSISTENT_MULTI_DEFECT_FLAG",
                    severity="error",
                    message="MultiDefectClassificationResult.is_multi_defect is True, but only 1 segment exists."
                ))
                checks["multi_defect_flag_consistency"] = False
            if segmentation_result and segmentation_result.is_multi_defect:
                errors.append(ValidationIssue(
                    code="INCONSISTENT_MULTI_DEFECT_FLAG",
                    severity="error",
                    message="DefectSegmentationResult.is_multi_defect is True, but only 1 segment exists."
                ))
                checks["multi_defect_flag_consistency"] = False
            if multi_defect_assessment and multi_defect_assessment.is_multi_defect:
                errors.append(ValidationIssue(
                    code="INCONSISTENT_MULTI_DEFECT_FLAG",
                    severity="error",
                    message="MultiDefectAssessment.is_multi_defect is True, but only 1 segment exists."
                ))
                checks["multi_defect_flag_consistency"] = False
        elif num_segs == 0:
            if target_md_result and target_md_result.is_multi_defect:
                errors.append(ValidationIssue(
                    code="INCONSISTENT_MULTI_DEFECT_FLAG",
                    severity="error",
                    message="MultiDefectClassificationResult.is_multi_defect is True for 0 segments."
                ))
                checks["multi_defect_flag_consistency"] = False
            if segmentation_result and segmentation_result.is_multi_defect:
                errors.append(ValidationIssue(
                    code="INCONSISTENT_MULTI_DEFECT_FLAG",
                    severity="error",
                    message="DefectSegmentationResult.is_multi_defect is True for 0 segments."
                ))
                checks["multi_defect_flag_consistency"] = False

        # -------------------------------------------------------------------
        # Check K: Ambiguity Consistency
        # -------------------------------------------------------------------
        is_disjunctive_ambiguity = False
        if multi_defect_assessment is not None and "alternative hypothetical causes" in str(multi_defect_assessment.detection_reason).lower():
            is_disjunctive_ambiguity = True
        elif (" or " in f" {text_str.lower()} " or " లేదా " in f" {text_str.lower()} ") and (
            " and " not in f" {text_str.lower()} " and " మరియు " not in f" {text_str.lower()} "
        ):
            is_disjunctive_ambiguity = True

        if is_disjunctive_ambiguity:
            if (
                (target_md_result and target_md_result.is_multi_defect)
                or (segmentation_result and segmentation_result.is_multi_defect)
                or len(resolved_segments) > 1
            ):
                errors.append(ValidationIssue(
                    code="AMBIGUITY_MISTAKEN_FOR_MULTI_DEFECT",
                    severity="error",
                    message="Ambiguous single-defect alternative hypothesis was erroneously marked or split as multiple defects."
                ))
                checks["ambiguity_consistency"] = False
            else:
                warnings.append(ValidationIssue(
                    code="AMBIGUOUS_DEFECT",
                    severity="warning",
                    message="Report contains ambiguous alternative hypotheses ('or')."
                ))

        for d in resolved_defects:
            # If an individual defect segment contains disjunctive 'or' or low confidence with ambiguity
            d_text_lower = f" {d.text.lower()} "
            if (" or " in d_text_lower or " లేదా " in d_text_lower) and not any(w.code == "AMBIGUOUS_DEFECT" for w in warnings):
                warnings.append(ValidationIssue(
                    code="AMBIGUOUS_DEFECT",
                    severity="warning",
                    message=f"Defect {d.defect_id} contains internal disjunctive alternative hypotheses.",
                    defect_id=d.defect_id,
                    segment_id=d.segment_id
                ))
            elif (
                d.ambiguity_assessment
                and d.ambiguity_assessment.is_ambiguous
                and (d.reliability == "Low" or (d.confidence_assessment and d.confidence_assessment.level == "Low"))
                and not any(w.code == "AMBIGUOUS_DEFECT" and w.defect_id == d.defect_id for w in warnings)
            ):
                warnings.append(ValidationIssue(
                    code="AMBIGUOUS_DEFECT",
                    severity="warning",
                    message=f"Defect {d.defect_id} has low confidence and competing categories: {d.ambiguity_assessment.reason or 'competing categories'}",
                    defect_id=d.defect_id,
                    segment_id=d.segment_id
                ))

        # -------------------------------------------------------------------
        # Final Status Resolution
        # -------------------------------------------------------------------
        if errors:
            status = "INVALID"
            is_valid = False
        elif len(resolved_segments) == 0 and len(resolved_defects) == 0:
            status = "UNKNOWN"
            is_valid = True
        elif warnings:
            status = "PARTIAL"
            is_valid = True
        else:
            status = "VALID"
            is_valid = True

        return MultiDefectValidationResult(
            is_valid=is_valid,
            status=status,
            errors=errors,
            warnings=warnings,
            checks=checks,
            validated_defect_count=len(resolved_defects),
            expected_segment_count=len(resolved_segments),
            validation_method="deterministic_rule_based_validator"
        )


# ---------------------------------------------------------------------------
# Global Singleton & Public Service Function
# ---------------------------------------------------------------------------

_validator_instance: Optional[MultiDefectValidator] = None


def get_multi_defect_validator() -> MultiDefectValidator:
    """Returns singleton MultiDefectValidator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = MultiDefectValidator()
    return _validator_instance


def validate_multi_defect_result(
    classification_result: Optional[Union[MultiDefectClassificationResult, ClassificationResult]] = None,
    segmentation_result: Optional[DefectSegmentationResult] = None,
    multi_defect_assessment: Optional[MultiDefectAssessment] = None,
    original_text: Optional[str] = None,
    defects: Optional[List[PerDefectClassification]] = None,
    segments: Optional[List[DefectSegment]] = None
) -> MultiDefectValidationResult:
    """Public helper to validate consistency of multi-defect classification results."""
    validator = get_multi_defect_validator()
    return validator.validate(
        classification_result=classification_result,
        segmentation_result=segmentation_result,
        multi_defect_assessment=multi_defect_assessment,
        original_text=original_text,
        defects=defects,
        segments=segments
    )
