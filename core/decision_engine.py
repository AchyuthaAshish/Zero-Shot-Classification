"""Decision Engine for Zero-Shot Industrial Defect Classification.

Conforms to SRS Section 3 (C9), SRS Section 4 (FR-008, FR-010),
SRS Section 5 (Decision Rules DR-001 to DR-006), and BR-002, BR-004, BR-010.

PROVISIONAL DETERMINISTIC RELIABILITY RULE (DOCUMENTATION):
------------------------------------------------------------
Per PRD Section 16/20 and SRS FR-010, the formal calibration and mathematical definition
of confidence/reliability is an unresolved human decision reserved for the project owner.
In the current implementation:
1. Reliability values ('High', 'Medium', 'Low') are assigned purely as a provisional,
   deterministic heuristic:
   - 'High': The classification is an approved, specific fault category (not 'Unknown')
     that passed strict taxonomy validation, supported by evidence, and uncontested
     (either confirmed by verification or accepted in single-pass mode without dispute).
   - 'Medium': The classification involves technical uncertainty or reconciliation:
     * A legitimate primary 'Unknown' outcome due to insufficient evidence in the description.
     * A proposal adjusted by secondary verification from the primary proposal to an
       alternative approved category.
   - 'Low': The classification encountered dispute, rejection, or failure:
     * Primary proposal challenged/rejected by secondary verification without an approved
       alternative (defaulted to 'Unknown').
     * Primary proposal rejected by strict taxonomy validation (unapproved/invented category).
     * Controlled model, API, or infrastructure failure.
2. These values do NOT represent calibrated statistical or Bayesian probabilities.
3. No numerical percentages or arbitrary probability thresholds are claimed or computed
   without explicit human approval.


CORE INVARIANTS:
1. Model/infrastructure failure is NEVER disguised as Unknown (BR-010).
2. Unknown is the legitimate outcome for vague or insufficient evidence (BR-004, DR-006).
3. The final category is strictly validated against the authoritative taxonomy.
4. When verification rejects a proposal with no valid alternative:
   category = "Unknown", status = "unknown", reliability = "Low".
"""

from typing import Optional, Dict, Any
from core.schemas import ClassificationResult, PreprocessedInput, LLMRawResponse, ConfidenceAssessment
from validation.category_validator import get_category_validator, CategoryValidator


class DecisionEngine:
    """Evaluates classification proposals, validation status, and verification feedback."""

    def __init__(self, validator: Optional[CategoryValidator] = None):
        self._validator = validator or get_category_validator()

    def decide_from_llm(
        self,
        processed_input: PreprocessedInput,
        llm_response: LLMRawResponse,
        verification_feedback: Optional[Dict[str, Any]] = None,
        _is_subsegment: bool = False,
        defect_classifier: Optional[Any] = None
    ) -> ClassificationResult:
        """
        Processes the LLM proposal, validates it, evaluates verification (if present),
        and returns a canonical ClassificationResult.
        """
        proposed_category = llm_response.proposed_category
        reason = llm_response.reason
        language = processed_input.detected_language

        # 1. Strict Taxonomy Validation against authoritative taxonomy
        is_valid, canonical_category, error_msg = self._validator.validate_category(proposed_category)

        if not is_valid:
            # Rejection of invented or non-taxonomy categories
            return ClassificationResult(
                category="Unknown",
                reason=f"LLM proposed an unapproved category ('{proposed_category}'). Strict validation rejected it: {error_msg}",
                language=language,
                reliability="Low",
                status="validation_error",
                original_description=processed_input.raw_text,
                normalized_description=processed_input.normalized_text,
                error_message=error_msg,
                confidence=None,
                confidence_assessment=ConfidenceAssessment(
                    level="Uncertain",
                    approximate_range="Insufficient Evidence",
                    raw_score=None,
                    calibrated_prob=None,
                    top2_margin=None,
                    is_calibrated=False,
                    calibration_method=None,
                    is_ambiguous=True
                )
            )

        # 2. Check for Unknown outcome from primary LLM
        if canonical_category == "Unknown":
            ca_unknown = ConfidenceAssessment(
                level="Uncertain",
                approximate_range="Insufficient Evidence",
                raw_score=None,
                calibrated_prob=None,
                top2_margin=None,
                is_calibrated=False,
                calibration_method=None,
                is_ambiguous=True
            )
            from core.ambiguity_detector import get_ambiguity_detector
            amb_assessment = get_ambiguity_detector().evaluate_ambiguity(
                raw_or_preprocessed=processed_input,
                candidate_category="Unknown",
                confidence=None,
                confidence_assessment=ca_unknown,
                model_source="gemini"
            )
            return ClassificationResult(
                category="Unknown",
                reason=reason or "The description does not contain sufficient technical evidence to assign a specific fault category.",
                language=language,
                reliability="Medium",
                status="unknown",
                original_description=processed_input.raw_text,
                normalized_description=processed_input.normalized_text,
                confidence=None,
                confidence_assessment=ca_unknown,
                ambiguity_assessment=amb_assessment
            )

        # 3. Verification evaluation (if enabled and provided)
        # Capture qualitative reliability from LLM proposal if valid
        reliability = "High"
        llm_reliability = getattr(llm_response, "reliability", None)
        if llm_reliability and str(llm_reliability).strip().capitalize() in ("High", "Medium", "Low"):
            reliability = str(llm_reliability).strip().capitalize()

        final_category = canonical_category
        status = "success"

        if verification_feedback:
            is_verified = verification_feedback.get("verified", True)
            alternative = verification_feedback.get("alternative_category")
            if not is_verified:
                if alternative and self._validator.is_valid_category(alternative):
                    final_category = alternative
                    reliability = "Medium"
                    status = "success"
                    reason = f"Primary proposal '{canonical_category}' adjusted to '{alternative}' by verification. {reason}"
                else:
                    # Verification challenged proposal and no valid approved alternative exists
                    final_category = "Unknown"
                    reliability = "Low"
                    status = "unknown"
                    reason = f"Verification challenged proposal '{canonical_category}' due to insufficient evidence. Defaulted to Unknown."

        if final_category == "Unknown":
            ca = ConfidenceAssessment(
                level="Uncertain",
                approximate_range="Insufficient Evidence",
                raw_score=None,
                calibrated_prob=None,
                top2_margin=None,
                is_calibrated=False,
                calibration_method=None,
                is_ambiguous=True
            )
        else:
            ca = ConfidenceAssessment(
                level=reliability,
                approximate_range="Qualitative / N/A",
                raw_score=None,
                calibrated_prob=None,
                top2_margin=None,
                is_calibrated=False,
                calibration_method=None,
                is_ambiguous=(reliability == "Low")
            )

        # 4. Unknown Detection Layer (Phase 1 Step 1.3)
        from core.unknown_detector import get_unknown_detector
        ud_result = get_unknown_detector().evaluate_unknown(
            raw_or_preprocessed=processed_input,
            candidate_category=final_category,
            confidence=None,
            confidence_assessment=ca,
            model_source="gemini"
        )

        if ud_result.is_unknown:
            final_category = "Unknown"
            if status != "validation_error":
                status = "unknown"
            if "Verification challenged" not in reason and canonical_category != "Unknown":
                reason = ud_result.reason
            ca = ud_result.confidence_assessment
            reliability = "Low" if status in ("validation_error", "model_error") else ("Low" if "Verification challenged" in reason else "Medium")
        # 5. Ambiguity Detection Layer (Phase 1 Step 1.4)
        from core.ambiguity_detector import get_ambiguity_detector
        ambiguity_assessment = get_ambiguity_detector().evaluate_ambiguity(
            raw_or_preprocessed=processed_input,
            candidate_category=final_category,
            confidence=None,
            confidence_assessment=ca,
            model_source="gemini",
            verification_feedback=verification_feedback
        )

        multi_defect_assessment = None
        segmentation_result = None
        multi_defect_classification = None

        if not _is_subsegment:
            # 6. Multi-Defect Detection Layer (Phase 2 Step 2.1)
            from core.multi_defect_detector import get_multi_defect_detector
            multi_defect_assessment = get_multi_defect_detector().detect_multi_defect(
                raw_or_preprocessed=processed_input,
                candidate_category=final_category
            )

            # 7. Defect Segmentation Layer (Phase 2 Step 2.2)
            from core.defect_segmenter import get_defect_segmenter
            segmentation_result = get_defect_segmenter().segment_defects(
                raw_or_preprocessed=processed_input,
                multi_defect_assessment=multi_defect_assessment
            )

            # 8. Per-Defect Classification Layer (Phase 2 Step 2.3)
            if segmentation_result.is_multi_defect and segmentation_result.segment_count >= 2:
                from core.per_defect_classifier import get_per_defect_classifier
                multi_defect_classification = get_per_defect_classifier().classify_segments(
                    segments=segmentation_result.segments,
                    original_text=processed_input.raw_text,
                    mode="gemini",
                    multi_defect_assessment=multi_defect_assessment,
                    segmentation_result=segmentation_result,
                    defect_classifier=defect_classifier
                )

        return ClassificationResult(
            category=final_category,
            reason=reason,
            language=language,
            reliability=reliability,
            status=status,
            original_description=processed_input.raw_text,
            normalized_description=processed_input.normalized_text,
            confidence=None,
            confidence_assessment=ca,
            ambiguity_assessment=ambiguity_assessment,
            multi_defect_assessment=multi_defect_assessment,
            segmentation_result=segmentation_result,
            multi_defect_classification=multi_defect_classification
        )

    def create_error_result(
        self,
        raw_text: str,
        error_message: str,
        error_type: str = "model_error"
    ) -> ClassificationResult:
        """
        Constructs a controlled error result.
        Crucial requirement (BR-010): Infrastructure or model errors MUST NOT be
        represented as successful 'Unknown' classifications.
        """
        from core.schemas import AmbiguityAssessment, MultiDefectAssessment, DefectSegmentationResult
        return ClassificationResult(
            category="Unknown",
            reason=f"Processing halted due to {error_type.upper()}: {error_message}",
            language="Unknown",
            reliability="Low",
            status=error_type,
            original_description=raw_text,
            normalized_description=raw_text.strip(),
            error_message=error_message,
            confidence=None,
            confidence_assessment=ConfidenceAssessment(
                level="Uncertain",
                approximate_range="Insufficient Evidence",
                raw_score=None,
                calibrated_prob=None,
                top2_margin=None,
                is_calibrated=False,
                calibration_method=None,
                is_ambiguous=True
            ),
            ambiguity_assessment=AmbiguityAssessment(
                is_ambiguous=False,
                reason=f"Processing halted due to error: {error_message}",
                method="error_not_ambiguous"
            ),
            multi_defect_assessment=MultiDefectAssessment(
                is_multi_defect=False,
                defect_signal_count=0,
                signals=[],
                detection_reason=f"Processing halted due to {error_type.upper()}: {error_message}",
                primary_candidate_category="Unknown"
            ),
            segmentation_result=DefectSegmentationResult(
                original_text=raw_text,
                is_multi_defect=False,
                segment_count=0,
                segments=[],
                segmentation_reason=f"Processing halted due to {error_type.upper()}: {error_message}"
            )
        )
