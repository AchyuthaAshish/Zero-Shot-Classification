"""Core data schemas and contracts for Zero-Shot Industrial Defect Classification.

Conforms to SRS Section 6 (Output Contract), Section 9 (Data Requirements),
FR-005, FR-006, FR-010, and FR-012.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import uuid


@dataclass(frozen=True)
class PreprocessedInput:
    """Preprocessed representation of user defect input."""
    raw_text: str
    normalized_text: str
    detected_language: str
    is_code_switched: bool = False
    char_count: int = 0
    word_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConfidenceAssessment:
    """Dedicated structured confidence assessment for model predictions.

    Distinguishes raw model scores, calibrated statistical probabilities,
    qualitative confidence levels, and ambiguity/margin information.
    """
    level: str  # High | Medium | Low | Uncertain
    approximate_range: str  # e.g. "Uncalibrated model score", "Qualitative / N/A", "Insufficient Evidence"
    raw_score: Optional[float] = None
    calibrated_prob: Optional[float] = None
    top2_margin: Optional[float] = None
    is_calibrated: bool = False
    calibration_method: Optional[str] = None
    is_ambiguous: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AmbiguityAssessment:
    """Dedicated structured ambiguity assessment for defect descriptions.

    Distinguishes:
    1. Insufficient evidence (Unknown, not ambiguous)
    2. Genuine competing evidence (ambiguous between specific categories)
    3. Clearly supported single-category evidence (unambiguous)
    """
    is_ambiguous: bool
    reason: Optional[str] = None
    top_category: Optional[str] = None
    competing_category: Optional[str] = None
    margin: Optional[float] = None
    evidence_summary: Optional[str] = None
    method: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DefectSignal:
    """Identified defect evidence group/signal within a defect report."""
    signal_id: int
    category: str
    text_span: str
    equipment: Optional[str] = None
    symptom: Optional[str] = None
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultiDefectAssessment:
    """Dedicated structured multi-defect assessment for defect descriptions.

    Conforms to Phase 2 Step 2.1:
    - Distinguishes single defect reports from co-occurring multi-defect reports.
    - Preserves original input text without destructive segmentation.
    - Differentiates co-occurring multi-defects from single-defect ambiguity.
    - Distinguishes independent co-occurring defects from causally connected single defects.
    """
    is_multi_defect: bool
    defect_signal_count: int
    signals: List[DefectSignal] = field(default_factory=list)
    detection_reason: str = ""
    method: str = "rule_based_evidence_partitioning"
    has_coordinating_conjunction: bool = False
    has_causal_relation: bool = False
    primary_candidate_category: Optional[str] = None

    @property
    def evidence_groups(self) -> List[DefectSignal]:
        """Alias for signals to support both naming conventions."""
        return self.signals

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DefectSegment:
    """A clean, isolated defect text span identified from a defect report.

    Conforms to Phase 2 Step 2.2:
    - Maintains exact character span boundaries (start_char, end_char) within original text.
    - Links to associated DefectSignal / evidence group.
    - Preserves equipment and symptom context for downstream per-defect classification.
    """
    segment_id: int
    text: str
    start_char: int
    end_char: int
    source_clause: Optional[str] = None
    evidence_group: Optional[DefectSignal] = None
    linked_equipment: Optional[str] = None
    segmentation_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DefectSegmentationResult:
    """Structured result of defect segmentation for an input description.

    Conforms to Phase 2 Step 2.2:
    - Exposes original text, multi-defect status, segment count, and extracted segments.
    - Extensible for Phase 2 Step 2.3 per-defect classification.
    """
    original_text: str
    is_multi_defect: bool
    segment_count: int
    segments: List[DefectSegment] = field(default_factory=list)
    method: str = "rule_based_evidence_span_extraction"
    segmentation_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerDefectClassification:
    """Independent classification result for an extracted defect segment.

    Conforms to Phase 2 Step 2.3:
    - Maintains defect_id, segment_id, and segment text.
    - Contains independent category, confidence assessment, reliability, and explanation.
    - Records classification mode, provider, model, and ambiguity assessment.
    - Preserves source start_char and end_char offsets from the original report text.
    """
    defect_id: int
    segment_id: int
    text: str
    category: str
    confidence_assessment: ConfidenceAssessment
    reliability: str
    explanation: str
    classification_mode: str
    provider: Optional[str] = None
    model: Optional[str] = None
    ambiguity_assessment: Optional[AmbiguityAssessment] = None
    source_start_char: int = 0
    source_end_char: int = 0
    status: str = "success"
    confidence: Optional[float] = None
    raw_score: Optional[float] = None
    calibrated_prob: Optional[float] = None
    top2_margin: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationIssue:
    """Structured validation error or warning for multi-defect outputs."""
    code: str
    severity: str  # "error" | "warning"
    message: str
    defect_id: Optional[int] = None
    segment_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultiDefectValidationResult:
    """Structured aggregate validation result for multi-defect classification.

    Conforms to Phase 2 Step 2.4:
    - Assesses internal structural consistency across detection, segmentation, and classification.
    - Exposes is_valid, status (VALID | PARTIAL | INVALID | UNKNOWN), errors, warnings, and checks.
    - Observational only: does not alter predictions or rewrite classifier outputs.
    """
    is_valid: bool
    status: str  # "VALID" | "PARTIAL" | "INVALID" | "UNKNOWN"
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    checks: Dict[str, bool] = field(default_factory=dict)
    validated_defect_count: int = 0
    expected_segment_count: int = 0
    validation_method: str = "deterministic_rule_based_validator"

    @property
    def error_codes(self) -> List[str]:
        return [e.code for e in self.errors]

    @property
    def warning_codes(self) -> List[str]:
        return [w.code for w in self.warnings]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultiDefectClassificationResult:
    """Structured aggregate multi-defect classification result.

    Conforms to Phase 2 Step 2.3 & Step 2.4:
    - Holds original text, multi-defect indicator, defect count, and list of PerDefectClassification.
    - Provides overall status, method, and references to segmentation, multi-defect assessments,
      and multi-defect validation results.
    """
    original_text: str
    is_multi_defect: bool
    defect_count: int
    defects: List[PerDefectClassification] = field(default_factory=list)
    overall_status: str = "success"  # success | unknown | partial_error | error
    method: str = "per_segment_independent_classification"
    segmentation_result: Optional[DefectSegmentationResult] = None
    multi_defect_assessment: Optional[MultiDefectAssessment] = None
    validation_result: Optional[MultiDefectValidationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMRawResponse:
    """Raw structured proposal from the LLM client."""
    proposed_category: str
    reason: str
    detected_language: Optional[str] = None
    raw_content: Optional[str] = None
    reliability: Optional[str] = None


@dataclass
class ClassificationResult:
    """Validated canonical classification result returned by the system."""
    category: str
    reason: str
    language: str
    reliability: str  # High | Medium | Low
    status: str       # success | unknown | model_error | validation_error | system_error
    original_description: str
    normalized_description: str
    error_message: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    model_source: Optional[str] = "gemini"
    confidence: Optional[float] = None
    confidence_assessment: Optional[ConfidenceAssessment] = None
    ambiguity_assessment: Optional[AmbiguityAssessment] = None
    multi_defect_assessment: Optional[MultiDefectAssessment] = None
    segmentation_result: Optional[DefectSegmentationResult] = None
    multi_defect_classification: Optional[MultiDefectClassificationResult] = None
    validation_result: Optional[MultiDefectValidationResult] = None

    @property
    def multi_defect_result(self) -> Optional[MultiDefectClassificationResult]:
        """Convenience alias for multi_defect_classification."""
        return self.multi_defect_classification

    @property
    def per_defect_classifications(self) -> List[PerDefectClassification]:
        """Convenience property accessing list of per-defect classifications."""
        if self.multi_defect_classification:
            return self.multi_defect_classification.defects
        return []

    def __post_init__(self):
        if self.confidence_assessment is None:
            self.confidence_assessment = self._create_default_assessment()
        if self.ambiguity_assessment is None:
            self.ambiguity_assessment = AmbiguityAssessment(
                is_ambiguous=bool(self.confidence_assessment and self.confidence_assessment.is_ambiguous),
                reason="Default ambiguity assessment",
                top_category=self.category if self.category != "Unknown" else None
            )
        if self.multi_defect_assessment is None:
            self.multi_defect_assessment = MultiDefectAssessment(
                is_multi_defect=False,
                defect_signal_count=1 if self.category != "Unknown" else 0,
                detection_reason="Default single defect assessment",
                primary_candidate_category=self.category if self.category != "Unknown" else None
            )
        if self.segmentation_result is None:
            clean_desc = self.original_description.strip()
            self.segmentation_result = DefectSegmentationResult(
                original_text=self.original_description,
                is_multi_defect=bool(self.multi_defect_assessment and self.multi_defect_assessment.is_multi_defect),
                segment_count=1 if self.category != "Unknown" and clean_desc else 0,
                segments=[
                    DefectSegment(
                        segment_id=1,
                        text=clean_desc,
                        start_char=0,
                        end_char=len(clean_desc),
                        segmentation_reason="Default single defect segment"
                    )
                ] if self.category != "Unknown" and clean_desc else [],
                segmentation_reason="Default single-defect segmentation"
            )
        if self.multi_defect_classification is None:
            seg_res = self.segmentation_result
            if seg_res and seg_res.segments and self.category != "Unknown":
                primary_seg = seg_res.segments[0]
                default_defect = PerDefectClassification(
                    defect_id=1,
                    segment_id=primary_seg.segment_id,
                    text=primary_seg.text,
                    category=self.category,
                    confidence_assessment=self.confidence_assessment,
                    reliability=self.reliability,
                    explanation=self.reason,
                    classification_mode=self.model_source or "unknown",
                    provider=self.model_source,
                    model=self.model_source,
                    ambiguity_assessment=self.ambiguity_assessment,
                    source_start_char=primary_seg.start_char,
                    source_end_char=primary_seg.end_char,
                    status=self.status,
                    confidence=self.confidence,
                    raw_score=self.confidence_assessment.raw_score if self.confidence_assessment else self.confidence,
                    calibrated_prob=self.confidence_assessment.calibrated_prob if self.confidence_assessment else None,
                    top2_margin=self.confidence_assessment.top2_margin if self.confidence_assessment else None
                )
                self.multi_defect_classification = MultiDefectClassificationResult(
                    original_text=self.original_description,
                    is_multi_defect=bool(self.multi_defect_assessment and self.multi_defect_assessment.is_multi_defect),
                    defect_count=1,
                    defects=[default_defect],
                    overall_status=self.status,
                    method="single_defect_default",
                    segmentation_result=seg_res,
                    multi_defect_assessment=self.multi_defect_assessment
                )
            else:
                self.multi_defect_classification = MultiDefectClassificationResult(
                    original_text=self.original_description,
                    is_multi_defect=bool(self.multi_defect_assessment and self.multi_defect_assessment.is_multi_defect),
                    defect_count=0,
                    defects=[],
                    overall_status=self.status if self.status in ("unknown", "error", "model_error", "system_error") else "unknown",
                    method="insufficient_evidence_default",
                    segmentation_result=seg_res,
                    multi_defect_assessment=self.multi_defect_assessment
                )
        if self.validation_result is None:
            if self.multi_defect_classification and self.multi_defect_classification.validation_result:
                self.validation_result = self.multi_defect_classification.validation_result
            else:
                try:
                    from core.multi_defect_validator import validate_multi_defect_result
                    self.validation_result = validate_multi_defect_result(
                        classification_result=self,
                        segmentation_result=self.segmentation_result,
                        multi_defect_assessment=self.multi_defect_assessment,
                        original_text=self.original_description
                    )
                    if self.multi_defect_classification and self.multi_defect_classification.validation_result is None:
                        self.multi_defect_classification.validation_result = self.validation_result
                except Exception:
                    self.validation_result = None

    def _create_default_assessment(self) -> ConfidenceAssessment:
        if self.category == "Unknown" or self.status.endswith("_error") or self.status in ("unknown", "error", "system_error"):
            return ConfidenceAssessment(
                level="Uncertain",
                approximate_range="Insufficient Evidence",
                raw_score=self.confidence,
                calibrated_prob=None,
                top2_margin=None,
                is_calibrated=False,
                calibration_method=None,
                is_ambiguous=True
            )

        is_local = bool(self.model_source and "local" in self.model_source.lower())
        if is_local or self.confidence is not None:
            return ConfidenceAssessment(
                level=self.reliability if self.reliability in ("High", "Medium", "Low") else "Medium",
                approximate_range="Uncalibrated model score",
                raw_score=self.confidence,
                calibrated_prob=None,
                top2_margin=None,
                is_calibrated=False,
                calibration_method=None,
                is_ambiguous=(self.reliability == "Low" or self.status == "low_confidence")
            )

        # External LLM / Gemini (qualitative)
        return ConfidenceAssessment(
            level=self.reliability if self.reliability in ("High", "Medium", "Low") else "High",
            approximate_range="Qualitative / N/A",
            raw_score=None,
            calibrated_prob=None,
            top2_margin=None,
            is_calibrated=False,
            calibration_method=None,
            is_ambiguous=(self.reliability == "Low")
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        """Clean, human-readable terminal presentation for manual testing and reporting."""
        is_error = self.status.endswith("_error") or self.status in ("system_error", "error")
        if is_error:
            raw_err = self.error_message or self.reason or "An error occurred during classification."
            raw_lower = raw_err.lower()
            if any(marker in raw_lower for marker in ["rate limit", "quota", "resource_exhausted", "429", "503", "unavailable", "busy", "exhausted"]):
                safe_err = "Gemini is temporarily busy or the request quota has been reached. Please wait a few minutes and try again."
            elif "api call failed" in raw_lower or "model_error" in raw_lower:
                safe_err = "Unable to classify the defect right now. Please try again later."
            elif self.status == "validation_error":
                safe_err = "Unable to produce a valid taxonomy category for this description."
            elif self.status == "configuration_error":
                safe_err = "Gemini API key is missing or invalid. Please check your local configuration."
            else:
                safe_err = raw_err

            return (
                "==================================================\n"
                "        INDUSTRIAL DEFECT CLASSIFICATION\n"
                "==================================================\n\n"
                "Status : ERROR\n\n"
                "Error:\n"
                f"{safe_err}\n\n"
                "=================================================="
            )

        status_display = self.status.upper()
        ca = self.confidence_assessment
        ca_display = ""
        if ca is not None:
            ca_display = (
                f"\nConfidence Level : {ca.level}"
                f"\nConfidence Range : {ca.approximate_range}"
            )
            if ca.raw_score is not None:
                ca_display += f"\nRaw Model Score  : {ca.raw_score:.4f}"
            if ca.calibrated_prob is not None:
                ca_display += f"\nCalibrated Prob  : {ca.calibrated_prob:.4f}"
            if ca.top2_margin is not None:
                ca_display += f"\nTop-2 Margin     : {ca.top2_margin:.4f}"
        elif self.confidence is not None:
            ca_display = f"\nConfidence       : {self.confidence:.2f}"

        source_display = f"\nModel Source     : {self.model_source}" if self.model_source else ""
        return (
            "==================================================\n"
            "        INDUSTRIAL DEFECT CLASSIFICATION\n"
            "==================================================\n\n"
            "Input:\n"
            f"{self.original_description}\n\n"
            "--------------------------------------------------\n"
            "Classification Result\n"
            "--------------------------------------------------\n\n"
            f"Category         : {self.category}\n"
            f"Language         : {self.language}\n"
            f"Reliability      : {self.reliability}\n"
            f"Status           : {status_display}"
            f"{ca_display}"
            f"{source_display}\n\n"
            "Reason:\n"
            f"{self.reason}\n\n"
            "=================================================="
        )


@dataclass
class ClassificationRecord:
    """Persisted record for classification history and statistics."""
    id: str
    original_description: str
    final_category: str
    explanation: str
    language: str
    reliability: str
    status: str
    created_at: str

    @classmethod
    def from_result(cls, result: ClassificationResult) -> "ClassificationRecord":
        return cls(
            id=result.id,
            original_description=result.original_description,
            final_category=result.category,
            explanation=result.reason,
            language=result.language,
            reliability=result.reliability,
            status=result.status,
            created_at=result.created_at
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
