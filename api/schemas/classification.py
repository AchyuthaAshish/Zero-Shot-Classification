"""Pydantic schemas for Defect Classification API (POST /api/v1/classify).

Conforms to Phase 3 Step 3.2:
- Versioned, strict request validation for description and classification mode
- Rejection of empty, whitespace-only, and excessively long inputs
- Full Unicode and multilingual preservation (English, Telugu script, Telugu-English)
- Standardized response envelope preserving single-defect and multi-defect structures
- Complete fidelity to existing ConfidenceAssessment and AmbiguityAssessment models
- Zero leakage of API keys, environment credentials, or server internals
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


class ClassificationMode(str, Enum):
    """Supported classification orchestration modes.
    
    - local: Offline deterministic ML inference (Zero cloud/LLM API calls)
    - gemini: Cloud zero-shot LLM inference
    - hybrid: Local-first inference with cloud fallback for low-confidence/uncertain cases
    - external: Alias for cloud zero-shot LLM inference
    """
    LOCAL = "local"
    GEMINI = "gemini"
    HYBRID = "hybrid"
    EXTERNAL = "external"


class ClassificationRequest(BaseModel):
    """Request payload for POST /api/v1/classify."""
    description: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Industrial machine/equipment defect description to classify. "
                    "Supports English, native Telugu script, and Telugu-English code-switched text. "
                    "Must be non-empty and non-whitespace, with a maximum length of 2000 characters.",
        examples=["Motor is making a grinding noise."]
    )
    mode: ClassificationMode = Field(
        default=ClassificationMode.HYBRID,
        description="Classification orchestration mode: 'local' (offline ML), 'gemini' (cloud zero-shot), "
                    "'hybrid' (local-first with fallback), or 'external' (cloud alias).",
        examples=["local"]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "description": "Motor is making a grinding noise.",
                "mode": "local"
            }
        }
    )

    @field_validator("description", mode="before")
    @classmethod
    def validate_description(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Description must be a string.")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Description cannot be empty or contain only whitespace.")
        if len(v) > 2000:
            raise ValueError(f"Description length ({len(v)} characters) exceeds maximum allowed limit of 2000 characters.")
        return v


class ConfidenceAssessmentSchema(BaseModel):
    """Structured confidence evaluation preserving calibration and qualitative tiers."""
    level: str = Field(description="Confidence level: High, Medium, Low, or Uncertain")
    approximate_range: str = Field(description="Qualitative or calibrated statistical range description")
    raw_score: Optional[float] = Field(default=None, description="Raw uncalibrated model score if available")
    calibrated_prob: Optional[float] = Field(default=None, description="Calibrated statistical probability if available")
    top2_margin: Optional[float] = Field(default=None, description="Separation margin between top-1 and top-2 class probabilities")
    is_calibrated: bool = Field(default=False, description="Whether statistical probability calibration was applied")
    calibration_method: Optional[str] = Field(default=None, description="Applied calibration technique (e.g. Temperature Scaling)")
    is_ambiguous: bool = Field(default=False, description="Whether low confidence or narrow margin indicates ambiguity")


class AmbiguityAssessmentSchema(BaseModel):
    """Structured ambiguity assessment distinguishing competing categories from insufficient evidence."""
    is_ambiguous: bool = Field(description="Whether input exhibits competing defect signals")
    reason: Optional[str] = Field(default=None, description="Ambiguity explanation or rationale")
    top_category: Optional[str] = Field(default=None, description="Primary candidate category")
    competing_category: Optional[str] = Field(default=None, description="Secondary competing category")
    margin: Optional[float] = Field(default=None, description="Separation margin between competing hypotheses")
    evidence_summary: Optional[str] = Field(default=None, description="Summary of competing signals")
    method: Optional[str] = Field(default=None, description="Ambiguity evaluation method")


class PerDefectClassificationSchema(BaseModel):
    """Isolated classification result for an individual defect segment in a report."""
    defect_id: int = Field(description="1-based defect index within the report")
    segment_id: int = Field(description="Associated segment index from segmentation")
    text: str = Field(description="Isolated defect segment text")
    category: str = Field(description="Classified canonical taxonomy category")
    confidence_assessment: ConfidenceAssessmentSchema = Field(description="Segment-specific structured confidence assessment")
    reliability: str = Field(description="Reliability tier: High, Medium, or Low")
    explanation: str = Field(description="Isolated natural language explanation for this defect")
    classification_mode: str = Field(description="Mode used to classify this segment")
    provider: Optional[str] = Field(default=None, description="Inference provider used")
    model: Optional[str] = Field(default=None, description="Inference model used")
    ambiguity_assessment: Optional[AmbiguityAssessmentSchema] = Field(default=None, description="Segment ambiguity assessment")
    source_start_char: int = Field(default=0, description="Start character offset in original report")
    source_end_char: int = Field(default=0, description="End character offset in original report")
    status: str = Field(default="success", description="Classification status: success, unknown, low_confidence, error")
    raw_score: Optional[float] = Field(default=None, description="Raw uncalibrated score")
    calibrated_prob: Optional[float] = Field(default=None, description="Calibrated statistical probability")
    top2_margin: Optional[float] = Field(default=None, description="Top-2 class separation margin")


class ValidationIssueSchema(BaseModel):
    """Structural validation issue or warning for multi-defect assessments."""
    code: str = Field(description="Validation error or warning code")
    severity: str = Field(description="Issue severity: error or warning")
    message: str = Field(description="Issue description")
    defect_id: Optional[int] = Field(default=None, description="Associated defect index")
    segment_id: Optional[int] = Field(default=None, description="Associated segment index")


class MultiDefectValidationSchema(BaseModel):
    """Structural consistency validation report across segmentation and per-defect classification."""
    is_valid: bool = Field(description="Whether the multi-defect classification passed structural consistency checks")
    status: str = Field(description="Validation status: VALID, PARTIAL, INVALID, or UNKNOWN")
    errors: List[ValidationIssueSchema] = Field(default_factory=list, description="Validation errors detected")
    warnings: List[ValidationIssueSchema] = Field(default_factory=list, description="Validation warnings detected")
    checks: Dict[str, bool] = Field(default_factory=dict, description="Detailed dictionary of individual verification checks")
    validated_defect_count: int = Field(default=0, description="Count of successfully validated defects")
    expected_segment_count: int = Field(default=0, description="Count of expected segments")
    validation_method: str = Field(default="deterministic_rule_based_validator", description="Validation methodology")


class ClassificationData(BaseModel):
    """Unified classification payload supporting both single-defect and multi-defect results."""
    original_text: str = Field(description="Original input defect description")
    is_multi_defect: bool = Field(description="True if the input contains 2 or more co-occurring defects; False otherwise")
    defect_count: int = Field(description="Total count of defects identified (0 for Unknown, 1 for single defect, >=2 for multi-defect)")
    overall_status: str = Field(description="Overall status: success, unknown, low_confidence, partial_error, error")

    # Single-defect fields (populated for single-defect and unknown results)
    category: Optional[str] = Field(default=None, description="Canonical taxonomy category for single defect or Unknown")
    status: Optional[str] = Field(default=None, description="Classification status for single defect")
    reliability: Optional[str] = Field(default=None, description="Reliability tier (High, Medium, Low)")
    explanation: Optional[str] = Field(default=None, description="Natural language explanation of classification")
    classification_mode: Optional[str] = Field(default=None, description="Classification mode applied (local, gemini, hybrid)")
    provider: Optional[str] = Field(default=None, description="Inference provider (e.g. local, gemini, aimlapi)")
    model: Optional[str] = Field(default=None, description="Inference model identifier")
    confidence: Optional[float] = Field(default=None, description="Overall classification confidence score (calibrated probability [0.0 - 1.0] for local ML; None for multi-defect)")
    confidence_level: Optional[str] = Field(default=None, description="Confidence level tier: High, Medium, Low, or Uncertain")
    confidence_range: Optional[str] = Field(default=None, description="Qualitative or calibrated statistical range description (e.g. '~94%')")
    raw_score: Optional[float] = Field(default=None, description="Raw uncalibrated model decision score if available")
    calibrated_score: Optional[float] = Field(default=None, description="Calibrated statistical probability [0.0 - 1.0] if available")
    top2_margin: Optional[float] = Field(default=None, description="Separation margin between top-1 and top-2 class probabilities")
    confidence_assessment: Optional[ConfidenceAssessmentSchema] = Field(default=None, description="Structured confidence assessment")
    ambiguity_assessment: Optional[AmbiguityAssessmentSchema] = Field(default=None, description="Structured ambiguity assessment")

    # Multi-defect fields
    method: Optional[str] = Field(default=None, description="Multi-defect classification methodology applied")
    validation: Optional[MultiDefectValidationSchema] = Field(default=None, description="Multi-defect structural validation result")
    defects: List[PerDefectClassificationSchema] = Field(default_factory=list, description="List of per-defect classification objects")


class ClassificationResponse(BaseModel):
    """Standardized top-level API response envelope."""
    success: bool = Field(default=True, description="Indicates whether classification completed successfully")
    data: ClassificationData = Field(description="Structured classification data payload")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "Single-Defect Classification",
                    "description": "Representative response for a single mechanical defect report classified via local ML.",
                    "value": {
                        "success": True,
                        "data": {
                            "original_text": "Motor is making a grinding noise.",
                            "is_multi_defect": False,
                            "defect_count": 1,
                            "overall_status": "success",
                            "category": "Mechanical Fault",
                            "status": "success",
                            "reliability": "High",
                            "explanation": "Detected acoustic grinding symptoms indicating mechanical component failure.",
                            "classification_mode": "local",
                            "provider": "local",
                            "model": "Local ML (TF-IDF / MiniLM)",
                            "confidence": 0.9412,
                            "confidence_level": "High",
                            "confidence_range": "~94%",
                            "raw_score": 0.9412,
                            "calibrated_score": 0.9412,
                            "top2_margin": 0.6521,
                            "confidence_assessment": {
                                "level": "High",
                                "approximate_range": "~94%",
                                "raw_score": 0.9412,
                                "calibrated_prob": 0.9412,
                                "top2_margin": 0.6521,
                                "is_calibrated": True,
                                "calibration_method": "Temperature Scaling",
                                "is_ambiguous": False
                            },
                            "ambiguity_assessment": {
                                "is_ambiguous": False,
                                "reason": "Single uncontested mechanical fault.",
                                "top_category": "Mechanical Fault",
                                "competing_category": None,
                                "margin": 0.6521,
                                "evidence_summary": None,
                                "method": None
                            },
                            "method": "single_defect_classification",
                            "validation": {
                                "is_valid": True,
                                "status": "VALID",
                                "errors": [],
                                "warnings": [],
                                "checks": {"single_signal_verified": True},
                                "validated_defect_count": 1,
                                "expected_segment_count": 1,
                                "validation_method": "deterministic_rule_based_validator"
                            },
                            "defects": [
                                {
                                    "defect_id": 1,
                                    "segment_id": 1,
                                    "text": "Motor is making a grinding noise.",
                                    "category": "Mechanical Fault",
                                    "confidence_assessment": {
                                        "level": "High",
                                        "approximate_range": "~94%",
                                        "raw_score": 0.9412,
                                        "calibrated_prob": 0.9412,
                                        "top2_margin": 0.6521,
                                        "is_calibrated": True,
                                        "calibration_method": "Temperature Scaling",
                                        "is_ambiguous": False
                                    },
                                    "reliability": "High",
                                    "explanation": "Detected acoustic grinding symptoms.",
                                    "classification_mode": "local",
                                    "provider": "local",
                                    "model": "Local ML (TF-IDF / MiniLM)",
                                    "source_start_char": 0,
                                    "source_end_char": 33,
                                    "status": "success",
                                    "raw_score": 0.9412,
                                    "calibrated_prob": 0.9412,
                                    "top2_margin": 0.6521
                                }
                            ]
                        }
                    }
                },
                {
                    "summary": "Multi-Defect Classification",
                    "description": "Representative response for a compound description partitioned into 2 independent defect segments.",
                    "value": {
                        "success": True,
                        "data": {
                            "original_text": "The conveyor motor is making a grinding noise and the temperature sensor gives incorrect readings.",
                            "is_multi_defect": True,
                            "defect_count": 2,
                            "overall_status": "success",
                            "category": None,
                            "status": "success",
                            "reliability": "High",
                            "explanation": "Identified 2 co-occurring defect signals partitioned into independent segments.",
                            "classification_mode": "local",
                            "provider": "local",
                            "model": "Local ML (TF-IDF / MiniLM)",
                            "confidence": None,
                            "confidence_level": None,
                            "confidence_range": None,
                            "raw_score": None,
                            "calibrated_score": None,
                            "top2_margin": None,
                            "confidence_assessment": None,
                            "ambiguity_assessment": None,
                            "method": "per_segment_independent_classification",
                            "validation": {
                                "is_valid": True,
                                "status": "VALID",
                                "errors": [],
                                "warnings": [],
                                "checks": {"segment_count_match": True, "no_span_overlap": True},
                                "validated_defect_count": 2,
                                "expected_segment_count": 2,
                                "validation_method": "deterministic_rule_based_validator"
                            },
                            "defects": [
                                {
                                    "defect_id": 1,
                                    "segment_id": 1,
                                    "text": "The conveyor motor is making a grinding noise",
                                    "category": "Mechanical Fault",
                                    "confidence_assessment": {
                                        "level": "High",
                                        "approximate_range": "~94%",
                                        "raw_score": 0.9412,
                                        "calibrated_prob": 0.9412,
                                        "top2_margin": 0.6521,
                                        "is_calibrated": True,
                                        "calibration_method": "Temperature Scaling",
                                        "is_ambiguous": False
                                    },
                                    "reliability": "High",
                                    "explanation": "Mechanical grinding sound in conveyor motor.",
                                    "classification_mode": "local",
                                    "provider": "local",
                                    "model": "Local ML (TF-IDF / MiniLM)",
                                    "source_start_char": 0,
                                    "source_end_char": 45,
                                    "status": "success",
                                    "raw_score": 0.9412,
                                    "calibrated_prob": 0.9412,
                                    "top2_margin": 0.6521
                                },
                                {
                                    "defect_id": 2,
                                    "segment_id": 2,
                                    "text": "the temperature sensor gives incorrect readings",
                                    "category": "Sensor Fault",
                                    "confidence_assessment": {
                                        "level": "High",
                                        "approximate_range": "~91%",
                                        "raw_score": 0.9125,
                                        "calibrated_prob": 0.9125,
                                        "top2_margin": 0.5843,
                                        "is_calibrated": True,
                                        "calibration_method": "Temperature Scaling",
                                        "is_ambiguous": False
                                    },
                                    "reliability": "High",
                                    "explanation": "Incorrect sensor readings.",
                                    "classification_mode": "local",
                                    "provider": "local",
                                    "model": "Local ML (TF-IDF / MiniLM)",
                                    "source_start_char": 50,
                                    "source_end_char": 97,
                                    "status": "success",
                                    "raw_score": 0.9125,
                                    "calibrated_prob": 0.9125,
                                    "top2_margin": 0.5843
                                }
                            ]
                        }
                    }
                },
                {
                    "summary": "Unknown Classification Safeguard",
                    "description": "Representative response when text exhibits insufficient defect evidence to classify safely.",
                    "value": {
                        "success": True,
                        "data": {
                            "original_text": "Equipment encountered an unexpected anomaly during operation.",
                            "is_multi_defect": False,
                            "defect_count": 0,
                            "overall_status": "unknown",
                            "category": "Unknown",
                            "status": "unknown",
                            "reliability": "Low",
                            "explanation": "Insufficient evidence or ambiguous symptom description unable to map to approved taxonomy.",
                            "classification_mode": "local",
                            "provider": "local",
                            "model": "Local ML (TF-IDF / MiniLM)",
                            "confidence": 0.18,
                            "confidence_level": "Uncertain",
                            "confidence_range": "<50%",
                            "raw_score": 0.18,
                            "calibrated_score": 0.18,
                            "top2_margin": 0.05,
                            "confidence_assessment": {
                                "level": "Uncertain",
                                "approximate_range": "<50%",
                                "raw_score": 0.18,
                                "calibrated_prob": 0.18,
                                "top2_margin": 0.05,
                                "is_calibrated": True,
                                "calibration_method": "Temperature Scaling",
                                "is_ambiguous": True
                            },
                            "ambiguity_assessment": {
                                "is_ambiguous": True,
                                "reason": "Confidence below threshold; insufficient evidence.",
                                "top_category": "Unknown",
                                "competing_category": None,
                                "margin": 0.05,
                                "evidence_summary": None,
                                "method": "confidence_threshold_safeguard"
                            },
                            "method": "unknown_evidence_safeguard",
                            "validation": {
                                "is_valid": True,
                                "status": "VALID",
                                "errors": [],
                                "warnings": [],
                                "checks": {},
                                "validated_defect_count": 0,
                                "expected_segment_count": 0,
                                "validation_method": "deterministic_rule_based_validator"
                            },
                            "defects": []
                        }
                    }
                }
            ]
        }
    )
