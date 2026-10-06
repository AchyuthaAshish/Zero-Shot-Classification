"""Classification Router for Industrial Defect Intelligence FastAPI Backend.

Provides:
- POST /api/v1/classify: Versioned classification endpoint supporting single-defect
  and co-occurring multi-defect descriptions across Local ML, Gemini, and Hybrid modes.
- Strict Pydantic input validation (rejecting empty/whitespace and excessively long inputs).
- Safe error mapping without disclosure of internal secrets or stack traces.
- Zero persistence / zero database mutation (reserved exclusively for Phase 3 Step 3.3).
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from fastapi import APIRouter, Depends, HTTPException, status

from api.config import APISettings, get_api_settings
from api.dependencies import get_classifier_service
from api.errors import ProviderError, APIError
from api.schemas.classification import (
    ClassificationRequest,
    ClassificationResponse,
    ClassificationData,
    ConfidenceAssessmentSchema,
    AmbiguityAssessmentSchema,
    PerDefectClassificationSchema,
    ValidationIssueSchema,
    MultiDefectValidationSchema,
)
from api.schemas.errors import ErrorResponse
from core.schemas import (
    ConfidenceAssessment,
    AmbiguityAssessment,
    MultiDefectValidationResult,
    PerDefectClassification,
)
from config.settings import get_settings

logger = logging.getLogger("api.routes.classification")

router = APIRouter(tags=["Classification"])


def _resolve_provider_model(model_source: Optional[str], requested_mode: str) -> Tuple[str, str]:
    """Resolves sanitized provider name and model identifier without leaking sensitive data."""
    app_settings = get_settings()
    source_raw = (model_source or "local_ml").lower()

    if requested_mode == "local" or "local" in source_raw:
        return "local", "Local ML (TF-IDF / MiniLM)"
    elif "aimlapi" in source_raw:
        return "aimlapi", getattr(app_settings, "llm_model", "glm-5-turbo")
    elif "gemini" in source_raw:
        return "gemini", getattr(app_settings, "llm_model", "gemini-3.8-flash")
    else:
        return (
            getattr(app_settings, "llm_provider", "local"),
            getattr(app_settings, "llm_model", "Local ML (TF-IDF / MiniLM)")
        )


def _map_confidence_assessment(ca: Optional[ConfidenceAssessment]) -> Optional[ConfidenceAssessmentSchema]:
    """Maps domain ConfidenceAssessment to API schema preserving calibration and qualitative tiers."""
    if ca is None:
        return None
    return ConfidenceAssessmentSchema(
        level=ca.level,
        approximate_range=ca.approximate_range,
        raw_score=ca.raw_score,
        calibrated_prob=ca.calibrated_prob,
        top2_margin=ca.top2_margin,
        is_calibrated=ca.is_calibrated,
        calibration_method=ca.calibration_method,
        is_ambiguous=ca.is_ambiguous
    )


def _map_ambiguity_assessment(aa: Optional[AmbiguityAssessment]) -> Optional[AmbiguityAssessmentSchema]:
    """Maps domain AmbiguityAssessment to API schema."""
    if aa is None:
        return None
    return AmbiguityAssessmentSchema(
        is_ambiguous=aa.is_ambiguous,
        reason=aa.reason,
        top_category=aa.top_category,
        competing_category=aa.competing_category,
        margin=aa.margin,
        evidence_summary=aa.evidence_summary,
        method=aa.method
    )


def _map_validation_result(vr: Optional[MultiDefectValidationResult]) -> Optional[MultiDefectValidationSchema]:
    """Maps domain MultiDefectValidationResult to API schema."""
    if vr is None:
        return None
    return MultiDefectValidationSchema(
        is_valid=vr.is_valid,
        status=vr.status,
        errors=[
            ValidationIssueSchema(
                code=e.code,
                severity=e.severity,
                message=e.message,
                defect_id=e.defect_id,
                segment_id=e.segment_id
            )
            for e in vr.errors
        ],
        warnings=[
            ValidationIssueSchema(
                code=w.code,
                severity=w.severity,
                message=w.message,
                defect_id=w.defect_id,
                segment_id=w.segment_id
            )
            for w in vr.warnings
        ],
        checks=dict(vr.checks) if vr.checks else {},
        validated_defect_count=vr.validated_defect_count,
        expected_segment_count=vr.expected_segment_count,
        validation_method=vr.validation_method
    )


def _map_per_defect(d: PerDefectClassification, fallback_mode: str) -> PerDefectClassificationSchema:
    """Maps domain PerDefectClassification to API schema."""
    provider = d.provider
    model = d.model
    if not provider or not model or provider == "local_ml":
        resolved_p, resolved_m = _resolve_provider_model(d.classification_mode, fallback_mode)
        provider = resolved_p if (not provider or provider == "local_ml") else provider
        model = resolved_m if (not model or model == "local_ml") else model

    clf_mode = d.classification_mode
    if clf_mode == "local_ml":
        clf_mode = "local"

    return PerDefectClassificationSchema(
        defect_id=d.defect_id,
        segment_id=d.segment_id,
        text=d.text,
        category=d.category,
        confidence_assessment=_map_confidence_assessment(d.confidence_assessment),
        reliability=d.reliability,
        explanation=d.explanation,
        classification_mode=clf_mode,
        provider=provider,
        model=model,
        ambiguity_assessment=_map_ambiguity_assessment(d.ambiguity_assessment),
        source_start_char=d.source_start_char,
        source_end_char=d.source_end_char,
        status=d.status,
        raw_score=d.raw_score,
        calibrated_prob=d.calibrated_prob,
        top2_margin=d.top2_margin
    )


@router.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Classify Industrial Defect",
    description=(
        "Classifies an industrial machine or equipment defect description into approved taxonomy categories.\n\n"
        "**Key Features:**\n"
        "- **Modes**: `local` (free offline ML), `gemini` (cloud zero-shot), or `hybrid` (local-first with fallback).\n"
        "- **Multi-Defect**: Automatically detects co-occurring defects and partitions them into isolated per-defect segments.\n"
        "- **Multilingual**: Preserves English, native Telugu script, and Telugu-English code-switched inputs.\n"
        "- **Confidence**: Provides calibrated statistical probabilities for local ML and qualitative tiers for LLM.\n"
        "- **Stateless**: Does not persist results to Supabase (persistence handled in History API)."
    ),
    responses={
        200: {
            "description": "Classification completed successfully.",
            "content": {
                "application/json": {
                    "examples": {
                        "single_defect": {
                            "summary": "Single Defect Example",
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
                                        "margin": 0.6521
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
                        "multi_defect": {
                            "summary": "Multi-Defect Example",
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
                                                "raw_score": 0.94,
                                                "calibrated_prob": 0.94,
                                                "is_calibrated": True
                                            },
                                            "reliability": "High",
                                            "explanation": "Mechanical grinding sound in conveyor motor.",
                                            "classification_mode": "local",
                                            "source_start_char": 0,
                                            "source_end_char": 45,
                                            "status": "success"
                                        },
                                        {
                                            "defect_id": 2,
                                            "segment_id": 2,
                                            "text": "the temperature sensor gives incorrect readings",
                                            "category": "Sensor Fault",
                                            "confidence_assessment": {
                                                "level": "High",
                                                "approximate_range": "~91%",
                                                "raw_score": 0.91,
                                                "calibrated_prob": 0.91,
                                                "is_calibrated": True
                                            },
                                            "reliability": "High",
                                            "explanation": "Incorrect sensor readings.",
                                            "classification_mode": "local",
                                            "source_start_char": 50,
                                            "source_end_char": 97,
                                            "status": "success"
                                        }
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        },
        422: {
            "model": ErrorResponse,
            "description": "Validation error (missing/whitespace description, excessive length, or invalid mode).",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Description cannot be empty or contain only whitespace.",
                            "details": {
                                "fields": [
                                    {
                                        "field": "body -> description",
                                        "message": "Description cannot be empty or contain only whitespace.",
                                        "type": "value_error"
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal server error without sensitive disclosures.",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": {
                            "code": "INTERNAL_SERVER_ERROR",
                            "message": "An unexpected error occurred during defect classification.",
                            "details": {}
                        }
                    }
                }
            }
        },
        502: {
            "model": ErrorResponse,
            "description": "External classification model service unavailable.",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": {
                            "code": "PROVIDER_ERROR",
                            "message": "External classification service is currently unavailable. Please try again later or switch to local mode.",
                            "details": {"provider": "gemini"}
                        }
                    }
                }
            }
        },
        503: {
            "model": ErrorResponse,
            "description": "External classification provider configuration error.",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": {
                            "code": "PROVIDER_ERROR",
                            "message": "Classification provider is not properly configured. Please check provider settings or switch to local mode.",
                            "details": {"provider": "gemini"}
                        }
                    }
                }
            }
        }
    }
)
def classify_defect_endpoint(
    request: ClassificationRequest,
    classifier=Depends(get_classifier_service),
    settings: APISettings = Depends(get_api_settings)
) -> ClassificationResponse:
    """Classifies an industrial defect description, returning a single or multi-defect response."""
    raw_desc = request.description
    mode_str = request.mode.value

    # Execute classification via existing domain pipeline (zero duplication of ML/LLM logic)
    try:
        result = classifier.classify(raw_defect_text=raw_desc, mode_override=mode_str)
    except Exception as exc:
        logger.error(f"Classifier invocation failed unexpectedly: {exc}", exc_info=True)
        raise APIError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred during defect classification."
        )

    # Sanitize classifier error statuses without revealing stack traces, keys, or endpoints
    if result.status == "configuration_error":
        raise ProviderError(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            message="Classification provider is not properly configured. Please check provider settings or switch to local mode.",
            details={"provider": mode_str}
        )
    if result.status == "model_error":
        raise ProviderError(
            status_code=status.HTTP_502_BAD_GATEWAY,
            message="External classification service is currently unavailable. Please try again later or switch to local mode.",
            details={"provider": mode_str}
        )
    if result.status == "system_error":
        raise APIError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="INTERNAL_SERVER_ERROR",
            message="A system error occurred while processing the classification."
        )

    # Inspect multi-defect assessment and per-defect classifications
    md_res = getattr(result, "multi_defect_result", None)
    is_multi = False
    if md_res is not None and getattr(md_res, "is_multi_defect", False) is True:
        try:
            cnt = getattr(md_res, "defect_count", 0)
            is_multi = int(cnt) >= 2
        except (ValueError, TypeError):
            is_multi = False

    # Resolve provider and model labels
    provider, model = _resolve_provider_model(result.model_source, mode_str)

    if is_multi:
        # Multi-Defect Branch: Expose structured list of defect segments
        defects_list = [
            _map_per_defect(d, fallback_mode=mode_str)
            for d in md_res.defects
        ]
        val_schema = _map_validation_result(getattr(md_res, "validation_result", None) or getattr(result, "validation_result", None))

        data = ClassificationData(
            original_text=raw_desc,
            is_multi_defect=True,
            defect_count=len(defects_list),
            overall_status=md_res.overall_status,
            category=None,
            status=md_res.overall_status,
            reliability="High",
            explanation=f"Identified {len(defects_list)} co-occurring defect signals partitioned into independent segments.",
            classification_mode=mode_str,
            provider=provider,
            model=model,
            confidence_assessment=None,
            ambiguity_assessment=_map_ambiguity_assessment(result.ambiguity_assessment),
            method=md_res.method or "per_segment_independent_classification",
            validation=val_schema,
            defects=defects_list
        )
    elif result.category == "Unknown":
        # Unknown / Insufficient Evidence Branch (defect_count = 0)
        ca_schema = _map_confidence_assessment(result.confidence_assessment)
        aa_schema = _map_ambiguity_assessment(result.ambiguity_assessment)
        val_schema = _map_validation_result(getattr(result, "validation_result", None))

        raw_score = ca_schema.raw_score if ca_schema else None
        cal_score = ca_schema.calibrated_prob if ca_schema else None
        conf = cal_score if cal_score is not None else raw_score
        conf_level = ca_schema.level if ca_schema else result.reliability
        conf_range = ca_schema.approximate_range if ca_schema else None
        top2_margin = ca_schema.top2_margin if ca_schema else None

        data = ClassificationData(
            original_text=raw_desc,
            is_multi_defect=False,
            defect_count=0,
            overall_status=result.status,
            category="Unknown",
            status=result.status,
            reliability=result.reliability,
            explanation=result.reason,
            classification_mode=mode_str,
            provider=provider,
            model=model,
            confidence=conf,
            confidence_level=conf_level,
            confidence_range=conf_range,
            raw_score=raw_score,
            calibrated_score=cal_score,
            top2_margin=top2_margin,
            confidence_assessment=ca_schema,
            ambiguity_assessment=aa_schema,
            method="unknown_evidence_safeguard",
            validation=val_schema,
            defects=[]
        )
    else:
        # Single-Defect Branch (defect_count = 1)
        ca_schema = _map_confidence_assessment(result.confidence_assessment)
        aa_schema = _map_ambiguity_assessment(result.ambiguity_assessment)
        val_schema = _map_validation_result(getattr(result, "validation_result", None))

        raw_score = ca_schema.raw_score if ca_schema else None
        cal_score = ca_schema.calibrated_prob if ca_schema else None
        conf = cal_score if cal_score is not None else raw_score
        conf_level = ca_schema.level if ca_schema else result.reliability
        conf_range = ca_schema.approximate_range if ca_schema else None
        top2_margin = ca_schema.top2_margin if ca_schema else None

        # Check if single defect is encapsulated in md_res.defects
        single_defects = []
        if md_res and md_res.defects:
            single_defects = [_map_per_defect(d, fallback_mode=mode_str) for d in md_res.defects]
        else:
            single_defects = [
                PerDefectClassificationSchema(
                    defect_id=1,
                    segment_id=1,
                    text=raw_desc.strip(),
                    category=result.category,
                    confidence_assessment=ca_schema,
                    reliability=result.reliability,
                    explanation=result.reason,
                    classification_mode=mode_str,
                    provider=provider,
                    model=model,
                    ambiguity_assessment=aa_schema,
                    source_start_char=0,
                    source_end_char=len(raw_desc.strip()),
                    status=result.status,
                    raw_score=raw_score,
                    calibrated_prob=cal_score,
                    top2_margin=top2_margin
                )
            ]

        data = ClassificationData(
            original_text=raw_desc,
            is_multi_defect=False,
            defect_count=1,
            overall_status=result.status,
            category=result.category,
            status=result.status,
            reliability=result.reliability,
            explanation=result.reason,
            classification_mode=mode_str,
            provider=provider,
            model=model,
            confidence=conf,
            confidence_level=conf_level,
            confidence_range=conf_range,
            raw_score=raw_score,
            calibrated_score=cal_score,
            top2_margin=top2_margin,
            confidence_assessment=ca_schema,
            ambiguity_assessment=aa_schema,
            method="single_defect_classification",
            validation=val_schema,
            defects=single_defects
        )

    return ClassificationResponse(success=True, data=data)
