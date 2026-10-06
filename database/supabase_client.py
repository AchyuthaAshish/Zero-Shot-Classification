"""Supabase Client and Database Operations for Defect Reports.

Provides secure, RLS-compliant persistence for industrial defect classification reports.
Enforces strict schema constraints, data validation, and sanitized error handling.
"""

import logging
from typing import Dict, List, Optional, Any
from config.settings import get_settings
from core.exceptions import ConfigurationError, PersistenceError
from taxonomy.repository import get_taxonomy_repository

logger = logging.getLogger("database.supabase_client")

_supabase_client = None


class DatabaseError(PersistenceError):
    """Raised when a Supabase/PostgreSQL database operation fails."""
    def __init__(self, message: str = "Database operation failed."):
        super().__init__(message)


class MultiDefectPersistenceError(PersistenceError, ValueError):
    """Raised when multi-defect persistence fails or validation is structurally INVALID."""
    def __init__(self, message: str = "Multi-defect persistence error.", code: str = "PERSISTENCE_ERROR"):
        super().__init__(message)
        self.message = message
        self.code = code


def is_supabase_configured() -> bool:
    """Returns True if both SUPABASE_URL and SUPABASE_KEY are configured."""
    settings = get_settings()
    url = getattr(settings, "supabase_url", None)
    key = getattr(settings, "supabase_key", None)
    return bool(url and url.strip() and key and key.strip())


def get_supabase_client(token: Optional[str] = None) -> Any:
    """Returns the singleton Supabase client, or an isolated client scoped to an authenticated JWT."""
    global _supabase_client
    settings = get_settings()
    url = getattr(settings, "supabase_url", None)
    key = getattr(settings, "supabase_key", None)

    if not url or not key:
        raise ConfigurationError(
            "Supabase configuration missing: SUPABASE_URL and SUPABASE_KEY must be set in your environment or .env file."
        )

    clean_url = url.strip().rstrip("/")
    if not clean_url.startswith(("http://", "https://")):
        raise ConfigurationError(
            "Invalid SUPABASE_URL: Must be a valid URL starting with https:// (e.g., https://<project-id>.supabase.co). "
            "Please check that an API key or token was not accidentally placed in SUPABASE_URL."
        )

    # Automatically strip accidental sub-paths such as /rest/v1 so supabase-py does not double-append it
    if clean_url.endswith("/rest/v1"):
        clean_url = clean_url[:-len("/rest/v1")].rstrip("/")

    try:
        from supabase import create_client
        if token and str(token).strip():
            scoped_client = create_client(clean_url, key.strip())
            scoped_client.postgrest.auth(str(token).strip())
            return scoped_client

        if _supabase_client is not None:
            return _supabase_client

        _supabase_client = create_client(clean_url, key.strip())
        return _supabase_client
    except Exception as e:
        safe_msg = _sanitize_error(str(e))
        raise DatabaseError(f"Failed to initialize Supabase client: {safe_msg}") from e


def reset_supabase_client():
    """Resets the singleton client instance (useful for test isolation)."""
    global _supabase_client
    _supabase_client = None


def _sanitize_error(error_str: str) -> str:
    """Removes sensitive keys, URLs, and internal tokens from exception messages."""
    settings = get_settings()
    cleaned = error_str
    sb_key = getattr(settings, "supabase_key", None)
    if sb_key and isinstance(sb_key, str):
        cleaned = cleaned.replace(sb_key.strip(), "***MASKED***")
    sb_url = getattr(settings, "supabase_url", None)
    if sb_url and isinstance(sb_url, str):
        cleaned = cleaned.replace(sb_url.strip(), "***URL***")
    sb_jwt = getattr(settings, "supabase_jwt_secret", None)
    if sb_jwt and isinstance(sb_jwt, str):
        cleaned = cleaned.replace(sb_jwt.strip(), "***MASKED***")
    llm_key = getattr(settings, "llm_api_key", None)
    if llm_key and isinstance(llm_key, str):
        cleaned = cleaned.replace(llm_key.strip(), "***MASKED***")
    return cleaned



def validate_defect_report_payload(
    reporter_name: Any,
    employee_id: Optional[str] = None,
    defect_description: Optional[str] = None,
    category: Optional[str] = None,
    confidence: Optional[float] = None,
    reliability: Optional[str] = None,
    explanation: Optional[str] = None,
    classification_mode: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    confidence_level: Optional[str] = None,
    confidence_range: Optional[str] = None,
    raw_score: Optional[float] = None,
    calibrated_score: Optional[float] = None,
    is_multi_defect: Optional[bool] = None,
    defect_count: Optional[int] = None,
    validation_status: Optional[str] = None,
    user_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Validates required fields, ensures employee_id is text, and verifies the category
    strictly belongs to the authoritative 8-category taxonomy.
    Supports either dictionary object or keyword arguments.
    """
    if isinstance(reporter_name, dict):
        d = reporter_name
        reporter_name = d.get("reporter_name")
        employee_id = d.get("employee_id")
        defect_description = d.get("defect_description")
        category = d.get("category")
        confidence = d.get("confidence")
        reliability = d.get("reliability")
        explanation = d.get("explanation")
        classification_mode = d.get("classification_mode")
        provider = d.get("provider")
        model = d.get("model")
        confidence_level = d.get("confidence_level", confidence_level)
        confidence_range = d.get("confidence_range", confidence_range)
        raw_score = d.get("raw_score", raw_score)
        calibrated_score = d.get("calibrated_score", calibrated_score)
        is_multi_defect = d.get("is_multi_defect", is_multi_defect)
        defect_count = d.get("defect_count", defect_count)
        validation_status = d.get("validation_status", validation_status)
        user_id = d.get("user_id", user_id)

    if confidence_level is None and "confidence_level" in kwargs:
        confidence_level = kwargs["confidence_level"]
    if confidence_range is None and "confidence_range" in kwargs:
        confidence_range = kwargs["confidence_range"]
    if raw_score is None and "raw_score" in kwargs:
        raw_score = kwargs["raw_score"]
    if calibrated_score is None and "calibrated_score" in kwargs:
        calibrated_score = kwargs["calibrated_score"]
    if is_multi_defect is None and "is_multi_defect" in kwargs:
        is_multi_defect = kwargs["is_multi_defect"]
    if defect_count is None and "defect_count" in kwargs:
        defect_count = kwargs["defect_count"]
    if validation_status is None and "validation_status" in kwargs:
        validation_status = kwargs["validation_status"]

    if not reporter_name or not str(reporter_name).strip():
        raise ValueError("Reporter Name is required and cannot be empty.")

    if not employee_id or not str(employee_id).strip():
        raise ValueError("Employee ID is required and cannot be empty.")

    if not defect_description or not str(defect_description).strip():
        raise ValueError("Defect Description is required and cannot be empty.")

    tax_repo = get_taxonomy_repository()
    clean_cat = str(category).strip() if category is not None else ""
    if not tax_repo.is_valid_category(clean_cat):
        approved = ", ".join(tax_repo.get_categories())
        raise ValueError(f"Invalid category '{clean_cat}'. Must be one of approved taxonomy categories: {approved}.")

    # Confidence must be a valid float/numeric or None without artificial manipulation
    numeric_conf = None
    if confidence is not None:
        try:
            numeric_conf = float(confidence)
        except (ValueError, TypeError):
            numeric_conf = None

    numeric_raw_score = None
    if raw_score is not None:
        try:
            numeric_raw_score = float(raw_score)
        except (ValueError, TypeError):
            numeric_raw_score = None

    numeric_calibrated_score = None
    if calibrated_score is not None:
        try:
            numeric_calibrated_score = float(calibrated_score)
        except (ValueError, TypeError):
            numeric_calibrated_score = None

    int_defect_count = 1
    if defect_count is not None:
        try:
            int_defect_count = int(defect_count)
            if int_defect_count < 0:
                raise ValueError("defect_count must be >= 0.")
        except (TypeError, ValueError) as e:
            if "defect_count must be >= 0" in str(e):
                raise
            raise ValueError("defect_count must be a valid non-negative integer.")

    clean_validation_status = None
    if validation_status is not None:
        vs_str = str(validation_status).strip().upper()
        if vs_str not in ("VALID", "PARTIAL", "INVALID", "UNKNOWN"):
            raise ValueError(f"Invalid validation_status '{validation_status}'. Must be one of: VALID, PARTIAL, INVALID, UNKNOWN.")
        clean_validation_status = vs_str

    clean_user_id = None
    if user_id is not None and str(user_id).strip():
        from uuid import UUID
        try:
            val_uuid = UUID(str(user_id).strip())
            clean_user_id = str(val_uuid)
        except (ValueError, AttributeError, TypeError):
            raise ValueError(f"Invalid user_id '{user_id}': user_id must be a valid UUID string.")

    return {
        "reporter_name": str(reporter_name).strip(),
        "employee_id": str(employee_id).strip(),
        "defect_description": str(defect_description).strip(),
        "category": clean_cat,
        "confidence": numeric_conf,
        "reliability": str(reliability).strip() if reliability else None,
        "explanation": str(explanation).strip() if explanation else None,
        "classification_mode": str(classification_mode).strip() if classification_mode else None,
        "provider": str(provider).strip() if provider else None,
        "model": str(model).strip() if model else None,
        "confidence_level": str(confidence_level).strip() if confidence_level else None,
        "confidence_range": str(confidence_range).strip() if confidence_range else None,
        "raw_score": numeric_raw_score,
        "calibrated_score": numeric_calibrated_score,
        "is_multi_defect": bool(is_multi_defect) if is_multi_defect is not None else False,
        "defect_count": int_defect_count,
        "validation_status": clean_validation_status,
        "user_id": clean_user_id
    }


def save_defect_report(
    reporter_name: Any,
    employee_id: Optional[str] = None,
    defect_description: Optional[str] = None,
    category: Optional[str] = None,
    confidence: Optional[float] = None,
    reliability: Optional[str] = None,
    explanation: Optional[str] = None,
    classification_mode: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    client: Optional[Any] = None,
    confidence_level: Optional[str] = None,
    confidence_range: Optional[str] = None,
    raw_score: Optional[float] = None,
    calibrated_score: Optional[float] = None,
    is_multi_defect: Optional[bool] = None,
    defect_count: Optional[int] = None,
    validation_status: Optional[str] = None,
    user_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Validates and persists a defect report to the Supabase defect_reports table.
    Uses database-generated timestamp and UUID.
    Supports either dictionary object or keyword arguments.
    """
    if isinstance(reporter_name, dict):
        d = reporter_name
        return save_defect_report(
            reporter_name=d.get("reporter_name"),
            employee_id=d.get("employee_id"),
            defect_description=d.get("defect_description"),
            category=d.get("category"),
            confidence=d.get("confidence"),
            reliability=d.get("reliability"),
            explanation=d.get("explanation"),
            classification_mode=d.get("classification_mode"),
            provider=d.get("provider"),
            model=d.get("model"),
            client=client or d.get("client"),
            confidence_level=d.get("confidence_level", confidence_level),
            confidence_range=d.get("confidence_range", confidence_range),
            raw_score=d.get("raw_score", raw_score),
            calibrated_score=d.get("calibrated_score", calibrated_score),
            is_multi_defect=d.get("is_multi_defect", is_multi_defect),
            defect_count=d.get("defect_count", defect_count),
            validation_status=d.get("validation_status", validation_status),
            user_id=d.get("user_id", user_id),
            **kwargs
        )

    payload = validate_defect_report_payload(
        reporter_name=reporter_name,
        employee_id=employee_id,
        defect_description=defect_description,
        category=category,
        confidence=confidence,
        reliability=reliability,
        explanation=explanation,
        classification_mode=classification_mode,
        provider=provider,
        model=model,
        confidence_level=confidence_level,
        confidence_range=confidence_range,
        raw_score=raw_score,
        calibrated_score=calibrated_score,
        is_multi_defect=is_multi_defect,
        defect_count=defect_count,
        validation_status=validation_status,
        user_id=user_id,
        **kwargs
    )

    sb_client = client or get_supabase_client()

    try:
        response = sb_client.table("defect_reports").insert(payload).execute()
        if hasattr(response, "data") and response.data:
            return response.data[0]
        return payload
    except Exception as e:
        safe_msg = _sanitize_error(str(e))
        raise DatabaseError(f"Failed to save defect report: {safe_msg}") from e


def validate_defect_item_payload(
    report_id: Any,
    defect_index: Optional[int] = None,
    defect_text: Optional[str] = None,
    category: Optional[str] = None,
    defect_id: Optional[int] = None,
    segment_id: Optional[int] = None,
    start_char: Optional[int] = None,
    end_char: Optional[int] = None,
    confidence: Optional[float] = None,
    confidence_level: Optional[str] = None,
    confidence_range: Optional[str] = None,
    raw_score: Optional[float] = None,
    calibrated_score: Optional[float] = None,
    top2_margin: Optional[float] = None,
    reliability: Optional[str] = None,
    explanation: Optional[str] = None,
    classification_mode: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    status: Optional[str] = None,
    is_ambiguous: bool = False,
    ambiguity_reason: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Validates a child defect item payload for the defect_report_items table.
    Enforces required report_id, defect_text, defect_index >= 1, approved taxonomy category,
    source span boundaries, and confidence bounds [0.0 - 1.0].
    Supports either dictionary object or keyword arguments.
    """
    if isinstance(report_id, dict):
        d = report_id
        report_id = d.get("report_id")
        defect_index = d.get("defect_index", defect_index)
        defect_text = d.get("defect_text", defect_text)
        category = d.get("category", category)
        defect_id = d.get("defect_id", defect_id)
        segment_id = d.get("segment_id", segment_id)
        start_char = d.get("start_char", start_char)
        end_char = d.get("end_char", end_char)
        confidence = d.get("confidence", confidence)
        confidence_level = d.get("confidence_level", confidence_level)
        confidence_range = d.get("confidence_range", confidence_range)
        raw_score = d.get("raw_score", raw_score)
        calibrated_score = d.get("calibrated_score", calibrated_score)
        top2_margin = d.get("top2_margin", top2_margin)
        reliability = d.get("reliability", reliability)
        explanation = d.get("explanation", explanation)
        classification_mode = d.get("classification_mode", classification_mode)
        provider = d.get("provider", provider)
        model = d.get("model", model)
        status = d.get("status", status)
        is_ambiguous = d.get("is_ambiguous", is_ambiguous)
        ambiguity_reason = d.get("ambiguity_reason", ambiguity_reason)

    if not report_id or not str(report_id).strip():
        raise ValueError("report_id is required and cannot be empty.")

    if not defect_text or not str(defect_text).strip():
        raise ValueError("defect_text is required and cannot be empty.")

    if defect_index is None:
        raise ValueError("defect_index is required and cannot be None.")
    try:
        idx = int(defect_index)
        if idx < 1:
            raise ValueError("defect_index must be an integer >= 1.")
    except (ValueError, TypeError) as e:
        if "defect_index must be an integer >= 1" in str(e):
            raise
        raise ValueError("defect_index must be a valid integer >= 1.")

    tax_repo = get_taxonomy_repository()
    clean_cat = str(category).strip() if category is not None else ""
    if not tax_repo.is_valid_category(clean_cat):
        approved = ", ".join(tax_repo.get_categories())
        raise ValueError(f"Invalid category '{clean_cat}'. Must be one of approved taxonomy categories: {approved}.")

    # Source span validations
    int_start = None
    if start_char is not None:
        try:
            int_start = int(start_char)
            if int_start < 0:
                raise ValueError("start_char must be >= 0.")
        except (ValueError, TypeError) as e:
            if "start_char must be >= 0" in str(e):
                raise
            raise ValueError("start_char must be an integer >= 0.")

    int_end = None
    if end_char is not None:
        try:
            int_end = int(end_char)
            if int_end < 0:
                raise ValueError("end_char must be >= 0.")
        except (ValueError, TypeError) as e:
            if "end_char must be >= 0" in str(e):
                raise
            raise ValueError("end_char must be an integer >= 0.")

    if int_start is not None and int_end is not None:
        if int_start == 0 and int_end == 0:
            int_start = None
            int_end = None
        elif int_end <= int_start:
            raise ValueError("end_char must be strictly greater than start_char.")

    # Confidence validation
    numeric_conf = None
    if confidence is not None:
        try:
            numeric_conf = float(confidence)
            if numeric_conf < 0.0 or numeric_conf > 1.0:
                raise ValueError("confidence must be between 0.0 and 1.0.")
        except (ValueError, TypeError) as e:
            if "confidence must be between 0.0 and 1.0" in str(e):
                raise
            numeric_conf = None

    # Calibrated score validation
    numeric_calibrated = None
    if calibrated_score is not None:
        try:
            numeric_calibrated = float(calibrated_score)
            if numeric_calibrated < 0.0 or numeric_calibrated > 1.0:
                raise ValueError("calibrated_score must be between 0.0 and 1.0.")
        except (ValueError, TypeError) as e:
            if "calibrated_score must be between 0.0 and 1.0" in str(e):
                raise
            numeric_calibrated = None

    numeric_raw = None
    if raw_score is not None:
        try:
            numeric_raw = float(raw_score)
        except (ValueError, TypeError):
            numeric_raw = None

    numeric_margin = None
    if top2_margin is not None:
        try:
            numeric_margin = float(top2_margin)
        except (ValueError, TypeError):
            numeric_margin = None

    clean_status = str(status).strip().lower() if status else None
    valid_statuses = {"success", "unknown", "model_error", "validation_error", "system_error", "low_confidence"}
    if clean_status and clean_status not in valid_statuses:
        raise ValueError(f"Invalid status '{status}'. Must be one of: {', '.join(sorted(valid_statuses))}.")

    return {
        "report_id": str(report_id).strip(),
        "defect_index": idx,
        "defect_id": int(defect_id) if defect_id is not None else idx,
        "segment_id": int(segment_id) if segment_id is not None else idx,
        "defect_text": str(defect_text).strip(),
        "start_char": int_start,
        "end_char": int_end,
        "category": clean_cat,
        "confidence": numeric_conf,
        "confidence_level": str(confidence_level).strip() if confidence_level else None,
        "confidence_range": str(confidence_range).strip() if confidence_range else None,
        "raw_score": numeric_raw,
        "calibrated_score": numeric_calibrated,
        "top2_margin": numeric_margin,
        "reliability": str(reliability).strip() if reliability else None,
        "explanation": str(explanation).strip() if explanation else None,
        "classification_mode": str(classification_mode).strip() if classification_mode else None,
        "provider": str(provider).strip() if provider else None,
        "model": str(model).strip() if model else None,
        "status": clean_status or "success",
        "is_ambiguous": bool(is_ambiguous),
        "ambiguity_reason": str(ambiguity_reason).strip() if (is_ambiguous and ambiguity_reason) else None,
    }


def save_multi_defect_report(
    reporter_name: Any,
    employee_id: Optional[str] = None,
    classification_result: Optional[Any] = None,
    client: Optional[Any] = None,
    user_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Validates and persists a multi-defect classification report:
    1. Validates structural consistency (rejects structurally INVALID results).
    2. Persists parent report in public.defect_reports with multi-defect metadata and optional user_id.
    3. Persists individual child defect items in public.defect_report_items.
    4. For Unknown/non-defect inputs, preserves Unknown outcome without fabricating child items.
    """
    if isinstance(reporter_name, dict):
        d = reporter_name
        return save_multi_defect_report(
            reporter_name=d.get("reporter_name"),
            employee_id=d.get("employee_id"),
            classification_result=d.get("classification_result", classification_result),
            client=client or d.get("client"),
            user_id=d.get("user_id", user_id),
            **kwargs
        )

    if not classification_result:
        raise ValueError("classification_result is required and cannot be None.")

    # 1. Structural Validation Inspection
    val_res = getattr(classification_result, "validation_result", None)
    md_res = getattr(classification_result, "multi_defect_classification", None)
    if val_res is None and md_res is not None:
        val_res = getattr(md_res, "validation_result", None)

    if val_res is not None:
        val_status = getattr(val_res, "status", "VALID")
        if val_status == "INVALID":
            err_msgs = [getattr(e, "message", str(e)) for e in getattr(val_res, "errors", [])]
            err_summary = "; ".join(err_msgs) if err_msgs else "Multi-defect structural validation failed."
            raise MultiDefectPersistenceError(f"Cannot persist structurally INVALID multi-defect classification: {err_summary}")

    # 2. Resolve parent attributes
    if hasattr(classification_result, "original_description"):
        parent_desc = classification_result.original_description
        parent_cat = classification_result.category
        parent_conf = classification_result.confidence
        parent_rel = classification_result.reliability
        parent_exp = classification_result.reason
        parent_mode = getattr(classification_result, "model_source", None) or "local"
        parent_provider = getattr(classification_result, "model_source", None)
        parent_model = getattr(classification_result, "model_source", None)
        ca = getattr(classification_result, "confidence_assessment", None)
        parent_conf_level = ca.level if ca else parent_rel
        parent_conf_range = ca.approximate_range if ca else None
        parent_raw_score = ca.raw_score if ca else parent_conf
        parent_cal_score = ca.calibrated_prob if ca else None
        defects = md_res.defects if md_res else []
        is_multi_defect = bool(md_res and md_res.is_multi_defect)
        defect_count = md_res.defect_count if md_res else (len(defects) if defects else 1)
        validation_status = val_res.status if val_res else "VALID"
    elif hasattr(classification_result, "original_text"):
        # Direct MultiDefectClassificationResult passed
        parent_desc = classification_result.original_text
        defects = getattr(classification_result, "defects", [])
        is_multi_defect = getattr(classification_result, "is_multi_defect", False)
        defect_count = getattr(classification_result, "defect_count", len(defects))
        validation_status = val_res.status if val_res else "VALID"
        first_d = defects[0] if defects else None
        parent_cat = first_d.category if first_d else "Unknown"
        parent_conf = first_d.confidence if first_d else None
        parent_rel = first_d.reliability if first_d else "Medium"
        parent_exp = "; ".join([d.explanation for d in defects if getattr(d, "explanation", None)]) if defects else "Multi-defect report"
        parent_mode = first_d.classification_mode if first_d else "local"
        parent_provider = first_d.provider if first_d else None
        parent_model = first_d.model if first_d else None
        ca = first_d.confidence_assessment if first_d else None
        parent_conf_level = ca.level if ca else parent_rel
        parent_conf_range = ca.approximate_range if ca else None
        parent_raw_score = ca.raw_score if ca else parent_conf
        parent_cal_score = ca.calibrated_prob if ca else None
    else:
        raise ValueError("Invalid classification_result: must be ClassificationResult or MultiDefectClassificationResult.")

    # 3. Handle Unknown / Non-defect inputs (do not fabricate child items)
    if parent_cat == "Unknown" and (not defects or defect_count == 0):
        defects = []
        is_multi_defect = False
        defect_count = 0

    # 4. Pre-validate child payloads BEFORE writing to database
    child_payloads = []
    dummy_report_id = "00000000-0000-0000-0000-000000000000"
    for idx, d in enumerate(defects, start=1):
        d_ca = getattr(d, "confidence_assessment", None)
        d_amb = getattr(d, "ambiguity_assessment", None)
        item_dict = {
            "report_id": dummy_report_id,
            "defect_index": idx,
            "defect_id": getattr(d, "defect_id", idx),
            "segment_id": getattr(d, "segment_id", idx),
            "defect_text": getattr(d, "text", ""),
            "start_char": getattr(d, "source_start_char", None),
            "end_char": getattr(d, "source_end_char", None),
            "category": getattr(d, "category", ""),
            "confidence": getattr(d, "confidence", None),
            "confidence_level": d_ca.level if d_ca else getattr(d, "reliability", None),
            "confidence_range": d_ca.approximate_range if d_ca else None,
            "raw_score": getattr(d, "raw_score", None) if getattr(d, "raw_score", None) is not None else (d_ca.raw_score if d_ca else getattr(d, "confidence", None)),
            "calibrated_score": getattr(d, "calibrated_prob", None) if getattr(d, "calibrated_prob", None) is not None else (d_ca.calibrated_prob if d_ca else None),
            "top2_margin": getattr(d, "top2_margin", None) if getattr(d, "top2_margin", None) is not None else (d_ca.top2_margin if d_ca else None),
            "reliability": getattr(d, "reliability", None),
            "explanation": getattr(d, "explanation", None),
            "classification_mode": getattr(d, "classification_mode", parent_mode),
            "provider": getattr(d, "provider", parent_provider),
            "model": getattr(d, "model", parent_model),
            "status": getattr(d, "status", "success"),
            "is_ambiguous": bool(d_amb and d_amb.is_ambiguous),
            "ambiguity_reason": d_amb.reason if (d_amb and d_amb.is_ambiguous) else None,
        }
        validated_item = validate_defect_item_payload(item_dict)
        child_payloads.append(validated_item)

    # 5. Persist parent report
    sb_client = client or get_supabase_client()
    parent_record = save_defect_report(
        reporter_name=reporter_name,
        employee_id=employee_id,
        defect_description=parent_desc,
        category=parent_cat,
        confidence=parent_conf,
        reliability=parent_rel,
        explanation=parent_exp,
        classification_mode=parent_mode,
        provider=parent_provider,
        model=parent_model,
        client=sb_client,
        confidence_level=parent_conf_level,
        confidence_range=parent_conf_range,
        raw_score=parent_raw_score,
        calibrated_score=parent_cal_score,
        is_multi_defect=is_multi_defect,
        defect_count=defect_count,
        validation_status=validation_status,
        user_id=user_id,
        **kwargs
    )

    parent_id = parent_record.get("id") or getattr(classification_result, "id", None) or "00000000-0000-0000-0000-000000000000"
    parent_record["id"] = parent_id

    # 6. Persist child defect items
    child_records = []
    if child_payloads:
        for item in child_payloads:
            item["report_id"] = parent_id
        try:
            resp = sb_client.table("defect_report_items").insert(child_payloads).execute()
            if hasattr(resp, "data") and resp.data:
                child_records = resp.data
            else:
                child_records = child_payloads
        except Exception as e:
            safe_msg = _sanitize_error(str(e))
            raise DatabaseError(f"Failed to save defect report items: {safe_msg}") from e

    res = dict(parent_record)
    res["parent"] = parent_record
    res["items"] = child_records
    return res


def get_defect_report_items(report_id: str, client: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Retrieves all child defect items for a given report ID ordered by defect_index."""
    sb_client = client or get_supabase_client()
    try:
        query = sb_client.table("defect_report_items").select("*").eq("report_id", str(report_id)).order("defect_index", desc=False)
        response = query.execute()
        return response.data if hasattr(response, "data") and response.data is not None else []
    except Exception as e:
        safe_msg = _sanitize_error(str(e))
        raise DatabaseError(f"Failed to query defect report items: {safe_msg}") from e


def get_defect_report_by_id(report_id: str, client: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single defect report by its UUID."""
    sb_client = client or get_supabase_client()
    try:
        query = sb_client.table("defect_reports").select("*").eq("id", str(report_id)).limit(1)
        response = query.execute()
        if hasattr(response, "data") and response.data:
            return response.data[0]
        return None
    except Exception as e:
        safe_msg = _sanitize_error(str(e))
        raise DatabaseError(f"Failed to query defect report: {safe_msg}") from e


def get_defect_reports(
    limit: int = 50,
    category: Optional[str] = None,
    employee_id: Optional[str] = None,
    client: Optional[Any] = None,
    offset: int = 0,
    user_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves recent defect reports sorted with newest records first (created_at DESC).
    Supports optional filtering by category, employee_id, and user_id, with pagination offset.
    """
    sb_client = client or get_supabase_client()

    try:
        if offset > 0:
            query = sb_client.table("defect_reports").select("*").order("created_at", desc=True).range(offset, offset + limit - 1)
        else:
            query = sb_client.table("defect_reports").select("*").order("created_at", desc=True).limit(limit)

        if user_id and str(user_id).strip():
            query = query.eq("user_id", str(user_id).strip())

        if category and category.strip() and category != "All":
            query = query.eq("category", category.strip())

        if employee_id and employee_id.strip():
            query = query.ilike("employee_id", f"%{employee_id.strip()}%")

        response = query.execute()

        return response.data if hasattr(response, "data") and response.data is not None else []
    except Exception as e:
        safe_msg = _sanitize_error(str(e))
        raise DatabaseError(f"Failed to query defect reports: {safe_msg}") from e


def get_reports_by_employee(employee_id: str, limit: int = 50, client: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Retrieves defect reports for a specific employee ID."""
    return get_defect_reports(limit=limit, employee_id=employee_id, client=client)


def get_reports_by_category(category: str, limit: int = 50, client: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Retrieves defect reports for a specific fault category."""
    return get_defect_reports(limit=limit, category=category, client=client)


def check_database_readiness(client: Optional[Any] = None) -> Dict[str, Any]:
    """Lightweight read-only health check verifying Supabase connectivity.

    Performs a minimal limit(1) read on public.defect_reports without inserting,
    updating, or deleting any data. Strictly sanitizes errors to prevent disclosure
    of keys, tokens, or connection parameters.

    Returns:
        Dict[str, Any] with keys:
            - healthy: bool
            - status: "healthy" | "degraded"
            - message: str
            - details: Dict[str, Any]
    """
    if not is_supabase_configured():
        return {
            "healthy": False,
            "status": "degraded",
            "message": "Database is not configured",
            "details": {
                "configured": False,
                "connected": False
            }
        }

    try:
        sb_client = client or get_supabase_client()
        response = sb_client.table("defect_reports").select("id").limit(1).execute()
        return {
            "healthy": True,
            "status": "healthy",
            "message": "Database connection is available",
            "details": {
                "configured": True,
                "connected": True
            }
        }
    except Exception as e:
        safe_msg = _sanitize_error(str(e))
        logger.warning(f"Database readiness check failed: {safe_msg}")
        return {
            "healthy": False,
            "status": "degraded",
            "message": "Database connection is unavailable",
            "details": {
                "configured": True,
                "connected": False
            }
        }


def validate_profile_payload(
    user_id: Any,
    display_name: Optional[str] = None,
    email: Optional[str] = None,
    role: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Validates user profile payload ensuring non-empty ID and display_name.

    Supports either dictionary object or keyword arguments.
    Enforces that display_name is not empty and <= 100 characters.
    Enforces that role is the supported application role ('employee').
    """
    if isinstance(user_id, dict):
        d = user_id
        user_id = d.get("id", d.get("user_id"))
        display_name = d.get("display_name", display_name)
        email = d.get("email", email)
        role = d.get("role", role)

    if not user_id or not str(user_id).strip():
        raise ValueError("User ID (id) is required and cannot be empty.")

    clean_user_id = str(user_id).strip()

    clean_name = str(display_name).strip() if display_name is not None else ""
    if not clean_name:
        raise ValueError("Display name is required and cannot be empty.")
    if len(clean_name) > 100:
        raise ValueError("Display name cannot exceed 100 characters.")

    clean_email = str(email).strip() if email and str(email).strip() else None

    # Step 5.3: Enforce valid application role ('employee')
    clean_role = str(role).strip().lower() if role is not None else "employee"
    if clean_role != "employee":
        raise ValueError(f"Invalid role '{role}'. Only 'employee' is currently supported.")

    return {
        "id": clean_user_id,
        "display_name": clean_name,
        "email": clean_email,
        "role": clean_role,
    }


def get_profile(user_id: str, client: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """Retrieves the user profile for a given user UUID.

    Returns the profile dict or None if not found.
    Sanitizes errors to prevent key/URL leaks.
    """
    if not user_id or not str(user_id).strip():
        raise ValueError("User ID is required to query profile.")

    sb_client = client or get_supabase_client()
    try:
        query = sb_client.table("profiles").select("*").eq("id", str(user_id).strip()).limit(1)
        response = query.execute()
        if hasattr(response, "data") and response.data:
            return response.data[0]
        return None
    except Exception as e:
        safe_msg = _sanitize_error(str(e))
        raise DatabaseError(f"Failed to query user profile: {safe_msg}") from e


def create_profile(
    user_id: str,
    display_name: str,
    email: Optional[str] = None,
    role: str = "employee",
    client: Optional[Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """Creates a new user profile record in public.profiles.

    Enforces validation and safe error handling.
    """
    payload = validate_profile_payload(
        user_id=user_id,
        display_name=display_name,
        email=email,
        role=role,
        **kwargs
    )
    sb_client = client or get_supabase_client()
    try:
        response = sb_client.table("profiles").insert(payload).execute()
        if hasattr(response, "data") and response.data:
            return response.data[0]
        return payload
    except Exception as e:
        safe_msg = _sanitize_error(str(e))
        raise DatabaseError(f"Failed to create user profile: {safe_msg}") from e


def update_profile(
    user_id: str,
    display_name: Optional[str] = None,
    email: Optional[str] = None,
    client: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """Updates an existing user profile record in public.profiles.

    Only provided fields are updated. updated_at is refreshed.
    """
    if not user_id or not str(user_id).strip():
        raise ValueError("User ID is required to update profile.")

    update_payload: Dict[str, Any] = {}
    if display_name is not None:
        clean_name = str(display_name).strip()
        if not clean_name:
            raise ValueError("Display name cannot be empty or blank.")
        if len(clean_name) > 100:
            raise ValueError("Display name cannot exceed 100 characters.")
        update_payload["display_name"] = clean_name

    if email is not None:
        update_payload["email"] = str(email).strip() if str(email).strip() else None

    if not update_payload:
        return get_profile(user_id=user_id, client=client)

    from datetime import datetime, timezone
    update_payload["updated_at"] = datetime.now(timezone.utc).isoformat()

    sb_client = client or get_supabase_client()
    try:
        response = sb_client.table("profiles").update(update_payload).eq("id", str(user_id).strip()).execute()
        if hasattr(response, "data") and response.data:
            return response.data[0]
        return get_profile(user_id=user_id, client=client)
    except Exception as e:
        safe_msg = _sanitize_error(str(e))
        raise DatabaseError(f"Failed to update user profile: {safe_msg}") from e


def get_or_create_profile(
    user_id: str,
    display_name: Optional[str] = None,
    email: Optional[str] = None,
    role: str = "employee",
    client: Optional[Any] = None
) -> Dict[str, Any]:
    """Retrieves the user profile, or initializes a default profile if it does not yet exist.

    Provides application-level fallback ensuring users always have an accessible profile.
    """
    existing = get_profile(user_id=user_id, client=client)
    if existing:
        return existing

    default_name = display_name or (email.split("@")[0] if email and "@" in email else "Operator")
    return create_profile(
        user_id=user_id,
        display_name=default_name,
        email=email,
        role=role,
        client=client
    )


