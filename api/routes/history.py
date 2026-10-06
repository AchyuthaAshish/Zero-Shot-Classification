"""History Router for Industrial Defect Intelligence FastAPI Backend.

Provides:
- GET /api/v1/history: List stored defect classification reports with optional
  filtering by category and employee_id, and safe pagination (limit, offset).
- GET /api/v1/history/{report_id}: Detailed inspection of a stored defect report,
  including child defect segments for multi-defect records.
- Strict repository pattern usage (Route -> Repository -> Supabase; zero direct SQL/table queries in route).
- Sanitized error handling preventing disclosure of database URLs, keys, or internal schema structures.
"""

import logging
from uuid import UUID
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Path, status, Depends

from api.dependencies import get_optional_user
from api.schemas.auth import AuthenticatedUser

from api.schemas.history import (
    DefectReportSummarySchema,
    HistoryListData,
    HistoryListResponse,
    DefectReportItemSchema,
    DefectReportDetailData,
    HistoryDetailResponse,
)
from api.schemas.errors import ErrorResponse
import persistence.repository as repository
from persistence.repository import DatabaseError
from api.errors import NotFoundError, ValidationErrorAPI, DatabaseErrorAPI, APIError


logger = logging.getLogger("api.routes.history")

router = APIRouter(prefix="/history", tags=["History"])


def _map_report_summary(r: Dict[str, Any]) -> DefectReportSummarySchema:
    """Transforms a raw repository defect report dictionary into DefectReportSummarySchema."""
    is_multi = bool(r.get("is_multi_defect"))
    raw_cnt = r.get("defect_count")
    if raw_cnt is None:
        defect_count = 2 if is_multi else (0 if r.get("category") == "Unknown" else 1)
    else:
        try:
            defect_count = int(raw_cnt)
        except (ValueError, TypeError):
            defect_count = 1

    return DefectReportSummarySchema(
        id=str(r.get("id")),
        reporter_name=r.get("reporter_name"),
        employee_id=r.get("employee_id"),
        defect_description=r.get("defect_description"),
        category=r.get("category"),
        confidence=r.get("confidence"),
        confidence_level=r.get("confidence_level") or r.get("reliability"),
        confidence_range=r.get("confidence_range"),
        calibrated_score=r.get("calibrated_score"),
        raw_score=r.get("raw_score") if r.get("raw_score") is not None else r.get("confidence"),
        top2_margin=r.get("top2_margin"),
        reliability=r.get("reliability"),
        explanation=r.get("explanation"),
        classification_mode=r.get("classification_mode"),
        provider=r.get("provider"),
        model=r.get("model"),
        is_multi_defect=is_multi,
        defect_count=defect_count,
        validation_status=r.get("validation_status") or ("VALID" if not is_multi else "N/A"),
        created_at=r.get("created_at")
    )


def _map_report_item(item: Dict[str, Any]) -> DefectReportItemSchema:
    """Transforms a raw repository child defect item dictionary into DefectReportItemSchema."""
    return DefectReportItemSchema(
        id=str(item.get("id")) if item.get("id") else None,
        report_id=str(item.get("report_id")),
        defect_index=int(item.get("defect_index", 1)),
        defect_id=item.get("defect_id"),
        segment_id=item.get("segment_id"),
        defect_text=str(item.get("defect_text", "")),
        start_char=item.get("start_char"),
        end_char=item.get("end_char"),
        category=str(item.get("category", "")),
        confidence=item.get("confidence"),
        confidence_level=item.get("confidence_level") or item.get("reliability"),
        confidence_range=item.get("confidence_range"),
        raw_score=item.get("raw_score"),
        calibrated_score=item.get("calibrated_score"),
        top2_margin=item.get("top2_margin"),
        reliability=item.get("reliability"),
        explanation=item.get("explanation"),
        classification_mode=item.get("classification_mode"),
        provider=item.get("provider"),
        model=item.get("model"),
        status=item.get("status") or "success",
        is_ambiguous=bool(item.get("is_ambiguous")),
        ambiguity_reason=item.get("ambiguity_reason"),
        created_at=item.get("created_at")
    )


@router.get(
    "",
    response_model=HistoryListResponse,
    summary="List Defect Reports History",
    description=(
        "Retrieves previously stored defect classification reports from the persistent Supabase database.\n\n"
        "**Features:**\n"
        "- Sorted in descending order by `created_at` (newest reports first).\n"
        "- Supports optional category filtering (`category`) and employee ID filtering (`employee_id`).\n"
        "- Safe pagination via `limit` (1-100) and non-negative `offset`.\n"
        "- Returns an empty list with HTTP 200 when no records match (does not return 404 for empty results)."
    ),
    responses={
        200: {
            "description": "Historical defect reports returned successfully.",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": {
                            "items": [
                                {
                                    "id": "a0000000-0000-0000-0000-000000000001",
                                    "reporter_name": "Jane Doe",
                                    "employee_id": "EMP-1001",
                                    "defect_description": "Hydraulic line pressure drop in sector 4.",
                                    "category": "Hydraulic System Defect",
                                    "confidence": 0.952,
                                    "confidence_level": "High",
                                    "confidence_range": "~95%",
                                    "calibrated_score": 0.952,
                                    "raw_score": 0.952,
                                    "top2_margin": 0.71,
                                    "reliability": "High",
                                    "explanation": "Pressure drop in hydraulic line indicates hydraulic failure.",
                                    "classification_mode": "local",
                                    "provider": "local",
                                    "model": "Local ML (TF-IDF / MiniLM)",
                                    "is_multi_defect": False,
                                    "defect_count": 1,
                                    "validation_status": "VALID",
                                    "created_at": "2026-10-05T10:00:00Z"
                                }
                            ],
                            "total": 1,
                            "limit": 50,
                            "offset": 0
                        }
                    }
                }
            }
        },
        422: {
            "model": ErrorResponse,
            "description": "Validation error on query parameters (e.g. invalid limit, negative offset, or unknown category filter).",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Invalid category filter 'Unknown Category'. Must be one of approved taxonomy categories.",
                            "details": {"category": "Unknown Category"}
                        }
                    }
                }
            }
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal server error during history query.",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": {
                            "code": "INTERNAL_SERVER_ERROR",
                            "message": "An unexpected error occurred while retrieving history records.",
                            "details": {}
                        }
                    }
                }
            }
        },
        503: {
            "model": ErrorResponse,
            "description": "Persistence database service is unconfigured or unavailable.",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": {
                            "code": "DATABASE_ERROR",
                            "message": "Persistence service is not configured or unavailable.",
                            "details": {}
                        }
                    }
                }
            }
        }
    }
)
def list_defect_reports(
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of reports to return (1 to 100).", examples=[50]),
    offset: int = Query(default=0, ge=0, description="Number of reports to skip for pagination (>= 0).", examples=[0]),
    category: Optional[str] = Query(default=None, description="Optional filter by defect category (e.g. 'Mechanical Fault').", examples=["Mechanical Fault"]),
    employee_id: Optional[str] = Query(default=None, max_length=100, description="Optional filter by reporter employee ID (e.g. 'EMP-001').", examples=["EMP-001"]),
    current_user: Optional[AuthenticatedUser] = Depends(get_optional_user)
) -> HistoryListResponse:
    """Retrieves a paginated list of defect reports from the persistent audit store."""
    if not repository.is_supabase_configured():
        logger.warning("History query attempted but Supabase is not configured.")
        raise DatabaseErrorAPI(
            message="Persistence service is not configured or unavailable."
        )

    # Validate category filter against canonical taxonomy
    if category is not None and category.strip():
        from taxonomy.repository import get_taxonomy_repository
        tax_repo = get_taxonomy_repository()
        clean_cat = category.strip()
        if clean_cat != "All" and not tax_repo.is_valid_category(clean_cat):
            approved = ", ".join(tax_repo.get_categories())
            raise ValidationErrorAPI(
                message=f"Invalid category filter '{clean_cat}'. Must be one of approved taxonomy categories: {approved}.",
                details={"category": clean_cat, "approved_categories": tax_repo.get_categories()}
            )

    # Validate employee_id filter
    if employee_id is not None and len(employee_id.strip()) > 100:
        raise ValidationErrorAPI(
            message="employee_id filter exceeds maximum allowed length of 100 characters.",
            details={"employee_id": employee_id}
        )

    user_id_filter = current_user.user_id if current_user else None
    query_kwargs = {
        "limit": limit,
        "category": category,
        "employee_id": employee_id,
        "offset": offset,
    }
    if user_id_filter:
        query_kwargs["user_id"] = user_id_filter

    try:
        raw_reports = repository.get_defect_reports(**query_kwargs)
    except DatabaseError as exc:
        logger.error(f"Database error during get_defect_reports: {exc}", exc_info=True)
        raise DatabaseErrorAPI(
            message="Database service is currently unavailable. Please try again later."
        )
    except Exception as exc:
        logger.error(f"Unexpected error during get_defect_reports: {exc}", exc_info=True)
        raise APIError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred while retrieving history records."
        )

    # Ensure anonymous callers do not receive protected employee report data
    if current_user is None:
        filtered_reports = [r for r in (raw_reports or []) if not r.get("user_id")]
    else:
        filtered_reports = [r for r in (raw_reports or []) if r.get("user_id") == current_user.user_id or not r.get("user_id")]

    items = [_map_report_summary(r) for r in filtered_reports]

    return HistoryListResponse(
        success=True,
        data=HistoryListData(
            items=items,
            total=len(items),
            limit=limit,
            offset=offset
        )
    )


@router.get(
    "/{report_id}",
    response_model=HistoryDetailResponse,
    summary="Get Defect Report Detail",
    description=(
        "Retrieves a single defect report by its unique UUID.\n\n"
        "**Features:**\n"
        "- For multi-defect reports, automatically queries and returns all child defect segments from `defect_report_items`.\n"
        "- For single-defect and unknown reports, returns the parent record with an empty or single defect items list.\n"
        "- Returns HTTP 404 if the specified report UUID is not found in the database.\n"
        "- Validates UUID format, returning HTTP 422 for malformed identifiers."
    ),
    responses={
        200: {
            "description": "Defect report details and child items retrieved successfully.",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": {
                            "id": "a0000000-0000-0000-0000-000000000002",
                            "reporter_name": "John Smith",
                            "employee_id": "EMP-1002",
                            "defect_description": "Conveyor belt slipping and motor overheating.",
                            "category": None,
                            "confidence": None,
                            "confidence_level": None,
                            "confidence_range": None,
                            "calibrated_score": None,
                            "raw_score": None,
                            "top2_margin": None,
                            "reliability": "High",
                            "explanation": "Identified 2 co-occurring defect signals.",
                            "classification_mode": "local",
                            "provider": "local",
                            "model": "Local ML (TF-IDF / MiniLM)",
                            "is_multi_defect": True,
                            "defect_count": 2,
                            "validation_status": "VALID",
                            "created_at": "2026-10-05T10:15:00Z",
                            "defects": [
                                {
                                    "id": "b0000000-0000-0000-0000-000000000001",
                                    "report_id": "a0000000-0000-0000-0000-000000000002",
                                    "defect_index": 1,
                                    "defect_id": 1,
                                    "segment_id": 1,
                                    "defect_text": "Conveyor belt slipping",
                                    "start_char": 0,
                                    "end_char": 22,
                                    "category": "Mechanical Fault",
                                    "confidence": 0.935,
                                    "confidence_level": "High",
                                    "confidence_range": "~94%",
                                    "raw_score": 0.935,
                                    "calibrated_score": 0.935,
                                    "top2_margin": 0.62,
                                    "reliability": "High",
                                    "explanation": "Belt slipping symptom indicates mechanical traction wear.",
                                    "classification_mode": "local",
                                    "provider": "local",
                                    "model": "Local ML (TF-IDF / MiniLM)",
                                    "status": "success",
                                    "is_ambiguous": False,
                                    "ambiguity_reason": None,
                                    "created_at": "2026-10-05T10:15:00Z"
                                }
                            ]
                        }
                    }
                }
            }
        },
        404: {
            "model": ErrorResponse,
            "description": "Defect report not found.",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": {
                            "code": "NOT_FOUND",
                            "message": "Defect report with ID 'a0000000-0000-0000-0000-000000000099' was not found.",
                            "details": {"report_id": "a0000000-0000-0000-0000-000000000099"}
                        }
                    }
                }
            }
        },
        422: {
            "model": ErrorResponse,
            "description": "Invalid UUID format in report_id parameter.",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "path -> report_id: Input should be a valid UUID, invalid character: expected an optional prefix of `urn:uuid:` followed by [0-9a-fA-F-], found `n` at 1",
                            "details": {
                                "fields": [
                                    {
                                        "field": "path -> report_id",
                                        "message": "Input should be a valid UUID",
                                        "type": "uuid_parsing"
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
            "description": "Internal server error during report detail retrieval.",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": {
                            "code": "INTERNAL_SERVER_ERROR",
                            "message": "An unexpected error occurred while retrieving the report detail.",
                            "details": {}
                        }
                    }
                }
            }
        },
        503: {
            "model": ErrorResponse,
            "description": "Persistence database service is unconfigured or unavailable.",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": {
                            "code": "DATABASE_ERROR",
                            "message": "Persistence service is not configured or unavailable.",
                            "details": {}
                        }
                    }
                }
            }
        }
    }
)
def get_defect_report_detail(
    report_id: UUID = Path(..., description="Unique UUID identifier of the defect report to inspect.", examples=["a0000000-0000-0000-0000-000000000001"]),
    current_user: Optional[AuthenticatedUser] = Depends(get_optional_user)
) -> HistoryDetailResponse:
    """Retrieves full details of a specific defect report, including all child defect segments."""
    if not repository.is_supabase_configured():
        logger.warning("History detail query attempted but Supabase is not configured.")
        raise DatabaseErrorAPI(
            message="Persistence service is not configured or unavailable."
        )

    str_id = str(report_id)

    try:
        report = repository.get_defect_report_by_id(str_id)
    except DatabaseError as exc:
        logger.error(f"Database error during get_defect_report_by_id: {exc}", exc_info=True)
        raise DatabaseErrorAPI(
            message="Database service is currently unavailable. Please try again later."
        )
    except Exception as exc:
        logger.error(f"Unexpected error during get_defect_report_by_id: {exc}", exc_info=True)
        raise APIError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred while retrieving the report detail."
        )

    if not report:
        raise NotFoundError(
            message=f"Defect report with ID '{str_id}' was not found.",
            details={"report_id": str_id}
        )

    # RLS ownership enforcement: protected employee data is inaccessible to unauthorized users
    report_owner = report.get("user_id")
    if report_owner:
        if not current_user or str(current_user.user_id) != str(report_owner):
            raise NotFoundError(
                message=f"Defect report with ID '{str_id}' was not found.",
                details={"report_id": str_id}
            )

    # Retrieve child defect items from repository
    child_items = []
    try:
        raw_items = repository.get_defect_report_items(str_id)
        if raw_items:
            child_items = [_map_report_item(it) for it in raw_items]
    except Exception as exc:
        logger.warning(f"Unable to retrieve child items for report {str_id}: {exc}")
        child_items = []


    summary_data = _map_report_summary(report)
    detail_data = DefectReportDetailData(
        **summary_data.model_dump(),
        defects=child_items
    )

    return HistoryDetailResponse(success=True, data=detail_data)
