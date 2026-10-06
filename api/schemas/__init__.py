"""API Pydantic Schemas for Industrial Defect Intelligence Backend.

Defines versioned, strict request and response schemas for API endpoints.
Preserves existing core classification data models and structures without
duplicating classification logic or leaking secrets.
"""

from api.schemas.classification import (
    ClassificationMode,
    ClassificationRequest,
    ConfidenceAssessmentSchema,
    AmbiguityAssessmentSchema,
    PerDefectClassificationSchema,
    ValidationIssueSchema,
    MultiDefectValidationSchema,
    ClassificationData,
    ClassificationResponse,
)
from api.schemas.history import (
    DefectReportSummarySchema,
    HistoryListData,
    HistoryListResponse,
    DefectReportItemSchema,
    DefectReportDetailData,
    HistoryDetailResponse,
)
from api.schemas.health import (
    HealthStatus,
    SubsystemCheck,
    HealthChecks,
    StatusData,
    StatusResponse,
)
from api.schemas.errors import (
    ErrorCode,
    FieldValidationError,
    ErrorPayload,
    ErrorResponse,
)
from api.schemas.auth import (
    AuthenticatedUser,
    UserResponseData,
    UserResponse,
)
from api.schemas.profile import (
    ProfileData,
    ProfileResponse,
    ProfileUpdateRequest,
)

__all__ = [
    "ClassificationMode",
    "ClassificationRequest",
    "ConfidenceAssessmentSchema",
    "AmbiguityAssessmentSchema",
    "PerDefectClassificationSchema",
    "ValidationIssueSchema",
    "MultiDefectValidationSchema",
    "ClassificationData",
    "ClassificationResponse",
    "DefectReportSummarySchema",
    "HistoryListData",
    "HistoryListResponse",
    "DefectReportItemSchema",
    "DefectReportDetailData",
    "HistoryDetailResponse",
    "HealthStatus",
    "SubsystemCheck",
    "HealthChecks",
    "StatusData",
    "StatusResponse",
    "ErrorCode",
    "FieldValidationError",
    "ErrorPayload",
    "ErrorResponse",
    "AuthenticatedUser",
    "UserResponseData",
    "UserResponse",
    "ProfileData",
    "ProfileResponse",
    "ProfileUpdateRequest",
]

