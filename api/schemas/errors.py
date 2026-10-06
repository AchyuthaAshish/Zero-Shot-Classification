"""Pydantic schemas for standardized API error responses.

Conforms to Phase 3 Step 3.5:
- Unified error response envelope: {success: false, error: {code, message, details}}
- Controlled set of standardized error codes
- Structured field-level validation errors
- Absolute sanitization: zero keys, passwords, database URLs, or stack traces
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict


class ErrorCode(str, Enum):
    """Controlled set of standardized API error codes."""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    HTTP_ERROR = "HTTP_ERROR"



class FieldValidationError(BaseModel):
    """Field-specific validation error details."""
    field: Optional[str] = Field(
        default=None,
        description="Location path of the invalid parameter or request field."
    )
    message: str = Field(
        ...,
        description="Sanitized explanation of the validation failure."
    )
    type: Optional[str] = Field(
        default=None,
        description="Validation error type identifier."
    )


class ErrorPayload(BaseModel):
    """Standardized error content object."""
    code: ErrorCode = Field(
        ...,
        description="Machine-readable standardized error code."
    )
    message: str = Field(
        ...,
        description="Sanitized human-readable error description."
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured error context and diagnostics without sensitive information."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed.",
                "details": {
                    "fields": [
                        {
                            "field": "body -> description",
                            "message": "Description must not be empty or contain only whitespace.",
                            "type": "value_error"
                        }
                    ]
                }
            }
        }
    )


class ErrorResponse(BaseModel):
    """Standard envelope for all error responses across the API."""
    success: bool = Field(
        default=False,
        description="Indicates an unsuccessful operation; always false for errors."
    )
    error: ErrorPayload = Field(
        ...,
        description="Standardized error payload."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "Validation Error (422)",
                    "value": {
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
                },
                {
                    "summary": "Resource Not Found (404)",
                    "value": {
                        "success": False,
                        "error": {
                            "code": "NOT_FOUND",
                            "message": "Defect report with ID 'a0000000-0000-0000-0000-000000000099' was not found.",
                            "details": {"report_id": "a0000000-0000-0000-0000-000000000099"}
                        }
                    }
                },
                {
                    "summary": "External Provider Error (502)",
                    "value": {
                        "success": False,
                        "error": {
                            "code": "PROVIDER_ERROR",
                            "message": "External classification service is currently unavailable. Please try again later or switch to local mode.",
                            "details": {"provider": "gemini"}
                        }
                    }
                },
                {
                    "summary": "Database Unavailable (503)",
                    "value": {
                        "success": False,
                        "error": {
                            "code": "DATABASE_ERROR",
                            "message": "Persistence service is not configured or unavailable.",
                            "details": {}
                        }
                    }
                },
                {
                    "summary": "Internal Server Error (500)",
                    "value": {
                        "success": False,
                        "error": {
                            "code": "INTERNAL_SERVER_ERROR",
                            "message": "An unexpected error occurred while processing the request. Please try again.",
                            "details": {}
                        }
                    }
                }
            ]
        }
    )
