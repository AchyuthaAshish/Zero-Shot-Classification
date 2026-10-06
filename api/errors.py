"""API Exceptions and Custom Error Types for FastAPI Backend.

Provides:
- APIError base exception mapping directly to the standard error response envelope
- Specialized domain API exceptions (NotFoundError, ValidationErrorAPI, ProviderError, DatabaseErrorAPI)
- Clean decoupling of route logic from raw HTTP status handling
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from api.schemas.errors import ErrorCode


class APIError(HTTPException):
    """Base API exception that maps directly into the standard error response envelope."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        super().__init__(status_code=status_code, detail=message, headers=headers)
        self.code = code
        self.message = message
        self.details = details or {}


class NotFoundError(APIError):
    """Raised when a requested resource (e.g. defect report) cannot be found."""

    def __init__(
        self,
        message: str = "Resource not found",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            code=ErrorCode.NOT_FOUND.value,
            message=message,
            details=details
        )


class UnauthorizedError(APIError):
    """Raised when authentication credentials are missing, invalid, or expired."""

    def __init__(
        self,
        message: str = "Authentication required.",
        details: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        auth_headers = {"WWW-Authenticate": "Bearer"}
        if headers:
            auth_headers.update(headers)
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            code=ErrorCode.UNAUTHORIZED.value,
            message=message,
            details=details,
            headers=auth_headers
        )


class ForbiddenError(APIError):
    """Raised when an authenticated user lacks sufficient application privileges."""

    def __init__(
        self,
        message: str = "Access forbidden. Insufficient permissions.",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            code=ErrorCode.FORBIDDEN.value,
            message=message,
            details=details
        )


class ValidationErrorAPI(APIError):
    """Raised when request validation fails outside standard Pydantic parsing."""

    def __init__(
        self,
        message: str = "Request validation failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code=ErrorCode.VALIDATION_ERROR.value,
            message=message,
            details=details
        )


class ProviderError(APIError):
    """Raised when an external ML or LLM provider fails, times out, or is misconfigured."""

    def __init__(
        self,
        message: str = "Classification provider is unavailable",
        status_code: int = status.HTTP_502_BAD_GATEWAY,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status_code,
            code=ErrorCode.PROVIDER_ERROR.value,
            message=message,
            details=details
        )


class DatabaseErrorAPI(APIError):
    """Raised when database or persistence operations fail or are unconfigured."""

    def __init__(
        self,
        message: str = "Database service is currently unavailable",
        status_code: int = status.HTTP_503_SERVICE_UNAVAILABLE,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status_code,
            code=ErrorCode.DATABASE_ERROR.value,
            message=message,
            details=details
        )


class ServiceUnavailableError(APIError):
    """Raised when a generic upstream or internal subsystem is unavailable."""

    def __init__(
        self,
        message: str = "Service is temporarily unavailable",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code=ErrorCode.SERVICE_UNAVAILABLE.value,
            message=message,
            details=details
        )
