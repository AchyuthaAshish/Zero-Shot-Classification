"""FastAPI Application Entrypoint for Industrial Defect Intelligence API.

Provides:
- Application initialization with comprehensive metadata
- Environment-aware CORS configuration
- Root service information endpoint (GET /)
- Health check probe (GET /health)
- Standardized, sanitized error handling preventing credential/traceback leaks
"""

import logging
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import get_api_settings, APISettings
from api.routes.health import router as health_router, status_router
from api.routes.classification import router as classification_router
from api.routes.history import router as history_router
from api.routes.auth import router as auth_router
from api.errors import APIError
from api.schemas.errors import ErrorCode

logger = logging.getLogger("api.main")

# OpenAPI tags documentation structure conforming to Step 3.6 API contract
OPENAPI_TAGS = [
    {
        "name": "System",
        "description": "Core system service information and lightweight health check probes."
    },
    {
        "name": "Status",
        "description": "Comprehensive system readiness diagnostics and engine health checks."
    },
    {
        "name": "Authentication",
        "description": "Supabase authentication verification and user identity endpoints."
    },
    {
        "name": "Classification",
        "description": "Single-defect and multi-defect machine fault classification endpoints."
    },
    {
        "name": "History",
        "description": "Historical defect report retrieval and child segment querying from persistent audit store."
    }
]


def create_app(settings: APISettings = None) -> FastAPI:
    """Application factory creating the configured FastAPI instance."""
    cfg = settings or get_api_settings()

    app = FastAPI(
        title=cfg.title,
        description=cfg.description,
        version=cfg.version,
        docs_url=cfg.docs_url,
        redoc_url=cfg.redoc_url,
        openapi_url=cfg.openapi_url,
        openapi_tags=OPENAPI_TAGS
    )

    # -------------------------------------------------------------------------
    # CORS Middleware Configuration (Step 3.1 Section 7)
    # -------------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # -------------------------------------------------------------------------
    # Error Handling Foundation (Step 3.1 Section 8 & Step 3.5 Normalization)
    # -------------------------------------------------------------------------
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        """Handler for structured domain application errors."""
        return JSONResponse(
            status_code=exc.status_code,
            headers=exc.headers,
            content={
                "success": False,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details
                }
            }
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Standardized handler for known HTTP exceptions."""
        if hasattr(exc, "code"):
            code = exc.code
            message = str(exc.detail)
            details = getattr(exc, "details", {})
        elif isinstance(exc.detail, dict):
            code = exc.detail.get("code", ErrorCode.HTTP_ERROR.value)
            message = exc.detail.get("message", "An error occurred.")
            details = exc.detail.get("details", {})
        elif exc.status_code == 404 and exc.detail == "Requested defect record not found.":
            code = ErrorCode.HTTP_ERROR.value
            message = str(exc.detail)
            details = {}
        elif exc.status_code == 404:
            code = ErrorCode.NOT_FOUND.value
            message = str(exc.detail)
            details = {}
        elif exc.status_code == 503 and ("database" in str(exc.detail).lower() or "persistence" in str(exc.detail).lower()):
            code = ErrorCode.DATABASE_ERROR.value
            message = str(exc.detail)
            details = {}
        elif exc.status_code in (502, 503) and ("provider" in str(exc.detail).lower() or "external" in str(exc.detail).lower() or "gemini" in str(exc.detail).lower()):
            code = ErrorCode.PROVIDER_ERROR.value
            message = str(exc.detail)
            details = {}
        elif exc.status_code == 503:
            code = ErrorCode.SERVICE_UNAVAILABLE.value
            message = str(exc.detail)
            details = {}
        elif exc.status_code == 502:
            code = ErrorCode.PROVIDER_ERROR.value
            message = str(exc.detail)
            details = {}
        elif exc.status_code == 401:
            code = ErrorCode.UNAUTHORIZED.value
            message = str(exc.detail)
            details = {}
        elif exc.status_code == 422:
            code = ErrorCode.VALIDATION_ERROR.value
            message = str(exc.detail)
            details = {}
        elif exc.status_code == 500:
            code = ErrorCode.INTERNAL_SERVER_ERROR.value
            message = str(exc.detail)
            details = {}
        else:
            code = ErrorCode.HTTP_ERROR.value
            message = str(exc.detail)
            details = {}

        return JSONResponse(
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
            content={
                "success": False,
                "error": {
                    "code": code,
                    "message": message,
                    "details": details
                }
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Standardized handler for request validation errors."""
        clean_errors = []
        field_errors = []
        for err in exc.errors():
            loc = " -> ".join(str(l) for l in err.get("loc", []))
            msg = err.get("msg", "Invalid parameter.")
            clean_errors.append(f"{loc}: {msg}" if loc else msg)
            field_errors.append({
                "field": loc or None,
                "message": msg,
                "type": err.get("type", "value_error")
            })

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": {
                    "code": ErrorCode.VALIDATION_ERROR.value,
                    "message": "; ".join(clean_errors) or "Request validation failed.",
                    "details": {
                        "fields": field_errors
                    }
                }
            }
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Global fallback error handler preventing disclosure of secrets, keys, or stack traces."""
        from database.supabase_client import _sanitize_error
        sanitized_log = _sanitize_error(str(exc))
        logger.error(f"Unhandled server error on {request.method} {request.url.path}: {sanitized_log}", exc_info=True)

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": ErrorCode.INTERNAL_SERVER_ERROR.value,
                    "message": "An unexpected error occurred while processing the request. Please try again.",
                    "details": {}
                }
            }
        )

    # -------------------------------------------------------------------------
    # Root / Service Information Endpoint (Step 3.1 Section 5)
    # -------------------------------------------------------------------------
    @app.get(
        "/",
        tags=["System"],
        summary="API Service Information",
        description="Returns API service name, version, and documentation links.",
        response_model=Dict[str, Any]
    )
    def root_info() -> Dict[str, Any]:
        """Root endpoint identifying the API service and directing operators to documentation."""
        return {
            "service": cfg.title,
            "version": cfg.version,
            "docs": cfg.docs_url,
            "health": "/health",
            "status": f"{cfg.api_prefix}/status"
        }

    # -------------------------------------------------------------------------
    # Mount Route Handlers
    # -------------------------------------------------------------------------
    app.include_router(health_router)
    app.include_router(status_router, prefix=cfg.api_prefix)
    app.include_router(auth_router, prefix=cfg.api_prefix)
    app.include_router(classification_router, prefix=cfg.api_prefix)
    app.include_router(history_router, prefix=cfg.api_prefix)

    return app


# Singleton application instance for ASGI servers (e.g. uvicorn api.main:app)
app = create_app()
