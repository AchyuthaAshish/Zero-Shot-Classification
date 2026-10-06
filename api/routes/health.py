"""Health and System Status Router for FastAPI Backend.

Provides:
- GET /health: Lightweight liveness probe returning HTTP 200, service name, and version
  without initiating heavy ML inference, external LLM calls, or database queries.
- GET /api/v1/status: Detailed system diagnostics reporting readiness for application,
  classifier artifacts, database connectivity, and external provider configurations.
- Deterministic overall health status aggregation (HEALTHY, DEGRADED, UNHEALTHY).
- Strict credential and stack trace sanitization across all diagnostic probes.
"""

import os
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends

from api.config import APISettings, get_api_settings
from api.dependencies import get_classifier_service
from api.schemas.health import (
    HealthStatus,
    SubsystemCheck,
    HealthChecks,
    StatusData,
    StatusResponse,
)
from config.settings import get_settings
from ml.config import EMBEDDING_MODEL_DIR, TFIDF_SVM_MODEL_DIR

from api.schemas.errors import ErrorResponse

logger = logging.getLogger("api.routes.health")

# Lightweight liveness router mounted at root level (/health)
router = APIRouter(tags=["System"])

# Detailed status router mounted under API prefix (/api/v1/status)
status_router = APIRouter(tags=["Status"])


# -----------------------------------------------------------------------------
# Diagnostic Health Probe Functions
# -----------------------------------------------------------------------------

def check_application_liveness() -> SubsystemCheck:
    """Verifies that the FastAPI process and runtime are operational."""
    return SubsystemCheck(
        status=HealthStatus.HEALTHY,
        message="API process is operational",
        details={"process": "operational"}
    )


def check_classifier_readiness(classifier_service: Any = None) -> SubsystemCheck:
    """Lightweight readiness verification for local ML classifier.

    Inspects model artifacts on disk and local classifier instance readiness
    WITHOUT executing classification inference, external LLM calls, or repeated model loads.
    """
    has_embedding_model = (
        (EMBEDDING_MODEL_DIR / "calibrated_classifier.joblib").is_file()
        or (EMBEDDING_MODEL_DIR / "classifier.joblib").is_file()
    )
    has_tfidf_model = (TFIDF_SVM_MODEL_DIR / "model.joblib").is_file()

    if not has_embedding_model and not has_tfidf_model:
        logger.warning("Local classifier artifacts missing from models directory.")
        return SubsystemCheck(
            status=HealthStatus.UNHEALTHY,
            message="Local classifier is not ready",
            details={"local_model_available": False}
        )

    try:
        if classifier_service is not None:
            local_clf = classifier_service.local_classifier
        else:
            from ml.local_classifier import get_local_classifier
            local_clf = get_local_classifier()

        if local_clf is not None and getattr(local_clf, "model", None) is not None:
            return SubsystemCheck(
                status=HealthStatus.HEALTHY,
                message="Local classifier is ready",
                details={
                    "local_model_available": True,
                    "model_type": getattr(local_clf, "model_type", "multilingual_embedding"),
                    "is_calibrated": getattr(local_clf, "is_calibrated", False),
                    "calibration_method": getattr(local_clf, "calibration_method", None)
                }
            )
        else:
            return SubsystemCheck(
                status=HealthStatus.UNHEALTHY,
                message="Local classifier is not ready",
                details={"local_model_available": False}
            )
    except Exception as e:
        logger.warning(f"Error checking local classifier readiness: {e}")
        return SubsystemCheck(
            status=HealthStatus.UNHEALTHY,
            message="Local classifier is not ready",
            details={"local_model_available": False}
        )


def check_database_readiness_status() -> SubsystemCheck:
    """Checks Supabase database readiness using the repository abstraction.

    Executes a minimal read-only probe without modifying data or exposing credentials.
    """
    import persistence.repository as repository
    try:
        res = repository.check_database_readiness()
        status_str = res.get("status", "degraded")
        health_status = HealthStatus.HEALTHY if status_str == "healthy" else HealthStatus.DEGRADED
        return SubsystemCheck(
            status=health_status,
            message=res.get("message", "Database check completed"),
            details=res.get("details")
        )
    except Exception as e:
        logger.warning(f"Database readiness probe failed unexpectedly: {e}")
        return SubsystemCheck(
            status=HealthStatus.DEGRADED,
            message="Database connection is unavailable",
            details={"configured": False, "connected": False}
        )


def check_provider_readiness() -> SubsystemCheck:
    """Checks configured model provider status without initiating external requests or leaking keys."""
    settings = get_settings()
    mode = (os.getenv("CLASSIFICATION_MODE") or settings.classification_mode).lower()
    provider = getattr(settings, "llm_provider", "gemini").lower()
    has_key = bool(getattr(settings, "llm_api_key", None) and str(settings.llm_api_key).strip())

    if mode == "local":
        return SubsystemCheck(
            status=HealthStatus.HEALTHY,
            message="Local ML mode active; no external provider required",
            details={
                "mode": "local",
                "provider": "local"
            }
        )

    if mode == "hybrid":
        if has_key:
            return SubsystemCheck(
                status=HealthStatus.HEALTHY,
                message=f"Configured provider '{provider}' is available",
                details={
                    "mode": "hybrid",
                    "provider": provider,
                    "model": getattr(settings, "llm_model", "gemini-3.8-flash"),
                    "configured": True
                }
            )
        else:
            return SubsystemCheck(
                status=HealthStatus.DEGRADED,
                message=f"External provider '{provider}' is not configured; local ML fallback will be used",
                details={
                    "mode": "hybrid",
                    "provider": provider,
                    "configured": False
                }
            )

    # Pure external or gemini mode
    if has_key:
        return SubsystemCheck(
            status=HealthStatus.HEALTHY,
            message=f"Configured provider '{provider}' is available",
            details={
                "mode": mode,
                "provider": provider,
                "model": getattr(settings, "llm_model", "gemini-3.8-flash"),
                "configured": True
            }
        )
    else:
        return SubsystemCheck(
            status=HealthStatus.UNHEALTHY,
            message=f"External provider '{provider}' configuration is missing",
            details={
                "mode": mode,
                "provider": provider,
                "configured": False
            }
        )


def aggregate_overall_status(
    application: SubsystemCheck,
    classifier: SubsystemCheck,
    database: SubsystemCheck,
    providers: SubsystemCheck,
    mode: str
) -> HealthStatus:
    """Deterministically aggregates individual subsystem checks into overall system status.

    Semantic Rules:
    - UNHEALTHY:
      - Core application runtime is unhealthy.
      - Required local classifier is unhealthy when mode is 'local' or 'hybrid'.
      - External provider is unhealthy when mode is pure 'gemini' or 'external'.
    - DEGRADED:
      - Any non-critical dependency is unavailable (e.g. database is unavailable/unconfigured,
        or provider is unconfigured in hybrid mode where local classifier can still serve requests).
    - HEALTHY:
      - All required and optional subsystems report healthy.
    """
    # 1. Critical core components failure -> UNHEALTHY
    if application.status == HealthStatus.UNHEALTHY:
        return HealthStatus.UNHEALTHY

    if mode in ("local", "hybrid") and classifier.status == HealthStatus.UNHEALTHY:
        return HealthStatus.UNHEALTHY

    if mode in ("gemini", "external") and providers.status == HealthStatus.UNHEALTHY:
        return HealthStatus.UNHEALTHY

    # 2. Non-critical dependency failure -> DEGRADED
    if (
        database.status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)
        or providers.status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)
        or classifier.status == HealthStatus.DEGRADED
        or application.status == HealthStatus.DEGRADED
    ):
        return HealthStatus.DEGRADED

    # 3. All operational -> HEALTHY
    return HealthStatus.HEALTHY


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@router.get(
    "/health",
    summary="Service Health Check",
    description="Returns lightweight service health status, service name, and version without initiating heavy ML inference or database calls.",
    response_model=Dict[str, Any],
    responses={
        200: {
            "description": "Liveness probe succeeded.",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "service": "industrial-defect-intelligence-api",
                        "version": "1.0.0"
                    }
                }
            }
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal server error during health probe."
        }
    }
)
def get_health(settings: APISettings = Depends(get_api_settings)) -> Dict[str, Any]:
    """Lightweight health check endpoint for monitoring, container orchestrators, and liveness probes."""
    return {
        "status": "healthy",
        "service": "industrial-defect-intelligence-api",
        "version": settings.version
    }


@status_router.get(
    "/status",
    summary="System Status & Engine Diagnostics",
    description=(
        "Returns comprehensive system readiness diagnostics covering the core application runtime, "
        "local ML classifier artifacts, persistent Supabase database connectivity, and external model provider configuration.\n\n"
        "**Overall Health Semantics:**\n"
        "- `healthy`: All critical subsystems and optional components are fully operational.\n"
        "- `degraded`: Non-critical dependencies (e.g. database or external cloud provider in hybrid mode) are unavailable, "
        "while local ML classification continues to serve requests.\n"
        "- `unhealthy`: Core application or required classification engine is inoperable."
    ),
    response_model=StatusResponse,
    responses={
        200: {
            "description": "System readiness diagnostics returned successfully.",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": {
                            "status": "healthy",
                            "service": "industrial-defect-intelligence-api",
                            "version": "1.0.0",
                            "checks": {
                                "application": {
                                    "status": "healthy",
                                    "message": "API process is operational",
                                    "details": {"process": "operational"}
                                },
                                "classifier": {
                                    "status": "healthy",
                                    "message": "Local classifier is ready",
                                    "details": {
                                        "local_model_available": True,
                                        "model_type": "multilingual_embedding",
                                        "is_calibrated": True,
                                        "calibration_method": "Sigmoid / Temperature Scaling"
                                    }
                                },
                                "database": {
                                    "status": "healthy",
                                    "message": "Database connection verified",
                                    "details": {"configured": True, "connected": True}
                                },
                                "providers": {
                                    "status": "healthy",
                                    "message": "Local ML mode active; no external provider required",
                                    "details": {"mode": "local", "provider": "local"}
                                }
                            }
                        }
                    }
                }
            }
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal server error during engine diagnostics."
        }
    }
)
def get_status(
    settings: APISettings = Depends(get_api_settings),
    classifier_service: Any = Depends(get_classifier_service)
) -> StatusResponse:
    """Detailed technical diagnostics endpoint for frontend, monitoring systems, and operators."""
    app_settings = get_settings()
    mode = (os.getenv("CLASSIFICATION_MODE") or app_settings.classification_mode).lower()

    app_check = check_application_liveness()
    classifier_check = check_classifier_readiness(classifier_service)
    db_check = check_database_readiness_status()
    provider_check = check_provider_readiness()

    overall = aggregate_overall_status(
        application=app_check,
        classifier=classifier_check,
        database=db_check,
        providers=provider_check,
        mode=mode
    )

    return StatusResponse(
        success=True,
        data=StatusData(
            status=overall,
            service="industrial-defect-intelligence-api",
            version=settings.version,
            checks=HealthChecks(
                application=app_check,
                classifier=classifier_check,
                database=db_check,
                providers=provider_check
            )
        )
    )
