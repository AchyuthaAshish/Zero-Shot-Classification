"""Pydantic schemas for Health & Diagnostics API.

Conforms to Phase 3 Step 3.4:
- Standardized, stable response models for system status and readiness diagnostics
- Safe subsystem check representations for application, classifier, database, and providers
- Explicit prevention of credential, secret, URL, or internal stack trace leakage
"""

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class HealthStatus(str, Enum):
    """Enumeration of system and component health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class SubsystemCheck(BaseModel):
    """Readiness and health status for an individual subsystem or dependency."""
    status: HealthStatus = Field(
        ...,
        description="Operational status of the subsystem: healthy, degraded, or unhealthy."
    )
    message: str = Field(
        ...,
        description="Sanitized summary message describing component state."
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Sanitized diagnostic metadata and readiness flags."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "message": "Local classifier is ready",
                "details": {
                    "local_model_available": True
                }
            }
        }
    )


class HealthChecks(BaseModel):
    """Collection of component readiness and health checks."""
    application: SubsystemCheck = Field(
        ...,
        description="Core FastAPI application process and runtime liveness."
    )
    classifier: SubsystemCheck = Field(
        ...,
        description="Local defect classifier and ML model artifact readiness."
    )
    database: SubsystemCheck = Field(
        ...,
        description="Persistence layer and database connectivity readiness."
    )
    providers: SubsystemCheck = Field(
        ...,
        description="External model provider configuration status."
    )


class StatusData(BaseModel):
    """Detailed system health payload."""
    status: HealthStatus = Field(
        ...,
        description="Aggregated technical system health status."
    )
    service: str = Field(
        ...,
        description="Service identifier string."
    )
    version: str = Field(
        ...,
        description="API semantic version string."
    )
    checks: HealthChecks = Field(
        ...,
        description="Individual subsystem and dependency health checks."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "service": "industrial-defect-intelligence-api",
                "version": "1.0.0",
                "checks": {
                    "application": {
                        "status": "healthy",
                        "message": "API process is operational"
                    },
                    "classifier": {
                        "status": "healthy",
                        "message": "Local classifier is ready",
                        "details": {
                            "local_model_available": True
                        }
                    },
                    "database": {
                        "status": "healthy",
                        "message": "Database connection is available"
                    },
                    "providers": {
                        "status": "healthy",
                        "message": "Configured provider is available"
                    }
                }
            }
        }
    )


class StatusResponse(BaseModel):
    """Envelope response model for GET /api/v1/status."""
    success: bool = Field(
        default=True,
        description="Indicates whether the status diagnostics executed successfully."
    )
    data: StatusData = Field(
        ...,
        description="Detailed system health and engine diagnostics."
    )
