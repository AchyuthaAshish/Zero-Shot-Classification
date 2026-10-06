"""API Configuration for Industrial Defect Intelligence FastAPI Backend.

Reuses central system settings from config.settings while defining
API-specific parameters (CORS origins, documentation, versioning, environment).
Strictly prevents disclosure of secrets and sensitive credentials.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from config.settings import get_settings, load_env_file

# Default safe development origins (Vite/React dev servers and Streamlit UI)
DEFAULT_DEV_CORS_ORIGINS: List[str] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8501",
    "http://127.0.0.1:8501"
]


def _parse_cors_origins() -> List[str]:
    """Parses CORS_ORIGINS from environment, falling back to safe local dev origins."""
    raw = os.getenv("CORS_ORIGINS")
    if not raw or not raw.strip():
        return list(DEFAULT_DEV_CORS_ORIGINS)

    # Allow comma-separated values (e.g. "http://localhost:5173,http://localhost:3000")
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins if origins else list(DEFAULT_DEV_CORS_ORIGINS)


API_DESCRIPTION: str = (
    "Production-grade REST API backend for automated industrial machine defect classification, "
    "multi-defect co-occurrence segmentation, audit trail logging, and system diagnostics.\n\n"
    "### Core Capabilities:\n"
    "- **Defect Classification**: Single-defect and co-occurring multi-defect classification across "
    "the approved 8-category industrial taxonomy using Local ML (multilingual embeddings / calibrated SVM), "
    "Gemini zero-shot, or Hybrid modes.\n"
    "- **Confidence & Ambiguity**: Calibrated statistical probabilities, reliability tiers, and "
    "competing hypothesis ambiguity assessments.\n"
    "- **Multilingual Support**: High-fidelity classification of English, native Telugu script, and "
    "Telugu-English code-switched defect reports.\n"
    "- **Audit Trail History**: Query historical defect reports and child segment records with category "
    "and employee filtering, and pagination.\n"
    "- **System Diagnostics**: Lightweight liveness probe and diagnostic engine readiness checks.\n"
    "- **Standardized Contracts**: Unified success response envelopes and sanitized error responses."
)


@dataclass(frozen=True)
class APISettings:
    """Centralized API configuration parameters."""
    title: str = "Industrial Defect Intelligence API"
    description: str = API_DESCRIPTION
    version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    environment: str = "development"
    cors_origins: List[str] = field(default_factory=list)
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"


def load_api_settings() -> APISettings:
    """Loads API configuration, ensuring .env is populated."""
    load_env_file()
    # Read environment
    env = os.getenv("ENVIRONMENT", os.getenv("APP_ENV", "development")).strip().lower()
    prefix = os.getenv("API_PREFIX", "/api/v1").strip()
    cors_origins = _parse_cors_origins()

    return APISettings(
        title="Industrial Defect Intelligence API",
        description=API_DESCRIPTION,
        version="1.0.0",
        api_prefix=prefix if prefix.startswith("/") else f"/{prefix}",
        environment=env,
        cors_origins=cors_origins,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )


_api_settings: Optional[APISettings] = None


def get_api_settings() -> APISettings:
    """Returns the singleton APISettings instance."""
    global _api_settings
    if _api_settings is None:
        _api_settings = load_api_settings()
    return _api_settings


def reset_api_settings() -> None:
    """Resets the singleton APISettings instance (useful in unit tests)."""
    global _api_settings
    _api_settings = None
