"""API Route Handlers Package."""

from api.routes.health import router as health_router, status_router
from api.routes.classification import router as classification_router
from api.routes.history import router as history_router
from api.routes.auth import router as auth_router

__all__ = ["health_router", "status_router", "classification_router", "history_router", "auth_router"]

