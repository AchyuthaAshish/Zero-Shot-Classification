"""Status router module for Industrial Defect Intelligence Backend.

Re-exports status_router from api.routes.health under canonical router name.
"""

from api.routes.health import status_router as router

__all__ = ["router"]
