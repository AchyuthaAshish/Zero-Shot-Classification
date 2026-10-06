"""Industrial Defect Intelligence API Backend.

FastAPI REST backend serving industrial defect classification, multi-defect analysis,
and persistent defect history.
"""

from api.main import app, create_app

__version__ = "1.0.0"
__all__ = ["app", "create_app", "__version__"]
