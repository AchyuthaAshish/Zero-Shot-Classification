"""Unit and Integration Tests for Step 3.1 FastAPI Backend Foundation.

Verifies:
A. FastAPI application imports successfully with proper metadata.
B. GET / returns expected API information and links.
C. GET /health returns HTTP 200.
D. Health response contains status "healthy" and service identifier.
E. API version is present and equals "1.0.0".
F. /docs is available with HTTP 200.
G. /openapi.json is available with HTTP 200 and valid schema.
H. Error handling does not expose secrets, keys, or internal stack traces.
I. CORS configuration loads correctly without unrestricted wildcards.
J. /redoc is available with HTTP 200.
K. Health endpoint does not invoke heavy ML inference or external services.
L. Protected datasets (600 training rows, 93 evaluation cases) remain strictly untouched.
"""

import json
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from fastapi.testclient import TestClient
from fastapi import FastAPI, HTTPException

from api.main import app, create_app
from api.config import get_api_settings, load_api_settings, APISettings, DEFAULT_DEV_CORS_ORIGINS
from ml.config import TRAINING_CSV_PATH, EVALUATION_DATASET_PATH


class TestFastAPIFoundation(unittest.TestCase):
    """Test suite for Step 3.1 FastAPI Foundation."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)
        cls.settings = get_api_settings()

    # -------------------------------------------------------------------------
    # Test A: Application Imports and Metadata
    # -------------------------------------------------------------------------
    def test_fastapi_app_metadata(self):
        """A. Verifies the FastAPI application initializes with meaningful metadata."""
        self.assertIsInstance(app, FastAPI)
        self.assertEqual(app.title, "Industrial Defect Intelligence API")
        self.assertIn("defect classification", app.description.lower())
        self.assertEqual(app.version, "1.0.0")

    # -------------------------------------------------------------------------
    # Test B: Root Endpoint (GET /)
    # -------------------------------------------------------------------------
    def test_root_endpoint_returns_info(self):
        """B. Verifies GET / returns HTTP 200 and API service information."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("service", data)
        self.assertEqual(data["service"], "Industrial Defect Intelligence API")
        self.assertIn("version", data)
        self.assertEqual(data["version"], "1.0.0")
        self.assertIn("docs", data)
        self.assertEqual(data["docs"], "/docs")
        self.assertIn("health", data)
        self.assertEqual(data["health"], "/health")

    # -------------------------------------------------------------------------
    # Test C, D, E: Health Check Endpoint (GET /health)
    # -------------------------------------------------------------------------
    def test_health_endpoint_success(self):
        """C & D & E. Verifies GET /health returns HTTP 200, status healthy, and version."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("service"), "industrial-defect-intelligence-api")
        self.assertEqual(data.get("version"), "1.0.0")

    # -------------------------------------------------------------------------
    # Test F: API Documentation (/docs)
    # -------------------------------------------------------------------------
    def test_docs_endpoint_available(self):
        """F. Verifies FastAPI Swagger documentation (/docs) is reachable."""
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200)
        self.assertIn("html", response.headers.get("content-type", "").lower())

    # -------------------------------------------------------------------------
    # Test G: OpenAPI Schema (/openapi.json)
    # -------------------------------------------------------------------------
    def test_openapi_schema_available(self):
        """G. Verifies /openapi.json returns valid schema with registered paths."""
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)

        schema = response.json()
        self.assertIn("openapi", schema)
        self.assertEqual(schema["info"]["title"], "Industrial Defect Intelligence API")
        self.assertEqual(schema["info"]["version"], "1.0.0")

        paths = schema.get("paths", {})
        self.assertIn("/", paths)
        self.assertIn("/health", paths)

    # -------------------------------------------------------------------------
    # Test H: Error Handling & Secret Protection
    # -------------------------------------------------------------------------
    def test_unhandled_exception_does_not_expose_secrets_or_traces(self):
        """H. Verifies unhandled exceptions return sanitized JSON without stack traces or secret keys."""
        # Create a test app with a route that deliberately raises an exception containing secret-like text
        test_app = create_app()

        @test_app.get("/test-error")
        def route_with_error():
            secret_leak = "SECRET_API_KEY=AIzaSyFakeKey12345 Database password=postgres"
            raise RuntimeError(f"Database crash with {secret_leak}")

        test_client = TestClient(test_app, raise_server_exceptions=False)
        response = test_client.get("/test-error")

        self.assertEqual(response.status_code, 500)
        data = response.json()

        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "INTERNAL_SERVER_ERROR")

        # Crucially verify no secrets or internal tracebacks appear in response text
        body_text = response.text
        self.assertNotIn("AIzaSyFakeKey", body_text)
        self.assertNotIn("Traceback", body_text)
        self.assertNotIn("postgres", body_text)
        self.assertNotIn("RuntimeError", body_text)

    def test_http_exception_handling(self):
        """H2. Verifies known HTTPExceptions return structured error responses."""
        test_app = create_app()

        @test_app.get("/test-not-found")
        def route_not_found():
            raise HTTPException(status_code=404, detail="Requested defect record not found.")

        test_client = TestClient(test_app, raise_server_exceptions=False)
        response = test_client.get("/test-not-found")

        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertEqual(data["error"]["code"], "HTTP_ERROR")
        self.assertEqual(data["error"]["message"], "Requested defect record not found.")

    # -------------------------------------------------------------------------
    # Test I: CORS Configuration
    # -------------------------------------------------------------------------
    def test_cors_configuration(self):
        """I. Verifies CORS middleware is configured with development origins and avoids wildcards."""
        settings = self.settings
        self.assertIsInstance(settings.cors_origins, list)
        self.assertTrue(len(settings.cors_origins) > 0)
        self.assertNotIn("*", settings.cors_origins)

        # Check expected development origins are in default CORS list
        for expected in ["http://localhost:3000", "http://localhost:5173", "http://localhost:8501"]:
            self.assertIn(expected, settings.cors_origins)

        # Test CORS preflight OPTIONS request
        response = self.client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET"
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:5173")

    # -------------------------------------------------------------------------
    # Test J: ReDoc Documentation (/redoc)
    # -------------------------------------------------------------------------
    def test_redoc_endpoint_available(self):
        """J. Verifies ReDoc documentation (/redoc) is reachable."""
        response = self.client.get("/redoc")
        self.assertEqual(response.status_code, 200)
        self.assertIn("html", response.headers.get("content-type", "").lower())

    # -------------------------------------------------------------------------
    # Test K: Health Endpoint Is Lightweight
    # -------------------------------------------------------------------------
    @patch("classification.classifier.classify_defect")
    @patch("persistence.repository.get_defect_reports")
    def test_health_check_does_not_invoke_heavy_services(self, mock_reports, mock_classify):
        """K. Verifies GET /health does not trigger ML inference or Supabase queries."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        # Ensure no ML inference or database calls were initiated
        self.assertFalse(mock_classify.called, "Health check must not invoke classify_defect.")
        self.assertFalse(mock_reports.called, "Health check must not query Supabase reports.")

    # -------------------------------------------------------------------------
    # Test L: Dataset Integrity (Protected 600 / 93 Datasets)
    # -------------------------------------------------------------------------
    def test_protected_datasets_integrity(self):
        """L. Verifies protected training (600) and evaluation (93) datasets remain strictly untouched."""
        self.assertTrue(TRAINING_CSV_PATH.exists(), "Training dataset must exist.")
        df = pd.read_csv(TRAINING_CSV_PATH)
        self.assertEqual(len(df), 600, "Training dataset must contain exactly 600 rows.")

        self.assertTrue(EVALUATION_DATASET_PATH.exists(), "Evaluation dataset must exist.")
        with open(EVALUATION_DATASET_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)
        self.assertEqual(len(cases), 93, "Evaluation dataset must contain exactly 93 cases.")


if __name__ == "__main__":
    unittest.main()
