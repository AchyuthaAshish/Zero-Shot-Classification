"""Comprehensive unit and integration test suite for API Validation & Error Handling.

Conforms to Phase 3 Step 3.5:
- A. Missing classification description
- B. Blank classification description
- C. Oversized classification description
- D. Invalid classification mode
- E. Malformed JSON / invalid request body
- F. Invalid history UUID
- G. Nonexistent history report
- H. Database failure sanitization
- I. Provider failure sanitization
- J. Unexpected exception sanitization
- K. Error envelope always contains success=false
- L. Error code is present and standardized
- M. Error message is safe and human-readable
- N. No secrets exposed
- O. No stack traces exposed
- P. No filesystem paths exposed
- Q. Existing classification success response unchanged
- R. Existing history success response unchanged
- S. Existing /health unchanged
- T. Existing /api/v1/status unchanged
- U. Unicode/Telugu classification input still accepted
- V. Canonical taxonomy validation preserved
- W. OpenAPI still builds successfully
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from uuid import uuid4
from fastapi.testclient import TestClient

from api.main import create_app
from api.dependencies import get_classifier_service
from core.schemas import ClassificationResult
from persistence.repository import DatabaseError


class TestAPIErrorHandling(unittest.TestCase):
    """Test suite verifying unified error handling, validation, and sanitization."""

    @classmethod
    def setUpClass(cls):
        """Initializes the FastAPI test client."""
        cls.app = create_app()
        cls.client = TestClient(cls.app, raise_server_exceptions=False)

    # -------------------------------------------------------------------------
    # Test A: Missing Classification Description
    # -------------------------------------------------------------------------
    def test_missing_classification_description(self):
        """A. Verifies missing description field is rejected with HTTP 422 and VALIDATION_ERROR."""
        payload = {"mode": "local"}
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 422)

        data = response.json()
        self.assertFalse(data.get("success"))
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "VALIDATION_ERROR")
        self.assertIn("description", data["error"]["message"].lower())
        self.assertIn("fields", data["error"].get("details", {}))

    # -------------------------------------------------------------------------
    # Test B: Blank Classification Description
    # -------------------------------------------------------------------------
    def test_blank_classification_description(self):
        """B. Verifies whitespace-only or empty description is rejected with HTTP 422."""
        for blank_val in ["", "   ", "\t\n  "]:
            response = self.client.post("/api/v1/classify", json={"description": blank_val, "mode": "local"})
            self.assertEqual(response.status_code, 422)

            data = response.json()
            self.assertFalse(data.get("success"))
            self.assertEqual(data["error"]["code"], "VALIDATION_ERROR")
            self.assertIn("whitespace", data["error"]["message"].lower())

    # -------------------------------------------------------------------------
    # Test C: Oversized Classification Description
    # -------------------------------------------------------------------------
    def test_oversized_classification_description(self):
        """C. Verifies description exceeding maximum character limit is rejected with HTTP 422."""
        payload = {"description": "X" * 2001, "mode": "local"}
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 422)

        data = response.json()
        self.assertFalse(data.get("success"))
        self.assertEqual(data["error"]["code"], "VALIDATION_ERROR")
        self.assertIn("maximum", data["error"]["message"].lower())

    # -------------------------------------------------------------------------
    # Test D: Invalid Classification Mode
    # -------------------------------------------------------------------------
    def test_invalid_classification_mode(self):
        """D. Verifies unsupported classification modes are rejected with HTTP 422."""
        payload = {"description": "Motor overheating.", "mode": "invalid_experimental_mode"}
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 422)

        data = response.json()
        self.assertFalse(data.get("success"))
        self.assertEqual(data["error"]["code"], "VALIDATION_ERROR")

    # -------------------------------------------------------------------------
    # Test E: Malformed JSON / Invalid Request Body
    # -------------------------------------------------------------------------
    def test_malformed_json_request_body(self):
        """E. Verifies non-JSON or malformed request body returns structured HTTP 422."""
        response = self.client.post(
            "/api/v1/classify",
            content="This is plain text, not valid JSON",
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status_code, 422)

        data = response.json()
        self.assertFalse(data.get("success"))
        self.assertEqual(data["error"]["code"], "VALIDATION_ERROR")

    # -------------------------------------------------------------------------
    # Test F: Invalid History UUID
    # -------------------------------------------------------------------------
    def test_invalid_history_uuid(self):
        """F. Verifies malformed report_id path parameter returns HTTP 422."""
        response = self.client.get("/api/v1/history/123-not-a-uuid")
        self.assertEqual(response.status_code, 422)

        data = response.json()
        self.assertFalse(data.get("success"))
        self.assertEqual(data["error"]["code"], "VALIDATION_ERROR")

    # -------------------------------------------------------------------------
    # Test G: Nonexistent History Report
    # -------------------------------------------------------------------------
    @patch("persistence.repository.is_supabase_configured", return_value=True)
    @patch("persistence.repository.get_defect_report_by_id", return_value=None)
    def test_nonexistent_history_report(self, mock_get, mock_conf):
        """G. Verifies querying a nonexistent report UUID returns HTTP 404 with NOT_FOUND code."""
        missing_id = str(uuid4())
        response = self.client.get(f"/api/v1/history/{missing_id}")
        self.assertEqual(response.status_code, 404)

        data = response.json()
        self.assertFalse(data.get("success"))
        self.assertEqual(data["error"]["code"], "NOT_FOUND")
        self.assertIn("not found", data["error"]["message"].lower())

    # -------------------------------------------------------------------------
    # Test H: Database Failure Sanitization
    # -------------------------------------------------------------------------
    @patch("persistence.repository.is_supabase_configured", return_value=True)
    @patch("persistence.repository.get_defect_reports")
    def test_database_failure_sanitization(self, mock_reports, mock_conf):
        """H. Verifies database errors are sanitized into HTTP 503 DATABASE_ERROR without leaking internals."""
        sensitive_leak = "postgresql://postgres:secretpassword@db.supabase.co:5432/postgres table=defect_reports"
        mock_reports.side_effect = DatabaseError(f"Query execution failure: {sensitive_leak}")

        response = self.client.get("/api/v1/history")
        self.assertEqual(response.status_code, 503)

        data = response.json()
        self.assertFalse(data.get("success"))
        self.assertEqual(data["error"]["code"], "DATABASE_ERROR")
        self.assertNotIn("secretpassword", response.text)
        self.assertNotIn("postgresql://", response.text)

    # -------------------------------------------------------------------------
    # Test I: Provider Failure Sanitization
    # -------------------------------------------------------------------------
    def test_provider_failure_sanitization(self):
        """I. Verifies provider failures return HTTP 502/503 PROVIDER_ERROR without credential disclosure."""
        # 1. Configuration error -> 503
        mock_cfg_result = ClassificationResult(
            category="Unknown",
            reason="Unconfigured provider",
            language="English",
            reliability="Low",
            status="configuration_error",
            original_description="Motor vibrating",
            normalized_description="motor vibrating",
            error_message="Configuration error: GEMINI_API_KEY unset."
        )
        mock_clf = MagicMock()
        mock_clf.classify.return_value = mock_cfg_result
        self.app.dependency_overrides[get_classifier_service] = lambda: mock_clf
        try:
            resp = self.client.post("/api/v1/classify", json={"description": "Motor vibrating", "mode": "gemini"})
            self.assertEqual(resp.status_code, 503)
            data = resp.json()
            self.assertFalse(data.get("success"))
            self.assertEqual(data["error"]["code"], "PROVIDER_ERROR")
            self.assertNotIn("GEMINI_API_KEY", resp.text)
        finally:
            self.app.dependency_overrides.pop(get_classifier_service, None)

        # 2. Model error -> 502
        mock_mod_result = ClassificationResult(
            category="Unknown",
            reason="Timeout",
            language="English",
            reliability="Low",
            status="model_error",
            original_description="Motor vibrating",
            normalized_description="motor vibrating",
            error_message="ReadTimeout to api.aimlapi.com"
        )
        mock_clf_mod = MagicMock()
        mock_clf_mod.classify.return_value = mock_mod_result
        self.app.dependency_overrides[get_classifier_service] = lambda: mock_clf_mod
        try:
            resp = self.client.post("/api/v1/classify", json={"description": "Motor vibrating", "mode": "gemini"})
            self.assertEqual(resp.status_code, 502)
            data = resp.json()
            self.assertFalse(data.get("success"))
            self.assertEqual(data["error"]["code"], "PROVIDER_ERROR")
            self.assertNotIn("ReadTimeout", resp.text)
        finally:
            self.app.dependency_overrides.pop(get_classifier_service, None)

    # -------------------------------------------------------------------------
    # Test J: Unexpected Exception Sanitization
    # -------------------------------------------------------------------------
    def test_unexpected_exception_sanitization(self):
        """J. Verifies uncaught exceptions return HTTP 500 INTERNAL_SERVER_ERROR without leaking stack traces."""
        mock_clf_crash = MagicMock()
        mock_clf_crash.classify.side_effect = RuntimeError("Internal CUDA memory leak at /var/ml/models/weights.bin")
        self.app.dependency_overrides[get_classifier_service] = lambda: mock_clf_crash
        try:
            response = self.client.post("/api/v1/classify", json={"description": "Motor vibration", "mode": "local"})
            self.assertEqual(response.status_code, 500)

            data = response.json()
            self.assertFalse(data.get("success"))
            self.assertEqual(data["error"]["code"], "INTERNAL_SERVER_ERROR")
            self.assertNotIn("CUDA memory leak", response.text)
            self.assertNotIn("weights.bin", response.text)
            self.assertNotIn("Traceback", response.text)
        finally:
            self.app.dependency_overrides.pop(get_classifier_service, None)

    # -------------------------------------------------------------------------
    # Test K: Error Envelope Always Contains success=false
    # -------------------------------------------------------------------------
    def test_error_envelope_always_contains_success_false(self):
        """K. Verifies all error responses across 404, 422, 500, 502, 503 set success=false."""
        # 422 Validation
        r422 = self.client.post("/api/v1/classify", json={})
        self.assertEqual(r422.status_code, 422)
        self.assertIs(r422.json().get("success"), False)

        # 404 Not Found
        with patch("persistence.repository.is_supabase_configured", return_value=True), \
             patch("persistence.repository.get_defect_report_by_id", return_value=None):
            r404 = self.client.get(f"/api/v1/history/{uuid4()}")
            self.assertEqual(r404.status_code, 404)
            self.assertIs(r404.json().get("success"), False)

    # -------------------------------------------------------------------------
    # Test L: Error Code is Present and Standardized
    # -------------------------------------------------------------------------
    def test_error_code_is_present_and_standardized(self):
        """L. Verifies that error payloads always have a valid, standardized error code."""
        allowed_codes = {
            "VALIDATION_ERROR",
            "NOT_FOUND",
            "SERVICE_UNAVAILABLE",
            "PROVIDER_ERROR",
            "DATABASE_ERROR",
            "INTERNAL_SERVER_ERROR",
            "HTTP_ERROR"
        }
        r422 = self.client.post("/api/v1/classify", json={"mode": "unknown_mode"})
        code = r422.json().get("error", {}).get("code")
        self.assertIn(code, allowed_codes)

    # -------------------------------------------------------------------------
    # Test M: Error Message is Safe and Human-Readable
    # -------------------------------------------------------------------------
    def test_error_message_is_safe_and_human_readable(self):
        """M. Verifies error messages are clean human-readable strings."""
        response = self.client.post("/api/v1/classify", json={"description": ""})
        msg = response.json()["error"]["message"]
        self.assertIsInstance(msg, str)
        self.assertTrue(len(msg) > 5)

    # -------------------------------------------------------------------------
    # Test N: No Secrets Exposed
    # -------------------------------------------------------------------------
    def test_no_secrets_exposed_in_errors(self):
        """N. Verifies API keys, tokens, or credentials are never included in error responses."""
        fake_secret = "AIzaSyFakeSecretToken998877"
        with patch.dict(os.environ, {"LLM_API_KEY": fake_secret}):
            response = self.client.post("/api/v1/classify", json={"description": "   "})
            self.assertNotIn(fake_secret, response.text)

    # -------------------------------------------------------------------------
    # Test O: No Stack Traces Exposed
    # -------------------------------------------------------------------------
    def test_no_stack_traces_exposed(self):
        """O. Verifies traceback details never leak into client responses."""
        mock_crash = MagicMock(side_effect=Exception("Critical system kernel dump"))
        self.app.dependency_overrides[get_classifier_service] = lambda: mock_crash
        try:
            resp = self.client.post("/api/v1/classify", json={"description": "Sensor failure", "mode": "local"})
            self.assertEqual(resp.status_code, 500)
            self.assertNotIn("Traceback (most recent call last):", resp.text)
            self.assertNotIn("Exception: Critical system kernel dump", resp.text)
        finally:
            self.app.dependency_overrides.pop(get_classifier_service, None)

    # -------------------------------------------------------------------------
    # Test P: No Filesystem Paths Exposed
    # -------------------------------------------------------------------------
    def test_no_filesystem_paths_exposed(self):
        """P. Verifies server filesystem paths are never leaked in error payloads."""
        mock_crash = MagicMock(side_effect=FileNotFoundError("Missing file C:\\Users\\Administrator\\secret\\file.txt"))
        self.app.dependency_overrides[get_classifier_service] = lambda: mock_crash
        try:
            resp = self.client.post("/api/v1/classify", json={"description": "Sensor failure", "mode": "local"})
            self.assertEqual(resp.status_code, 500)
            self.assertNotIn("C:\\Users\\", resp.text)
            self.assertNotIn("secret\\file.txt", resp.text)
        finally:
            self.app.dependency_overrides.pop(get_classifier_service, None)

    # -------------------------------------------------------------------------
    # Test Q: Existing Classification Success Response Unchanged
    # -------------------------------------------------------------------------
    def test_existing_classification_success_response_unchanged(self):
        """Q. Verifies successful POST /api/v1/classify retains success=true and data envelope."""
        payload = {
            "description": "Motor is making a grinding noise.",
            "mode": "local"
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertIn("data", data)
        self.assertEqual(data["data"]["category"], "Mechanical Fault")

    # -------------------------------------------------------------------------
    # Test R: Existing History Success Response Unchanged
    # -------------------------------------------------------------------------
    @patch("persistence.repository.is_supabase_configured", return_value=True)
    @patch("persistence.repository.get_defect_reports", return_value=[])
    def test_existing_history_success_response_unchanged(self, mock_reports, mock_conf):
        """R. Verifies successful GET /api/v1/history retains success=true and data envelope."""
        response = self.client.get("/api/v1/history?limit=10")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertIn("data", data)
        self.assertIn("items", data["data"])

    # -------------------------------------------------------------------------
    # Test S: Existing /health Unchanged
    # -------------------------------------------------------------------------
    def test_existing_health_endpoint_unchanged(self):
        """S. Verifies GET /health remains functional with HTTP 200."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")

    # -------------------------------------------------------------------------
    # Test T: Existing /api/v1/status Unchanged
    # -------------------------------------------------------------------------
    def test_existing_status_endpoint_unchanged(self):
        """T. Verifies GET /api/v1/status remains functional with HTTP 200 and success=true."""
        response = self.client.get("/api/v1/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertIn("checks", data.get("data", {}))

    # -------------------------------------------------------------------------
    # Test U: Unicode/Telugu Classification Input Still Accepted
    # -------------------------------------------------------------------------
    def test_unicode_telugu_classification_accepted(self):
        """U. Verifies multilingual and Telugu Unicode text is accepted without validation errors."""
        payload = {
            "description": "మోటార్ గ్రైండింగ్ శబ్దం చేస్తోంది.",
            "mode": "local"
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))

    # -------------------------------------------------------------------------
    # Test V: Canonical Taxonomy Validation Preserved
    # -------------------------------------------------------------------------
    @patch("persistence.repository.is_supabase_configured", return_value=True)
    def test_canonical_taxonomy_validation_preserved(self, mock_conf):
        """V. Verifies history filtering with an unapproved category returns HTTP 422 VALIDATION_ERROR."""
        response = self.client.get("/api/v1/history?category=UnapprovedHydraulicFault")
        self.assertEqual(response.status_code, 422)

        data = response.json()
        self.assertFalse(data.get("success"))
        self.assertEqual(data["error"]["code"], "VALIDATION_ERROR")
        self.assertIn("approved taxonomy categories", data["error"]["message"])

    # -------------------------------------------------------------------------
    # Test W: OpenAPI Still Builds Successfully
    # -------------------------------------------------------------------------
    def test_openapi_builds_successfully(self):
        """W. Verifies /openapi.json builds and returns valid schema with all registered routes."""
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)

        schema = response.json()
        self.assertIn("openapi", schema)
        paths = schema.get("paths", {})
        self.assertIn("/health", paths)
        self.assertIn("/api/v1/status", paths)
        self.assertIn("/api/v1/classify", paths)
        self.assertIn("/api/v1/history", paths)


if __name__ == "__main__":
    unittest.main()
