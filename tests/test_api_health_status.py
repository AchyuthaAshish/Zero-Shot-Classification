"""Comprehensive test suite for Health/Status API and Engine Diagnostics.

Conforms to Phase 3 Step 3.4:
- A. Existing /health remains functional
- B. /api/v1/status exists
- C. Status response has stable schema
- D. Application check is healthy
- E. Local classifier readiness check
- F. Database healthy path
- G. Database failure/degraded path
- H. Provider configuration check does not expose secrets
- I. External provider is NOT called merely for health
- J. No Supabase write operations occur
- K. Sanitized dependency failure
- L. Overall healthy status
- M. Overall degraded status
- N. Overall unhealthy status if applicable
- O. OpenAPI includes /api/v1/status
- P. Existing classification endpoint regression
- Q. Existing history endpoint regression
"""

import os
import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from fastapi.testclient import TestClient

from api.main import create_app
from api.config import get_api_settings
from api.routes.health import (
    check_application_liveness,
    check_classifier_readiness,
    check_database_readiness_status,
    check_provider_readiness,
    aggregate_overall_status,
)
from api.schemas.health import HealthStatus, SubsystemCheck


class TestApiHealthAndStatus(unittest.TestCase):
    """Test suite covering health, status, engine diagnostics, and regressions."""

    @classmethod
    def setUpClass(cls):
        """Set up test client with default application."""
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    # -------------------------------------------------------------------------
    # Test A: Existing /health remains functional
    # -------------------------------------------------------------------------
    def test_existing_health_endpoint_functional(self):
        """A. Verifies GET /health returns HTTP 200, status healthy, and version."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("service"), "industrial-defect-intelligence-api")
        self.assertEqual(data.get("version"), get_api_settings().version)

    # -------------------------------------------------------------------------
    # Test B: /api/v1/status exists
    # -------------------------------------------------------------------------
    def test_status_endpoint_exists(self):
        """B. Verifies GET /api/v1/status is reachable and returns HTTP 200."""
        response = self.client.get("/api/v1/status")
        self.assertEqual(response.status_code, 200)

    # -------------------------------------------------------------------------
    # Test C: Status response has stable schema
    # -------------------------------------------------------------------------
    def test_status_response_schema_structure(self):
        """C. Verifies that GET /api/v1/status conforms to the defined schema."""
        response = self.client.get("/api/v1/status")
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        self.assertTrue(payload.get("success"))
        data = payload.get("data", {})
        self.assertIn("status", data)
        self.assertIn(data["status"], ["healthy", "degraded", "unhealthy"])
        self.assertEqual(data.get("service"), "industrial-defect-intelligence-api")
        self.assertIn("version", data)
        self.assertIn("checks", data)

        checks = data["checks"]
        for key in ["application", "classifier", "database", "providers"]:
            self.assertIn(key, checks, f"Missing check: {key}")
            self.assertIn("status", checks[key])
            self.assertIn(checks[key]["status"], ["healthy", "degraded", "unhealthy"])
            self.assertIn("message", checks[key])
            self.assertIsInstance(checks[key]["message"], str)

    # -------------------------------------------------------------------------
    # Test D: Application check is healthy
    # -------------------------------------------------------------------------
    def test_application_check_is_healthy(self):
        """D. Verifies the core application process is reported as healthy."""
        check = check_application_liveness()
        self.assertEqual(check.status, HealthStatus.HEALTHY)
        self.assertIn("operational", check.message.lower())

        response = self.client.get("/api/v1/status")
        data = response.json().get("data", {})
        app_check = data.get("checks", {}).get("application", {})
        self.assertEqual(app_check.get("status"), "healthy")

    # -------------------------------------------------------------------------
    # Test E: Local classifier readiness check
    # -------------------------------------------------------------------------
    def test_classifier_readiness_check_healthy(self):
        """E1. Verifies local classifier check reports healthy when artifacts exist."""
        response = self.client.get("/api/v1/status")
        data = response.json().get("data", {})
        clf_check = data.get("checks", {}).get("classifier", {})
        self.assertEqual(clf_check.get("status"), "healthy")
        self.assertTrue(clf_check.get("details", {}).get("local_model_available"))

    def test_classifier_readiness_check_unhealthy_when_model_fails(self):
        """E2. Verifies classifier check returns unhealthy when classifier fails to load."""
        broken_service = MagicMock()
        type(broken_service).local_classifier = PropertyMock(side_effect=RuntimeError("Model corrupted"))

        check = check_classifier_readiness(broken_service)
        self.assertEqual(check.status, HealthStatus.UNHEALTHY)
        self.assertIn("not ready", check.message)
        self.assertFalse(check.details.get("local_model_available"))

    # -------------------------------------------------------------------------
    # Test F: Database healthy path
    # -------------------------------------------------------------------------
    @patch("persistence.repository.check_database_readiness")
    def test_database_healthy_path(self, mock_db_check):
        """F. Verifies database check reports healthy when connection succeeds."""
        mock_db_check.return_value = {
            "healthy": True,
            "status": "healthy",
            "message": "Database connection is available",
            "details": {"configured": True, "connected": True}
        }

        response = self.client.get("/api/v1/status")
        db_check = response.json()["data"]["checks"]["database"]
        self.assertEqual(db_check["status"], "healthy")
        self.assertEqual(db_check["message"], "Database connection is available")
        self.assertTrue(db_check["details"]["connected"])

    # -------------------------------------------------------------------------
    # Test G: Database failure/degraded path
    # -------------------------------------------------------------------------
    @patch("persistence.repository.check_database_readiness")
    def test_database_degraded_path(self, mock_db_check):
        """G. Verifies database check reports degraded when database is unavailable."""
        mock_db_check.return_value = {
            "healthy": False,
            "status": "degraded",
            "message": "Database connection is unavailable",
            "details": {"configured": True, "connected": False}
        }

        response = self.client.get("/api/v1/status")
        data = response.json()["data"]
        db_check = data["checks"]["database"]
        self.assertEqual(db_check["status"], "degraded")
        self.assertFalse(db_check["details"]["connected"])
        # Overall status should degrade
        self.assertEqual(data["status"], "degraded")

    # -------------------------------------------------------------------------
    # Test H: Provider configuration check does not expose secrets
    # -------------------------------------------------------------------------
    def test_provider_configuration_does_not_expose_secrets(self):
        """H. Verifies external provider checks never return API keys or tokens."""
        secret_key = "AIzaSySecretTestKey_123456789_NEVER_LEAK"
        with patch.dict(os.environ, {"LLM_API_KEY": secret_key, "CLASSIFICATION_MODE": "hybrid"}):
            with patch("config.settings.get_settings") as mock_settings:
                settings_obj = MagicMock()
                settings_obj.classification_mode = "hybrid"
                settings_obj.llm_provider = "gemini"
                settings_obj.llm_model = "gemini-3.8-flash"
                settings_obj.llm_api_key = secret_key
                mock_settings.return_value = settings_obj

                check = check_provider_readiness()
                self.assertEqual(check.status, HealthStatus.HEALTHY)
                # Confirm the secret key is NOT present in any attribute
                self.assertNotIn(secret_key, check.message)
                if check.details:
                    for val in check.details.values():
                        self.assertNotIn(secret_key, str(val))

                response = self.client.get("/api/v1/status")
                raw_text = response.text
                self.assertNotIn(secret_key, raw_text)

    # -------------------------------------------------------------------------
    # Test I: External provider is NOT called merely for health
    # -------------------------------------------------------------------------
    @patch("llm.client.GeminiLLMClient.classify")
    @patch("llm.client.BaseLLMClient.classify")
    def test_external_provider_not_invoked_for_health(self, mock_base_classify, mock_gemini_classify):
        """I. Verifies neither GET /health nor GET /api/v1/status invokes external LLM inference."""
        self.client.get("/health")
        self.client.get("/api/v1/status")

        mock_base_classify.assert_not_called()
        mock_gemini_classify.assert_not_called()

    # -------------------------------------------------------------------------
    # Test J: No Supabase write operations occur
    # -------------------------------------------------------------------------
    @patch("database.supabase_client.is_supabase_configured", return_value=True)
    @patch("database.supabase_client.get_supabase_client")
    def test_no_supabase_write_operations(self, mock_get_sb, mock_is_conf):
        """J. Verifies health and status endpoints never trigger insert, update, or delete."""
        mock_table = MagicMock()
        mock_client = MagicMock()
        mock_client.table.return_value = mock_table
        mock_get_sb.return_value = mock_client

        # Mock select to return a valid dummy response
        mock_select = MagicMock()
        mock_limit = MagicMock()
        mock_table.select.return_value = mock_limit
        mock_limit.limit.return_value = mock_limit
        mock_limit.execute.return_value = MagicMock(data=[{"id": "test-uuid"}])

        self.client.get("/health")
        self.client.get("/api/v1/status")

        mock_table.insert.assert_not_called()
        mock_table.update.assert_not_called()
        mock_table.delete.assert_not_called()

    # -------------------------------------------------------------------------
    # Test K: Sanitized dependency failure
    # -------------------------------------------------------------------------
    @patch("persistence.repository.check_database_readiness")
    def test_sanitized_dependency_failure(self, mock_db_check):
        """K. Verifies exceptions containing secrets are completely sanitized."""
        sensitive_connection_string = "postgresql://postgres:p@ssword123@db.supabase.co:5432/postgres"
        mock_db_check.side_effect = RuntimeError(f"Connection failed: {sensitive_connection_string}")

        response = self.client.get("/api/v1/status")
        self.assertEqual(response.status_code, 200)

        raw_text = response.text
        self.assertNotIn("p@ssword123", raw_text)
        self.assertNotIn("postgresql://", raw_text)

    # -------------------------------------------------------------------------
    # Test L: Overall healthy status
    # -------------------------------------------------------------------------
    def test_overall_healthy_status_aggregation(self):
        """L. Verifies overall status is healthy when all components are operational."""
        app_check = SubsystemCheck(status=HealthStatus.HEALTHY, message="OK")
        clf_check = SubsystemCheck(status=HealthStatus.HEALTHY, message="OK")
        db_check = SubsystemCheck(status=HealthStatus.HEALTHY, message="OK")
        prov_check = SubsystemCheck(status=HealthStatus.HEALTHY, message="OK")

        status = aggregate_overall_status(app_check, clf_check, db_check, prov_check, mode="hybrid")
        self.assertEqual(status, HealthStatus.HEALTHY)

    # -------------------------------------------------------------------------
    # Test M: Overall degraded status
    # -------------------------------------------------------------------------
    def test_overall_degraded_status_aggregation(self):
        """M. Verifies overall status degrades when database or provider is degraded."""
        app_check = SubsystemCheck(status=HealthStatus.HEALTHY, message="OK")
        clf_check = SubsystemCheck(status=HealthStatus.HEALTHY, message="OK")
        db_check = SubsystemCheck(status=HealthStatus.DEGRADED, message="DB down")
        prov_check = SubsystemCheck(status=HealthStatus.HEALTHY, message="OK")

        status = aggregate_overall_status(app_check, clf_check, db_check, prov_check, mode="hybrid")
        self.assertEqual(status, HealthStatus.DEGRADED)

    # -------------------------------------------------------------------------
    # Test N: Overall unhealthy status
    # -------------------------------------------------------------------------
    def test_overall_unhealthy_status_when_classifier_fails(self):
        """N. Verifies overall status is unhealthy when local classifier fails in local/hybrid mode."""
        app_check = SubsystemCheck(status=HealthStatus.HEALTHY, message="OK")
        clf_check = SubsystemCheck(status=HealthStatus.UNHEALTHY, message="Classifier missing")
        db_check = SubsystemCheck(status=HealthStatus.HEALTHY, message="OK")
        prov_check = SubsystemCheck(status=HealthStatus.HEALTHY, message="OK")

        status = aggregate_overall_status(app_check, clf_check, db_check, prov_check, mode="hybrid")
        self.assertEqual(status, HealthStatus.UNHEALTHY)

    # -------------------------------------------------------------------------
    # Test O: OpenAPI includes /api/v1/status
    # -------------------------------------------------------------------------
    def test_openapi_schema_contains_status_endpoint(self):
        """O. Verifies OpenAPI schema registers /api/v1/status."""
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)

        schema = response.json()
        paths = schema.get("paths", {})
        self.assertIn("/health", paths)
        self.assertIn("/api/v1/status", paths)

        status_path = paths["api/v1/status"] if "api/v1/status" in paths else paths["/api/v1/status"]
        self.assertIn("get", status_path)

    # -------------------------------------------------------------------------
    # Test P: Existing classification endpoint regression
    # -------------------------------------------------------------------------
    def test_classification_endpoint_regression(self):
        """P. Verifies POST /api/v1/classify remains functional in LOCAL mode."""
        payload = {
            "description": "Motor is making a grinding noise.",
            "mode": "local"
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data.get("success"))
        classification = data.get("data", {})
        self.assertEqual(classification.get("category"), "Mechanical Fault")
        self.assertEqual(classification.get("status"), "success")

    # -------------------------------------------------------------------------
    # Test Q: Existing history endpoint regression
    # -------------------------------------------------------------------------
    @patch("persistence.repository.is_supabase_configured", return_value=True)
    @patch("persistence.repository.get_defect_reports")
    def test_history_endpoint_regression(self, mock_reports, mock_is_conf):
        """Q. Verifies GET /api/v1/history remains functional."""
        mock_reports.return_value = [
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "reporter_name": "Test Operator",
                "employee_id": "EMP-001",
                "defect_description": "Conveyor motor stopped unexpectedly.",
                "category": "Mechanical Fault",
                "confidence": 0.92,
                "reliability": "High",
                "explanation": "Mechanical failure confirmed.",
                "classification_mode": "local",
                "created_at": "2026-10-04T12:00:00Z"
            }
        ]

        response = self.client.get("/api/v1/history?limit=10")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(len(data.get("data", {}).get("items", [])), 1)


if __name__ == "__main__":
    unittest.main()
