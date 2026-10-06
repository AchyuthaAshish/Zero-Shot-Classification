"""Unit and Integration Tests for Step 3.3 History API (GET /api/v1/history).

Verifies:
A. GET /api/v1/history returns HTTP 200.
B. History response conforms to stable envelope schema.
C. Empty history returns HTTP 200 with empty list (total=0, not 404).
D. Single-defect history records are represented correctly (is_multi_defect=False, defect_count=1).
E. Multi-defect history records contain correct defect_count (is_multi_defect=True, defect_count>=2).
F. Multi-defect detail endpoint retrieves child defect items via repository.
G. Child defect categories remain strictly within authoritative taxonomy.
H. Unknown reports are returned normally with category="Unknown".
I. GET /api/v1/history/{valid_id} returns the expected report.
J. Nonexistent report ID returns HTTP 404.
K. Invalid UUID format returns HTTP 422 validation error.
L. Database errors are sanitized into HTTP 503 without internal disclosures.
M. Credentials, secrets, and database URLs are never leaked in responses.
N. API route code imports exclusively from persistence.repository (no direct Supabase queries).
O. Classification endpoint (POST /api/v1/classify) continues to work.
P. Pagination and filtering parameters (limit, offset, category, employee_id) are respected.
"""

import inspect
import unittest
from unittest.mock import patch, MagicMock
from uuid import uuid4
from fastapi.testclient import TestClient

from api.main import app
from persistence.repository import DatabaseError


class TestHistoryAPI(unittest.TestCase):
    """Test suite for Phase 3 Step 3.3 History API."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)

    # -------------------------------------------------------------------------
    # Test Fixtures
    # -------------------------------------------------------------------------
    @staticmethod
    def _create_sample_single_report(report_id: str = None) -> dict:
        return {
            "id": report_id or str(uuid4()),
            "reporter_name": "Alice Engineer",
            "employee_id": "EMP-101",
            "defect_description": "Conveyor belt motor is grinding loudly.",
            "category": "Mechanical Fault",
            "confidence": 0.9412,
            "confidence_level": "High",
            "confidence_range": "~94%",
            "raw_score": 0.8123,
            "calibrated_score": 0.9412,
            "top2_margin": 0.6521,
            "reliability": "High",
            "explanation": "Acoustic symptoms indicate mechanical component wear.",
            "classification_mode": "local",
            "provider": "local",
            "model": "Local ML (TF-IDF / MiniLM)",
            "is_multi_defect": False,
            "defect_count": 1,
            "validation_status": "VALID",
            "created_at": "2026-10-03T18:00:00Z"
        }

    @staticmethod
    def _create_sample_multi_report(report_id: str = None) -> dict:
        return {
            "id": report_id or str(uuid4()),
            "reporter_name": "Bob Technician",
            "employee_id": "EMP-202",
            "defect_description": "Motor grinding and temperature sensor reading erroneous.",
            "category": "Mechanical Fault",
            "confidence": 0.92,
            "confidence_level": "High",
            "confidence_range": "~92%",
            "raw_score": 0.92,
            "calibrated_score": 0.92,
            "top2_margin": 0.60,
            "reliability": "High",
            "explanation": "Multiple defect signals detected in report.",
            "classification_mode": "local",
            "provider": "local",
            "model": "Local ML (TF-IDF / MiniLM)",
            "is_multi_defect": True,
            "defect_count": 2,
            "validation_status": "VALID",
            "created_at": "2026-10-03T18:30:00Z"
        }

    @staticmethod
    def _create_sample_child_items(report_id: str) -> list:
        return [
            {
                "id": str(uuid4()),
                "report_id": report_id,
                "defect_index": 1,
                "defect_id": 1,
                "segment_id": 1,
                "defect_text": "Motor grinding",
                "start_char": 0,
                "end_char": 14,
                "category": "Mechanical Fault",
                "confidence": 0.94,
                "confidence_level": "High",
                "confidence_range": "~94%",
                "raw_score": 0.94,
                "calibrated_score": 0.94,
                "top2_margin": 0.65,
                "reliability": "High",
                "explanation": "Grinding noise in motor.",
                "classification_mode": "local",
                "provider": "local",
                "model": "Local ML (TF-IDF / MiniLM)",
                "status": "success",
                "is_ambiguous": False,
                "ambiguity_reason": None,
                "created_at": "2026-10-03T18:30:01Z"
            },
            {
                "id": str(uuid4()),
                "report_id": report_id,
                "defect_index": 2,
                "defect_id": 2,
                "segment_id": 2,
                "defect_text": "temperature sensor reading erroneous",
                "start_char": 19,
                "end_char": 55,
                "category": "Sensor Fault",
                "confidence": 0.91,
                "confidence_level": "High",
                "confidence_range": "~91%",
                "raw_score": 0.91,
                "calibrated_score": 0.91,
                "top2_margin": 0.55,
                "reliability": "High",
                "explanation": "Erroneous sensor reading.",
                "classification_mode": "local",
                "provider": "local",
                "model": "Local ML (TF-IDF / MiniLM)",
                "status": "success",
                "is_ambiguous": False,
                "ambiguity_reason": None,
                "created_at": "2026-10-03T18:30:02Z"
            }
        ]

    # -------------------------------------------------------------------------
    # Test A: GET /api/v1/history returns HTTP 200
    # -------------------------------------------------------------------------
    @patch("persistence.repository.is_supabase_configured", return_value=True)
    @patch("persistence.repository.get_defect_reports")
    def test_get_history_returns_200(self, mock_get_reports, mock_is_configured):
        """A. Verifies GET /api/v1/history returns HTTP 200 with report list."""
        sample_reports = [self._create_sample_single_report(), self._create_sample_multi_report()]
        mock_get_reports.return_value = sample_reports

        response = self.client.get("/api/v1/history")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(len(data["data"]["items"]), 2)
        self.assertEqual(data["data"]["total"], 2)

    # -------------------------------------------------------------------------
    # Test B: History Response Has Stable Schema
    # -------------------------------------------------------------------------
    @patch("persistence.repository.is_supabase_configured", return_value=True)
    @patch("persistence.repository.get_defect_reports")
    def test_history_response_stable_schema(self, mock_get_reports, mock_is_configured):
        """B. Verifies response envelope and item fields match expected schema."""
        sample = self._create_sample_single_report()
        mock_get_reports.return_value = [sample]

        response = self.client.get("/api/v1/history")
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        self.assertIn("success", payload)
        self.assertIn("data", payload)
        data = payload["data"]
        self.assertIn("items", data)
        self.assertIn("total", data)
        self.assertIn("limit", data)
        self.assertIn("offset", data)

        item = data["items"][0]
        required_keys = [
            "id", "reporter_name", "employee_id", "defect_description",
            "category", "confidence", "confidence_level", "confidence_range",
            "calibrated_score", "raw_score", "top2_margin", "reliability",
            "explanation", "classification_mode", "provider", "model",
            "is_multi_defect", "defect_count", "validation_status", "created_at"
        ]
        for key in required_keys:
            self.assertIn(key, item, f"Missing key '{key}' in history item schema")

    # -------------------------------------------------------------------------
    # Test C: Empty History Returns HTTP 200 with Empty List (Not 404)
    # -------------------------------------------------------------------------
    @patch("persistence.repository.is_supabase_configured", return_value=True)
    @patch("persistence.repository.get_defect_reports", return_value=[])
    def test_empty_history_returns_200_empty_items(self, mock_get_reports, mock_is_configured):
        """C. Verifies empty database returns HTTP 200 with items=[], total=0."""
        response = self.client.get("/api/v1/history")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data["data"]["items"], [])
        self.assertEqual(data["data"]["total"], 0)

    # -------------------------------------------------------------------------
    # Test D: Single-Defect Record Representation
    # -------------------------------------------------------------------------
    @patch("persistence.repository.is_supabase_configured", return_value=True)
    @patch("persistence.repository.get_defect_reports")
    def test_single_defect_record_representation(self, mock_get_reports, mock_is_configured):
        """D. Verifies single-defect record exposes is_multi_defect=False and defect_count=1."""
        sample = self._create_sample_single_report()
        mock_get_reports.return_value = [sample]

        response = self.client.get("/api/v1/history")
        self.assertEqual(response.status_code, 200)

        item = response.json()["data"]["items"][0]
        self.assertFalse(item["is_multi_defect"])
        self.assertEqual(item["defect_count"], 1)
        self.assertEqual(item["category"], "Mechanical Fault")

    # -------------------------------------------------------------------------
    # Test E: Multi-Defect Record Defect Count
    # -------------------------------------------------------------------------
    @patch("persistence.repository.is_supabase_configured", return_value=True)
    @patch("persistence.repository.get_defect_reports")
    def test_multi_defect_record_defect_count(self, mock_get_reports, mock_is_configured):
        """E. Verifies multi-defect record exposes is_multi_defect=True and defect_count>=2."""
        sample = self._create_sample_multi_report()
        mock_get_reports.return_value = [sample]

        response = self.client.get("/api/v1/history")
        self.assertEqual(response.status_code, 200)

        item = response.json()["data"]["items"][0]
        self.assertTrue(item["is_multi_defect"])
        self.assertEqual(item["defect_count"], 2)

    # -------------------------------------------------------------------------
    # Test F & G: Multi-Defect Detail Retrieves Child Defect Items
    # -------------------------------------------------------------------------
    @patch("persistence.repository.is_supabase_configured", return_value=True)
    @patch("persistence.repository.get_defect_report_by_id")
    @patch("persistence.repository.get_defect_report_items")
    def test_multi_defect_detail_child_items(self, mock_get_items, mock_get_by_id, mock_is_configured):
        """F & G. Verifies GET /api/v1/history/{id} returns parent + child defect items in taxonomy."""
        target_id = str(uuid4())
        mock_get_by_id.return_value = self._create_sample_multi_report(report_id=target_id)
        mock_get_items.return_value = self._create_sample_child_items(report_id=target_id)

        response = self.client.get(f"/api/v1/history/{target_id}")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data.get("success"))
        report_data = data.get("data", {})
        self.assertEqual(report_data.get("id"), target_id)
        self.assertTrue(report_data.get("is_multi_defect"))

        defects = report_data.get("defects", [])
        self.assertEqual(len(defects), 2)

        approved_categories = {
            "Mechanical Fault", "Electrical Fault", "Sensor Fault",
            "Temperature Fault", "Software Fault", "Power Supply Fault",
            "Communication Fault", "Unknown"
        }
        for d in defects:
            self.assertIn(d["category"], approved_categories)
            self.assertIn("defect_index", d)
            self.assertIn("defect_text", d)
            self.assertIn("start_char", d)
            self.assertIn("end_char", d)

    # -------------------------------------------------------------------------
    # Test H: Unknown Report Returned Correctly
    # -------------------------------------------------------------------------
    @patch("persistence.repository.is_supabase_configured", return_value=True)
    @patch("persistence.repository.get_defect_reports")
    def test_unknown_report_returned_correctly(self, mock_get_reports, mock_is_configured):
        """H. Verifies Unknown report is returned normally with category='Unknown' and defect_count=0."""
        unk_report = {
            "id": str(uuid4()),
            "reporter_name": "Charlie Operator",
            "employee_id": "EMP-303",
            "defect_description": "Something is not working right.",
            "category": "Unknown",
            "confidence": None,
            "confidence_level": "Uncertain",
            "confidence_range": "Insufficient Evidence",
            "raw_score": None,
            "calibrated_score": None,
            "top2_margin": None,
            "reliability": "Medium",
            "explanation": "No technical defect symptoms detected.",
            "classification_mode": "local",
            "provider": "local",
            "model": "Local ML (TF-IDF / MiniLM)",
            "is_multi_defect": False,
            "defect_count": 0,
            "validation_status": "VALID",
            "created_at": "2026-10-03T19:00:00Z"
        }
        mock_get_reports.return_value = [unk_report]

        response = self.client.get("/api/v1/history")
        self.assertEqual(response.status_code, 200)

        item = response.json()["data"]["items"][0]
        self.assertEqual(item["category"], "Unknown")
        self.assertEqual(item["defect_count"], 0)
        self.assertEqual(item["confidence_level"], "Uncertain")

    # -------------------------------------------------------------------------
    # Test I: GET /api/v1/history/{valid_id} Returns Expected Report
    # -------------------------------------------------------------------------
    @patch("persistence.repository.is_supabase_configured", return_value=True)
    @patch("persistence.repository.get_defect_report_by_id")
    @patch("persistence.repository.get_defect_report_items", return_value=[])
    def test_get_history_by_valid_id(self, mock_get_items, mock_get_by_id, mock_is_configured):
        """I. Verifies fetching an individual existing report by UUID returns HTTP 200."""
        target_id = str(uuid4())
        mock_get_by_id.return_value = self._create_sample_single_report(report_id=target_id)

        response = self.client.get(f"/api/v1/history/{target_id}")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data["data"]["id"], target_id)
        self.assertEqual(data["data"]["category"], "Mechanical Fault")

    # -------------------------------------------------------------------------
    # Test J: Nonexistent Report ID Returns HTTP 404
    # -------------------------------------------------------------------------
    @patch("persistence.repository.is_supabase_configured", return_value=True)
    @patch("persistence.repository.get_defect_report_by_id", return_value=None)
    def test_nonexistent_report_id_returns_404(self, mock_get_by_id, mock_is_configured):
        """J. Verifies querying a nonexistent report UUID returns HTTP 404."""
        nonexistent_id = str(uuid4())
        response = self.client.get(f"/api/v1/history/{nonexistent_id}")
        self.assertEqual(response.status_code, 404)

        data = response.json()
        self.assertIn("error", data)
        self.assertIn("not found", data["error"]["message"].lower())

    # -------------------------------------------------------------------------
    # Test K: Invalid Report ID Rejected with HTTP 422
    # -------------------------------------------------------------------------
    def test_invalid_uuid_format_rejected(self):
        """K. Verifies malformed report_id path parameter returns HTTP 422."""
        response = self.client.get("/api/v1/history/not-a-valid-uuid-12345")
        self.assertEqual(response.status_code, 422)

        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "VALIDATION_ERROR")

    # -------------------------------------------------------------------------
    # Test L: Database Errors Sanitized to HTTP 503
    # -------------------------------------------------------------------------
    @patch("persistence.repository.is_supabase_configured", return_value=True)
    @patch("persistence.repository.get_defect_reports")
    def test_database_errors_sanitized(self, mock_get_reports, mock_is_configured):
        """L. Verifies database errors raise sanitized HTTP 503 without leaking details."""
        mock_get_reports.side_effect = DatabaseError("Connection refused to db.supabase.co:5432 with password secret123")

        response = self.client.get("/api/v1/history")
        self.assertEqual(response.status_code, 503)

        content = response.text
        self.assertNotIn("password", content)
        self.assertNotIn("secret123", content)
        self.assertNotIn(":5432", content)

    # -------------------------------------------------------------------------
    # Test M: Secrets and Credentials Not Exposed
    # -------------------------------------------------------------------------
    @patch("persistence.repository.is_supabase_configured", return_value=True)
    @patch("persistence.repository.get_defect_reports")
    def test_secrets_not_exposed(self, mock_get_reports, mock_is_configured):
        """M. Verifies API responses never leak sensitive keys or database URLs."""
        mock_get_reports.return_value = [self._create_sample_single_report()]

        response = self.client.get("/api/v1/history")
        self.assertEqual(response.status_code, 200)

        content = response.text
        forbidden = ["AIzaSy", "supabase_key", "SUPABASE_KEY", "sk-", "postgresql://", "password"]
        for secret in forbidden:
            self.assertNotIn(secret, content)

    # -------------------------------------------------------------------------
    # Test N: API Route Does Not Directly Query Supabase
    # -------------------------------------------------------------------------
    def test_api_does_not_directly_query_supabase(self):
        """N. Verifies api/routes/history.py imports from repository, not supabase directly."""
        import api.routes.history as history_route_mod
        source = inspect.getsource(history_route_mod)

        # Forbidden direct imports
        self.assertNotIn("from supabase import", source)
        self.assertNotIn("import supabase", source)
        self.assertNotIn("database.supabase_client", source)

        # Required repository pattern import
        self.assertIn("from persistence.repository import", source)

    # -------------------------------------------------------------------------
    # Test O: Classification Endpoint Still Works (Regression Check)
    # -------------------------------------------------------------------------
    def test_classification_endpoint_still_works(self):
        """O. Verifies classification endpoint (POST /api/v1/classify) remains fully operational."""
        response = self.client.post("/api/v1/classify", json={
            "description": "Motor is making a grinding noise.",
            "mode": "local"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data["data"]["category"], "Mechanical Fault")

    # -------------------------------------------------------------------------
    # Test P: Filtering and Pagination Query Parameters
    # -------------------------------------------------------------------------
    @patch("persistence.repository.is_supabase_configured", return_value=True)
    @patch("persistence.repository.get_defect_reports", return_value=[])
    def test_filtering_and_pagination_params_passed(self, mock_get_reports, mock_is_configured):
        """P. Verifies limit, offset, category, and employee_id query params reach repository."""
        response = self.client.get("/api/v1/history?limit=25&offset=50&category=Sensor%20Fault&employee_id=EMP-999")
        self.assertEqual(response.status_code, 200)

        mock_get_reports.assert_called_once_with(
            limit=25,
            category="Sensor Fault",
            employee_id="EMP-999",
            offset=50
        )


if __name__ == "__main__":
    unittest.main()
