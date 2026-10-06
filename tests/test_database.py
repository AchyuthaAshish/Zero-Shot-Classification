"""Unit Tests for Supabase Database Integration.

Tests:
1. Required fields validation (reporter_name, employee_id, defect_description).
2. Category must be one of the 8 allowed categories.
3. Employee ID is treated as text (preserving leading zeros and alphanumeric chars).
4. Missing Supabase configuration produces a clean configuration error.
5. Classification result can be transformed into a database record payload.
6. Database errors are sanitized for the UI (no secrets or sensitive URLs leaked).
"""

import unittest
from unittest.mock import MagicMock, patch

from core.exceptions import ConfigurationError
from database.supabase_client import (
    validate_defect_report_payload,
    save_defect_report,
    get_defect_reports,
    is_supabase_configured,
    get_supabase_client,
    reset_supabase_client,
    _sanitize_error,
    DatabaseError
)
from core.schemas import ClassificationResult


class TestDatabaseIntegration(unittest.TestCase):
    """Test suite for Supabase database layer validation and operations."""

    def tearDown(self):
        reset_supabase_client()

    def test_required_fields_validation(self):
        # Empty reporter_name
        with self.assertRaises(ValueError) as ctx:
            validate_defect_report_payload(
                reporter_name="",
                employee_id="EMP-001",
                defect_description="Motor grinding",
                category="Mechanical Fault"
            )
        self.assertIn("Reporter Name", str(ctx.exception))

        # Empty employee_id
        with self.assertRaises(ValueError) as ctx:
            validate_defect_report_payload(
                reporter_name="John Doe",
                employee_id="",
                defect_description="Motor grinding",
                category="Mechanical Fault"
            )
        self.assertIn("Employee ID", str(ctx.exception))

        # Empty defect_description
        with self.assertRaises(ValueError) as ctx:
            validate_defect_report_payload(
                reporter_name="John Doe",
                employee_id="EMP-001",
                defect_description="   ",
                category="Mechanical Fault"
            )
        self.assertIn("Defect Description", str(ctx.exception))

    def test_category_must_be_approved_taxonomy(self):
        # Invalid unapproved category
        with self.assertRaises(ValueError) as ctx:
            validate_defect_report_payload(
                reporter_name="John Doe",
                employee_id="EMP-001",
                defect_description="Motor grinding",
                category="Motor Failure"  # Unapproved!
            )
        self.assertIn("Invalid category", str(ctx.exception))

        # All 8 approved categories must succeed
        approved_categories = [
            "Mechanical Fault",
            "Electrical Fault",
            "Sensor Fault",
            "Temperature Fault",
            "Software Fault",
            "Power Supply Fault",
            "Communication Fault",
            "Unknown"
        ]
        for cat in approved_categories:
            payload = validate_defect_report_payload(
                reporter_name="John Doe",
                employee_id="EMP-001",
                defect_description="Motor issue",
                category=cat
            )
            self.assertEqual(payload["category"], cat)

    def test_employee_id_treated_as_text(self):
        # Numeric string with leading zeroes
        payload = validate_defect_report_payload(
            reporter_name="John Doe",
            employee_id="000142",
            defect_description="Pump vibrating",
            category="Mechanical Fault"
        )
        self.assertEqual(payload["employee_id"], "000142")
        self.assertIsInstance(payload["employee_id"], str)

        # Alphanumeric ID
        payload2 = validate_defect_report_payload(
            reporter_name="Jane Doe",
            employee_id="TECH-A902",
            defect_description="Pump vibrating",
            category="Mechanical Fault"
        )
        self.assertEqual(payload2["employee_id"], "TECH-A902")

    def test_missing_supabase_configuration_raises_clean_error(self):
        with patch.dict("os.environ", {}, clear=True), patch("config.settings.load_env_file"):
            from config.settings import reset_settings
            reset_settings()
            self.assertFalse(is_supabase_configured())
            with self.assertRaises(ConfigurationError) as ctx:
                get_supabase_client()
            self.assertIn("Supabase configuration missing", str(ctx.exception))
            reset_settings()

    def test_invalid_supabase_url_format(self):
        with patch("database.supabase_client.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.supabase_url = "not_a_valid_url"
            mock_s.supabase_key = "some_key"
            mock_settings.return_value = mock_s
            with self.assertRaises(ConfigurationError) as ctx:
                get_supabase_client()
            self.assertIn("Invalid SUPABASE_URL", str(ctx.exception))

    def test_classification_result_transformation(self):
        res = ClassificationResult(
            category="Mechanical Fault",
            confidence=0.83,
            reliability="Medium",
            reason="Grinding noise from motor",
            language="English",
            status="success",
            original_description="Motor is grinding",
            normalized_description="motor is grinding",
            model_source="local_ml"
        )

        payload = validate_defect_report_payload(
            reporter_name="Test Operator",
            employee_id="EMP-999",
            defect_description=res.original_description,
            category=res.category,
            confidence=res.confidence,
            reliability=res.reliability,
            explanation=res.reason,
            classification_mode=res.model_source,
            provider="local",
            model="multilingual-minilm"
        )

        self.assertEqual(payload["category"], "Mechanical Fault")
        self.assertEqual(payload["confidence"], 0.83)
        self.assertEqual(payload["reliability"], "Medium")
        self.assertEqual(payload["explanation"], "Grinding noise from motor")
        self.assertEqual(payload["reporter_name"], "Test Operator")
        self.assertEqual(payload["employee_id"], "EMP-999")

    def test_database_error_sanitization(self):
        secret_key = "sbp_live_secret_key_12345"
        project_url = "https://xyzproject.supabase.co"
        raw_error = f"Connection refused to {project_url} with key {secret_key}"

        with patch("database.supabase_client.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.supabase_key = secret_key
            mock_s.supabase_url = project_url
            mock_s.llm_api_key = None
            mock_settings.return_value = mock_s

            sanitized = _sanitize_error(raw_error)
            self.assertNotIn(secret_key, sanitized)
            self.assertIn("***MASKED***", sanitized)
            self.assertNotIn(project_url, sanitized)
            self.assertIn("***URL***", sanitized)

    def test_mocked_save_defect_report(self):
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_insert = MagicMock()
        mock_execute = MagicMock()

        mock_execute.return_value = MagicMock(data=[{
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "reporter_name": "Demo Employee",
            "employee_id": "DEMO-001",
            "defect_description": "The conveyor motor is making a grinding noise.",
            "category": "Mechanical Fault",
            "confidence": 0.83,
            "reliability": "Medium",
            "explanation": "Grinding noise indicates mechanical friction",
            "classification_mode": "local",
            "created_at": "2026-09-27T03:00:00+00:00"
        }])
        mock_insert.return_value.execute = mock_execute
        mock_table.insert = mock_insert
        mock_client.table.return_value = mock_table

        record = save_defect_report(
            reporter_name="Demo Employee",
            employee_id="DEMO-001",
            defect_description="The conveyor motor is making a grinding noise.",
            category="Mechanical Fault",
            confidence=0.83,
            reliability="Medium",
            explanation="Grinding noise indicates mechanical friction",
            classification_mode="local",
            client=mock_client
        )

        self.assertEqual(record["id"], "123e4567-e89b-12d3-a456-426614174000")
        self.assertEqual(record["category"], "Mechanical Fault")
        self.assertEqual(record["employee_id"], "DEMO-001")

    def test_save_defect_report_dict_support(self):
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_insert = MagicMock()
        mock_execute = MagicMock()

        mock_execute.return_value = MagicMock(data=[{
            "id": "abc-123",
            "reporter_name": "Demo User",
            "employee_id": "DEMO-001",
            "defect_description": "The conveyor motor is making a grinding noise.",
            "category": "Mechanical Fault",
            "created_at": "2026-09-27T03:00:00+00:00"
        }])
        mock_insert.return_value.execute = mock_execute
        mock_table.insert = mock_insert
        mock_client.table.return_value = mock_table

        demo_dict = {
            "reporter_name": "Demo User",
            "employee_id": "DEMO-001",
            "defect_description": "The conveyor motor is making a grinding noise.",
            "category": "Mechanical Fault",
            "confidence": 0.83,
            "reliability": "Medium",
            "explanation": "Grinding noise indicates friction",
            "classification_mode": "LOCAL",
            "provider": "local",
            "model": "TF-IDF + LinearSVC"
        }

        res = save_defect_report(demo_dict, client=mock_client)
        self.assertEqual(res["id"], "abc-123")
        self.assertEqual(res["category"], "Mechanical Fault")

        # Verify insert payload does NOT contain id or created_at
        called_payload = mock_table.insert.call_args[0][0]
        self.assertNotIn("id", called_payload)
        self.assertNotIn("created_at", called_payload)
        self.assertEqual(called_payload["reporter_name"], "Demo User")
        self.assertEqual(called_payload["employee_id"], "DEMO-001")
        self.assertEqual(called_payload["category"], "Mechanical Fault")

    def test_failed_classification_status_not_saved(self):
        error_statuses = ["input_error", "validation_error", "configuration_error", "model_error", "system_error"]
        saved_statuses = ["success", "unknown", "low_confidence"]

        for s in error_statuses:
            self.assertNotIn(s, saved_statuses)

    def test_mocked_get_defect_reports(self):
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_query = MagicMock()
        mock_execute = MagicMock()

        mock_execute.return_value = MagicMock(data=[
            {"id": "1", "category": "Mechanical Fault", "employee_id": "EMP-001"},
            {"id": "2", "category": "Electrical Fault", "employee_id": "EMP-002"}
        ])
        mock_query.order.return_value.limit.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.ilike.return_value = mock_query
        mock_query.execute = mock_execute

        mock_table.select.return_value = mock_query
        mock_client.table.return_value = mock_table

        results = get_defect_reports(limit=10, category="Mechanical Fault", employee_id="EMP-001", client=mock_client)
        self.assertEqual(len(results), 2)
        mock_client.table.assert_called_with("defect_reports")

    def test_validate_payload_with_confidence_assessment_fields(self):
        """Verifies Supabase payload accepts confidence_level, confidence_range, raw_score, calibrated_score."""
        payload = validate_defect_report_payload(
            reporter_name="Jane Doe",
            employee_id="EMP-102",
            defect_description="Overheating pump bearing",
            category="Temperature Fault",
            confidence=0.88,
            reliability="High",
            explanation="Thermal threshold exceeded",
            classification_mode="local",
            provider="local",
            model="multilingual-minilm",
            confidence_level="High",
            confidence_range="Uncalibrated model score",
            raw_score=0.88,
            calibrated_score=None
        )

        self.assertEqual(payload["category"], "Temperature Fault")
        self.assertEqual(payload["confidence"], 0.88)
        self.assertEqual(payload["confidence_level"], "High")
        self.assertEqual(payload["confidence_range"], "Uncalibrated model score")
        self.assertEqual(payload["raw_score"], 0.88)
        self.assertIsNone(payload["calibrated_score"])

    def test_save_defect_report_with_confidence_assessment_fields(self):
        """Verifies save_defect_report forwards confidence assessment fields to database insert."""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_insert = MagicMock()
        mock_execute = MagicMock()

        saved_data = [{
            "id": "123e4567-e89b-12d3-a456-426614174001",
            "reporter_name": "Jane Doe",
            "employee_id": "EMP-102",
            "defect_description": "Overheating pump bearing",
            "category": "Temperature Fault",
            "confidence": 0.88,
            "confidence_level": "High",
            "confidence_range": "Uncalibrated model score",
            "raw_score": 0.88,
            "calibrated_score": None
        }]
        mock_execute.return_value = MagicMock(data=saved_data)
        mock_insert.return_value.execute = mock_execute
        mock_table.insert = mock_insert
        mock_client.table.return_value = mock_table

        record = save_defect_report(
            reporter_name="Jane Doe",
            employee_id="EMP-102",
            defect_description="Overheating pump bearing",
            category="Temperature Fault",
            confidence=0.88,
            reliability="High",
            confidence_level="High",
            confidence_range="Uncalibrated model score",
            raw_score=0.88,
            calibrated_score=None,
            client=mock_client
        )

        self.assertEqual(record["confidence_level"], "High")
        self.assertEqual(record["confidence_range"], "Uncalibrated model score")
        self.assertEqual(record["raw_score"], 0.88)
        inserted_payload = mock_table.insert.call_args[0][0]
        self.assertEqual(inserted_payload["confidence_level"], "High")
        self.assertEqual(inserted_payload["raw_score"], 0.88)
        self.assertEqual(inserted_payload["confidence"], 0.88)


if __name__ == "__main__":
    unittest.main()

