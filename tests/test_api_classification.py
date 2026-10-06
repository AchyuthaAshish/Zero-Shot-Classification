"""Unit and Integration Tests for Step 3.2 Classification API (POST /api/v1/classify).

Verifies:
A. Valid single-defect request returns HTTP 200 with complete envelope.
B. Valid multi-defect request returns HTTP 200 with segmented defects.
C. Empty description is rejected with HTTP 422.
D. Whitespace-only description is rejected with HTTP 422.
E. Overly long description (>2000 chars) is rejected with HTTP 422.
F. Invalid classification mode is rejected with HTTP 422.
G. Single-defect response contains category.
H. Single-defect response contains full structured confidence assessment.
I. Multi-defect response contains correct defect_count.
J. Multi-defect response contains per-defect categories.
K. Multi-defect response contains segment spans where available.
L. Unknown input returns Unknown rather than crashing.
M. Ambiguous input preserves ambiguity information.
N. Unicode/Telugu native script input is accepted and preserved.
O. Code-switched Telugu-English input is accepted and preserved.
P. API does not expose secrets, credentials, or internal stack traces.
Q. Classification endpoint does NOT write to Supabase (zero DB persistence).
R. Classifier-level errors (model_error, configuration_error, system_error) are sanitized.
S. OpenAPI documentation accurately registers POST /api/v1/classify with examples.
"""

import json
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.main import app
from core.schemas import (
    ClassificationResult,
    ConfidenceAssessment,
    AmbiguityAssessment,
    PerDefectClassification,
    MultiDefectClassificationResult,
    DefectSegment,
    DefectSegmentationResult,
    MultiDefectAssessment,
    MultiDefectValidationResult
)


class TestClassificationAPI(unittest.TestCase):
    """Test suite for Phase 3 Step 3.2 Classification API."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)

    # -------------------------------------------------------------------------
    # Test A & G & H: Valid Single-Defect Request
    # -------------------------------------------------------------------------
    def test_valid_single_defect_request(self):
        """A, G, H. Verifies single-defect request returns HTTP 200, category, and confidence."""
        payload = {
            "description": "Motor is making a grinding noise.",
            "mode": "local"
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data.get("success"))
        inner = data.get("data", {})
        self.assertFalse(inner.get("is_multi_defect"))
        self.assertEqual(inner.get("defect_count"), 1)
        self.assertEqual(inner.get("category"), "Mechanical Fault")
        self.assertEqual(inner.get("status"), "success")
        self.assertIn("grinding", inner.get("explanation", "").lower())
        self.assertEqual(inner.get("classification_mode"), "local")
        self.assertEqual(inner.get("provider"), "local")

        # Confidence assessment
        ca = inner.get("confidence_assessment")
        self.assertIsInstance(ca, dict)
        self.assertIn(ca.get("level"), ["High", "Medium", "Low"])
        self.assertIn("approximate_range", ca)
        self.assertIn("is_calibrated", ca)
        self.assertIn("is_ambiguous", ca)

        # Defects list contains exactly 1 defect matching category
        defects = inner.get("defects", [])
        self.assertEqual(len(defects), 1)
        self.assertEqual(defects[0].get("category"), "Mechanical Fault")
        self.assertEqual(defects[0].get("defect_id"), 1)

    # -------------------------------------------------------------------------
    # Test B, I, J, K: Valid Multi-Defect Request
    # -------------------------------------------------------------------------
    def test_valid_multi_defect_request(self):
        """B, I, J, K. Verifies multi-defect request returns HTTP 200, defect_count, categories, and spans."""
        payload = {
            "description": "The conveyor motor is making a grinding noise and the temperature sensor gives incorrect readings.",
            "mode": "local"
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data.get("success"))
        inner = data.get("data", {})
        self.assertTrue(inner.get("is_multi_defect"))
        self.assertEqual(inner.get("defect_count"), 2)

        defects = inner.get("defects", [])
        self.assertEqual(len(defects), 2)

        categories = [d.get("category") for d in defects]
        self.assertIn("Mechanical Fault", categories)
        self.assertIn("Sensor Fault", categories)

        # Spans check
        for d in defects:
            self.assertIn("source_start_char", d)
            self.assertIn("source_end_char", d)
            self.assertGreaterEqual(d["source_end_char"], d["source_start_char"])
            self.assertTrue(len(d["text"]) > 0)
            self.assertIn("confidence_assessment", d)
            self.assertIn("reliability", d)
            self.assertIn("explanation", d)

    # -------------------------------------------------------------------------
    # Test C: Empty Description Rejected
    # -------------------------------------------------------------------------
    def test_empty_description_rejected(self):
        """C. Verifies empty description is rejected with HTTP 422."""
        payload = {"description": "", "mode": "local"}
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 422)
        err = response.json()
        self.assertIn("error", err)
        self.assertEqual(err["error"]["code"], "VALIDATION_ERROR")

    # -------------------------------------------------------------------------
    # Test D: Whitespace-Only Description Rejected
    # -------------------------------------------------------------------------
    def test_whitespace_only_description_rejected(self):
        """D. Verifies whitespace-only description is rejected with HTTP 422."""
        payload = {"description": "   \n\t   ", "mode": "local"}
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 422)
        err = response.json()
        self.assertIn("error", err)
        self.assertEqual(err["error"]["code"], "VALIDATION_ERROR")

    # -------------------------------------------------------------------------
    # Test E: Overly Long Description Rejected
    # -------------------------------------------------------------------------
    def test_overly_long_description_rejected(self):
        """E. Verifies descriptions exceeding 2,000 characters are rejected with HTTP 422."""
        payload = {"description": "a" * 2001, "mode": "local"}
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 422)
        err = response.json()
        self.assertIn("error", err)
        self.assertEqual(err["error"]["code"], "VALIDATION_ERROR")

    # -------------------------------------------------------------------------
    # Test Missing Description Rejected
    # -------------------------------------------------------------------------
    def test_missing_description_rejected(self):
        """Verifies missing description field is rejected with HTTP 422."""
        payload = {"mode": "local"}
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 422)

    # -------------------------------------------------------------------------
    # Test F: Invalid Mode Rejected
    # -------------------------------------------------------------------------
    def test_invalid_mode_rejected(self):
        """F. Verifies unsupported classification modes are rejected with HTTP 422."""
        payload = {
            "description": "Motor is making a grinding noise.",
            "mode": "unsupported_super_ai"
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 422)
        err = response.json()
        self.assertIn("error", err)
        self.assertEqual(err["error"]["code"], "VALIDATION_ERROR")

    # -------------------------------------------------------------------------
    # Test L: Unknown Input Returns Unknown Rather Than Crashing
    # -------------------------------------------------------------------------
    def test_unknown_input_returns_unknown(self):
        """L. Verifies vague or insufficient input returns Unknown without crashing."""
        payload = {
            "description": "Something is wrong.",
            "mode": "local"
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertTrue(data.get("success"))
        inner = data.get("data", {})
        self.assertEqual(inner.get("category"), "Unknown")
        self.assertEqual(inner.get("status"), "unknown")
        self.assertEqual(inner.get("defect_count"), 0)
        self.assertFalse(inner.get("is_multi_defect"))
        self.assertEqual(inner.get("defects"), [])

        ca = inner.get("confidence_assessment")
        self.assertIsNotNone(ca)
        self.assertEqual(ca.get("level"), "Uncertain")
        self.assertEqual(ca.get("approximate_range"), "Insufficient Evidence")

    # -------------------------------------------------------------------------
    # Test M: Ambiguous Input Preserves Ambiguity Information
    # -------------------------------------------------------------------------
    def test_ambiguous_input_preserves_ambiguity(self):
        """M. Verifies ambiguous input preserves structured ambiguity assessment."""
        # Using a mock classifier to deterministically test ambiguity structure in the API layer
        mock_result = ClassificationResult(
            category="Mechanical Fault",
            reason="Ambiguous between Mechanical and Electrical.",
            language="English",
            reliability="Low",
            status="low_confidence",
            original_description="Motor vibrating and sparks flying from breaker.",
            normalized_description="motor vibrating and sparks flying from breaker.",
            model_source="local_ml",
            confidence=0.55,
            confidence_assessment=ConfidenceAssessment(
                level="Low",
                approximate_range="~55%",
                raw_score=0.55,
                calibrated_prob=0.55,
                top2_margin=0.04,
                is_calibrated=True,
                calibration_method="Temperature Scaling",
                is_ambiguous=True
            ),
            ambiguity_assessment=AmbiguityAssessment(
                is_ambiguous=True,
                reason="Competing technical signals between Mechanical Fault and Electrical Fault.",
                top_category="Mechanical Fault",
                competing_category="Electrical Fault",
                margin=0.04,
                evidence_summary="Mechanical Fault vs Electrical Fault",
                method="competing_category_margin"
            )
        )

        with patch("api.routes.classification.get_classifier_service") as mock_get_clf:
            mock_clf = MagicMock()
            mock_clf.classify.return_value = mock_result
            mock_get_clf.return_value = mock_clf

            # Re-instantiate test client or send request
            from api.dependencies import get_classifier_service
            app.dependency_overrides[get_classifier_service] = lambda: mock_clf
            try:
                response = self.client.post("/api/v1/classify", json={
                    "description": "Motor vibrating and sparks flying from breaker.",
                    "mode": "local"
                })
                self.assertEqual(response.status_code, 200)
                data = response.json()
                inner = data["data"]
                aa = inner.get("ambiguity_assessment")
                self.assertIsNotNone(aa)
                self.assertTrue(aa["is_ambiguous"])
                self.assertEqual(aa["top_category"], "Mechanical Fault")
                self.assertEqual(aa["competing_category"], "Electrical Fault")
                self.assertEqual(aa["margin"], 0.04)
            finally:
                app.dependency_overrides.pop(get_classifier_service, None)

    # -------------------------------------------------------------------------
    # Test N: Unicode Native Telugu Script Input
    # -------------------------------------------------------------------------
    def test_unicode_telugu_input(self):
        """N. Verifies native Telugu script input is accepted and preserved."""
        telugu_text = "మోటార్ ఎక్కువగా వైబ్రేట్ అవుతోంది మరియు శబ్దం వస్తోంది."
        payload = {
            "description": telugu_text,
            "mode": "local"
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        inner = data["data"]
        self.assertEqual(inner["original_text"], telugu_text)
        self.assertEqual(inner["category"], "Mechanical Fault")

    # -------------------------------------------------------------------------
    # Test O: Code-Switched Telugu-English Input
    # -------------------------------------------------------------------------
    def test_code_switched_input(self):
        """O. Verifies Telugu-English code-switched input is accepted and preserved."""
        cs_text = "Motor lo unusual sound vastundi, heavy vibration undi."
        payload = {
            "description": cs_text,
            "mode": "local"
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        inner = data["data"]
        self.assertEqual(inner["original_text"], cs_text)
        self.assertEqual(inner["category"], "Mechanical Fault")

    # -------------------------------------------------------------------------
    # Test P: API Does Not Expose Secrets
    # -------------------------------------------------------------------------
    def test_api_does_not_expose_secrets(self):
        """P. Verifies that API responses never leak environment keys or credentials."""
        response = self.client.post("/api/v1/classify", json={
            "description": "Motor is making a grinding noise.",
            "mode": "local"
        })
        self.assertEqual(response.status_code, 200)
        content_str = response.text

        # Forbidden secret patterns
        forbidden = [
            "AIzaSy", "supabase_key", "SUPABASE_KEY", "GEMINI_API_KEY",
            "AIMLAPI_KEY", "sk-", "password", "postgresql://"
        ]
        for secret in forbidden:
            self.assertNotIn(secret, content_str)

    # -------------------------------------------------------------------------
    # Test Q: Classification Endpoint Does NOT Write to Supabase
    # -------------------------------------------------------------------------
    def test_classification_does_not_write_to_supabase(self):
        """Q. Critical: Verifies POST /api/v1/classify performs zero writes to Supabase."""
        with patch("persistence.repository.save_defect_report") as mock_save_single, \
             patch("persistence.repository.save_multi_defect_report") as mock_save_multi, \
             patch("database.supabase_client.get_supabase_client") as mock_client:

            # 1. Test single defect
            resp_single = self.client.post("/api/v1/classify", json={
                "description": "Motor is making a grinding noise.",
                "mode": "local"
            })
            self.assertEqual(resp_single.status_code, 200)

            # 2. Test multi defect
            resp_multi = self.client.post("/api/v1/classify", json={
                "description": "The conveyor motor is making a grinding noise and the temperature sensor gives incorrect readings.",
                "mode": "local"
            })
            self.assertEqual(resp_multi.status_code, 200)

            # 3. Test unknown defect
            resp_unk = self.client.post("/api/v1/classify", json={
                "description": "Something is wrong.",
                "mode": "local"
            })
            self.assertEqual(resp_unk.status_code, 200)

            # Assert absolutely ZERO persistence calls were made
            mock_save_single.assert_not_called()
            mock_save_multi.assert_not_called()
            mock_client.assert_not_called()

    # -------------------------------------------------------------------------
    # Test R: Classifier-Level Errors are Sanitized
    # -------------------------------------------------------------------------
    def test_classifier_error_sanitization(self):
        """R. Verifies configuration_error, model_error, and system_error are safely sanitized."""
        from api.dependencies import get_classifier_service

        # 1. Configuration Error -> HTTP 503
        mock_cfg_res = ClassificationResult(
            category="Unknown",
            reason="Gemini API key is missing or invalid.",
            language="English",
            reliability="Low",
            status="configuration_error",
            original_description="Motor broken",
            normalized_description="motor broken",
            error_message="Configuration error: GEMINI_API_KEY unset."
        )
        mock_clf_cfg = MagicMock()
        mock_clf_cfg.classify.return_value = mock_cfg_res
        app.dependency_overrides[get_classifier_service] = lambda: mock_clf_cfg
        try:
            resp = self.client.post("/api/v1/classify", json={"description": "Motor broken", "mode": "gemini"})
            self.assertEqual(resp.status_code, 503)
            self.assertIn("provider is not properly configured", resp.json()["error"]["message"])
            self.assertNotIn("GEMINI_API_KEY", resp.text)
        finally:
            app.dependency_overrides.pop(get_classifier_service, None)

        # 2. Model Error -> HTTP 502
        mock_model_res = ClassificationResult(
            category="Unknown",
            reason="External service timed out.",
            language="English",
            reliability="Low",
            status="model_error",
            original_description="Motor broken",
            normalized_description="motor broken",
            error_message="HTTPSConnectionPool timeout to api.aimlapi.com"
        )
        mock_clf_mod = MagicMock()
        mock_clf_mod.classify.return_value = mock_model_res
        app.dependency_overrides[get_classifier_service] = lambda: mock_clf_mod
        try:
            resp = self.client.post("/api/v1/classify", json={"description": "Motor broken", "mode": "gemini"})
            self.assertEqual(resp.status_code, 502)
            self.assertIn("unavailable", resp.json()["error"]["message"])
            self.assertNotIn("HTTPSConnectionPool", resp.text)
        finally:
            app.dependency_overrides.pop(get_classifier_service, None)

        # 3. Uncaught Exception -> HTTP 500
        mock_clf_exc = MagicMock()
        mock_clf_exc.classify.side_effect = RuntimeError("Fatal crash in internal ML tensor core")
        app.dependency_overrides[get_classifier_service] = lambda: mock_clf_exc
        try:
            resp = self.client.post("/api/v1/classify", json={"description": "Motor broken", "mode": "local"})
            self.assertEqual(resp.status_code, 500)
            self.assertNotIn("tensor core", resp.text)
            self.assertNotIn("RuntimeError", resp.text)
        finally:
            app.dependency_overrides.pop(get_classifier_service, None)

    # -------------------------------------------------------------------------
    # Test S: OpenAPI Documentation Verification
    # -------------------------------------------------------------------------
    def test_openapi_schema_contains_classification(self):
        """S. Verifies /openapi.json properly registers POST /api/v1/classify with schemas."""
        resp = self.client.get("/openapi.json")
        self.assertEqual(resp.status_code, 200)
        schema = resp.json()

        paths = schema.get("paths", {})
        self.assertIn("/api/v1/classify", paths)

        classify_op = paths["/api/v1/classify"].get("post", {})
        self.assertEqual(classify_op.get("summary"), "Classify Industrial Defect")
        self.assertIn("Classification", classify_op.get("tags", []))

        # Check response definitions
        responses = classify_op.get("responses", {})
        self.assertIn("200", responses)
        self.assertIn("422", responses)


if __name__ == "__main__":
    unittest.main()
