"""Integration Tests for Step 3.7 Classifier Engine Integration.

Verifies the integration boundary between:
FastAPI -> Classifier Service -> Domain Classification Pipelines -> API Response Schema

Covers all 20 required dimensions:
A. Authoritative classifier entry point is used
B. Single-defect local classification
C. Unknown classification
D. Ambiguous classification
E. Multi-defect classification
F. Per-defect metadata preservation
G. Local mode makes no external call
H. Gemini integration boundary using mocks
I. Hybrid integration using mocks
J. Provider failure propagation
K. Classifier failure propagation
L. No Supabase writes (stateless)
M. API response schema compatibility
N. Taxonomy compliance
O. Confidence structure preservation
P. Stateless classification
Q. Existing classification endpoint regression
R. Unicode / Telugu native script input
S. Telugu-English code-switched input
T. OpenAPI classification schema remains valid
"""

import json
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.main import app, create_app
from api.dependencies import get_classifier_service
from api.schemas.classification import ClassificationResponse, ClassificationRequest
from classification.classifier import DefectClassifier, get_defect_classifier
from core.schemas import (
    ClassificationResult,
    ConfidenceAssessment,
    AmbiguityAssessment,
    PerDefectClassification,
    MultiDefectClassificationResult,
    MultiDefectValidationResult,
    DefectSegment,
    DefectSegmentationResult,
    MultiDefectAssessment
)
from taxonomy.repository import get_taxonomy_repository, REQUIRED_APPROVED_CATEGORIES
from core.exceptions import ModelError, ConfigurationError


class TestClassifierEngineIntegration(unittest.TestCase):
    """Test suite for Phase 3 Step 3.7 Classifier Engine Integration."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)
        cls.taxonomy_repo = get_taxonomy_repository()
        cls.approved_categories = set(cls.taxonomy_repo.get_categories())

    # -------------------------------------------------------------------------
    # Test A: Authoritative Classifier Entry Point Is Used
    # -------------------------------------------------------------------------
    def test_authoritative_classifier_entry_point(self):
        """A. Verifies get_classifier_service provides DefectClassifier instance and is used by API."""
        service = get_classifier_service()
        self.assertIsInstance(service, DefectClassifier)

        # Verify dependency injection can be resolved through FastAPI app
        with patch.object(service, "classify", wraps=service.classify) as spy_classify:
            payload = {"description": "Motor is making a grinding noise.", "mode": "local"}
            resp = self.client.post("/api/v1/classify", json=payload)
            self.assertEqual(resp.status_code, 200)
            spy_classify.assert_called_once()
            args, kwargs = spy_classify.call_args
            self.assertEqual(kwargs.get("raw_defect_text"), "Motor is making a grinding noise.")
            self.assertEqual(kwargs.get("mode_override"), "local")

    # -------------------------------------------------------------------------
    # Test B: Single-Defect Local Classification
    # -------------------------------------------------------------------------
    def test_single_defect_local_classification(self):
        """B. Verifies single-defect request maps correctly through engine and preserves metadata."""
        payload = {
            "description": "Motor is making a grinding noise.",
            "mode": "local"
        }
        resp = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(resp.status_code, 200)

        body = resp.json()
        self.assertTrue(body.get("success"))
        data = body.get("data", {})
        self.assertEqual(data.get("category"), "Mechanical Fault")
        self.assertEqual(data.get("status"), "success")
        self.assertFalse(data.get("is_multi_defect"))
        self.assertEqual(data.get("defect_count"), 1)
        self.assertEqual(data.get("classification_mode"), "local")
        self.assertEqual(data.get("provider"), "local")
        self.assertIn("Local ML", data.get("model", ""))

        # Confidence fields
        self.assertIsNotNone(data.get("confidence"))
        self.assertIn(data.get("confidence_level"), ["High", "Medium", "Low", "Uncertain"])
        self.assertIsNotNone(data.get("confidence_range"))
        self.assertIsNotNone(data.get("raw_score"))
        self.assertIsNotNone(data.get("calibrated_score"))
        self.assertIsNotNone(data.get("top2_margin"))
        self.assertIn("grinding", data.get("explanation", "").lower())

    # -------------------------------------------------------------------------
    # Test C: Unknown Classification
    # -------------------------------------------------------------------------
    def test_unknown_classification_preserved(self):
        """C. Verifies non-defect/vague input returns Unknown without fabricating another category."""
        payload = {
            "description": "Something is wrong.",
            "mode": "local"
        }
        resp = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(resp.status_code, 200)

        body = resp.json()
        self.assertTrue(body.get("success"))
        data = body.get("data", {})
        self.assertEqual(data.get("category"), "Unknown")
        self.assertEqual(data.get("overall_status"), "unknown")
        self.assertEqual(data.get("defect_count"), 0)
        self.assertFalse(data.get("is_multi_defect"))
        self.assertEqual(data.get("defects"), [])
        self.assertIn("vague", data.get("explanation", "").lower())

    # -------------------------------------------------------------------------
    # Test D: Ambiguous Classification
    # -------------------------------------------------------------------------
    def test_ambiguity_information_preservation(self):
        """D. Verifies ambiguity assessments reach API response without fake numerical probabilities."""
        ambiguity_assessment = AmbiguityAssessment(
            is_ambiguous=True,
            reason="Equal signal strength between Mechanical Fault and Electrical Fault.",
            top_category="Mechanical Fault",
            competing_category="Electrical Fault",
            margin=0.04,
            evidence_summary="Motor vibration and wiring spark",
            method="competing_textual_evidence"
        )
        mock_result = ClassificationResult(
            category="Mechanical Fault",
            confidence=0.48,
            status="success",
            reason="Ambiguous signals detected between mechanical and electrical components.",
            reliability="Low",
            language="en",
            original_description="Motor vibration and wiring spark detected.",
            normalized_description="Motor vibration and wiring spark detected.",
            model_source="local_ml",
            confidence_assessment=ConfidenceAssessment(
                level="Low",
                approximate_range="40-50%",
                raw_score=0.48,
                calibrated_prob=0.48,
                top2_margin=0.04,
                is_calibrated=True,
                calibration_method="Temperature Scaling",
                is_ambiguous=True
            ),
            ambiguity_assessment=ambiguity_assessment
        )

        with patch("classification.classifier.DefectClassifier.classify", return_value=mock_result):
            payload = {
                "description": "Motor vibration and wiring spark detected.",
                "mode": "local"
            }
            resp = self.client.post("/api/v1/classify", json=payload)
            self.assertEqual(resp.status_code, 200)

            data = resp.json().get("data", {})
            aa = data.get("ambiguity_assessment")
            self.assertIsNotNone(aa)
            self.assertTrue(aa.get("is_ambiguous"))
            self.assertEqual(aa.get("top_category"), "Mechanical Fault")
            self.assertEqual(aa.get("competing_category"), "Electrical Fault")
            self.assertAlmostEqual(aa.get("margin"), 0.04, places=2)

    # -------------------------------------------------------------------------
    # Test E: Multi-Defect Classification
    # -------------------------------------------------------------------------
    def test_multidefect_classification_integration(self):
        """E. Verifies co-occurring multi-defect descriptions partition into separate child categories."""
        payload = {
            "description": "Motor is making a grinding noise and the temperature sensor gives incorrect readings.",
            "mode": "local"
        }
        resp = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json().get("data", {})
        self.assertTrue(data.get("is_multi_defect"))
        self.assertEqual(data.get("defect_count"), 2)
        defects = data.get("defects", [])
        self.assertEqual(len(defects), 2)

        categories = [d.get("category") for d in defects]
        self.assertIn("Mechanical Fault", categories)
        self.assertIn("Sensor Fault", categories)

    # -------------------------------------------------------------------------
    # Test F: Per-Defect Metadata Preservation
    # -------------------------------------------------------------------------
    def test_per_defect_metadata_preservation(self):
        """F. Verifies all required per-defect metadata attributes are preserved in each child segment."""
        payload = {
            "description": "Motor is making a grinding noise and the temperature sensor gives incorrect readings.",
            "mode": "local"
        }
        resp = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json().get("data", {})
        defects = data.get("defects", [])
        for d in defects:
            self.assertIn("defect_id", d)
            self.assertIn("segment_id", d)
            self.assertIn("text", d)
            self.assertIn("category", d)
            self.assertIn(d["category"], self.approved_categories)
            self.assertIn("confidence_assessment", d)
            self.assertIn("reliability", d)
            self.assertIn("explanation", d)
            self.assertIn("classification_mode", d)
            self.assertIn("provider", d)
            self.assertIn("model", d)
            self.assertIn("status", d)
            self.assertIn("source_start_char", d)
            self.assertIn("source_end_char", d)

    # -------------------------------------------------------------------------
    # Test G: Local Mode Makes No External Call
    # -------------------------------------------------------------------------
    def test_local_mode_makes_no_external_call(self):
        """G. Proves LOCAL mode makes zero external API or LLM calls."""
        with patch("llm.client.BaseLLMClient.classify") as mock_llm_call:
            payload = {
                "description": "Conveyor roller bearing damaged and vibrating.",
                "mode": "local"
            }
            resp = self.client.post("/api/v1/classify", json=payload)
            self.assertEqual(resp.status_code, 200)
            mock_llm_call.assert_not_called()

    # -------------------------------------------------------------------------
    # Test H: Gemini Integration Boundary Using Mocks
    # -------------------------------------------------------------------------
    def test_gemini_integration_boundary_mock(self):
        """H. Verifies Gemini mode maps through LLM client and preserves qualitative tiers."""
        mock_result = ClassificationResult(
            category="Electrical Fault",
            confidence=None,  # Zero fake numerical score for LLM
            status="success",
            reason="High voltage arching detected in breaker panel.",
            reliability="High",
            language="en",
            original_description="High voltage arching detected in breaker panel.",
            normalized_description="High voltage arching detected in breaker panel.",
            model_source="gemini",
            confidence_assessment=ConfidenceAssessment(
                level="High",
                approximate_range="High Confidence Tier",
                raw_score=None,
                calibrated_prob=None,
                top2_margin=None,
                is_calibrated=False,
                calibration_method=None,
                is_ambiguous=False
            )
        )

        with patch("classification.classifier.DefectClassifier.classify", return_value=mock_result) as mock_clf:
            payload = {
                "description": "High voltage arching detected in breaker panel.",
                "mode": "gemini"
            }
            resp = self.client.post("/api/v1/classify", json=payload)
            self.assertEqual(resp.status_code, 200)

            mock_clf.assert_called_once_with(
                raw_defect_text="High voltage arching detected in breaker panel.",
                mode_override="gemini"
            )
            data = resp.json().get("data", {})
            self.assertEqual(data.get("category"), "Electrical Fault")
            self.assertEqual(data.get("provider"), "gemini")
            self.assertEqual(data.get("reliability"), "High")
            self.assertIsNone(data.get("calibrated_score"))

    # -------------------------------------------------------------------------
    # Test I: Hybrid Integration Using Mocks
    # -------------------------------------------------------------------------
    def test_hybrid_integration_behavior(self):
        """I. Verifies Hybrid mode falls back to external LLM and falls back to local if external fails."""
        service = get_classifier_service()

        # Case 1: High confidence local -> No external LLM call
        with patch.object(service, "_classify_gemini") as mock_gemini:
            payload = {
                "description": "Motor is making a grinding noise.",
                "mode": "hybrid"
            }
            resp = self.client.post("/api/v1/classify", json=payload)
            self.assertEqual(resp.status_code, 200)
            mock_gemini.assert_not_called()

        # Case 2: External provider failure during hybrid fallback -> Safely returns local prediction
        low_conf_local = ClassificationResult(
            category="Mechanical Fault",
            confidence=0.35,  # Below default threshold (0.60)
            status="success",
            reason="Low confidence mechanical signal.",
            reliability="Low",
            language="en",
            original_description="Uncertain machine rumbling sound.",
            normalized_description="Uncertain machine rumbling sound.",
            model_source="local_ml",
            confidence_assessment=ConfidenceAssessment(
                level="Low",
                approximate_range="30-40%",
                raw_score=0.35,
                calibrated_prob=0.35,
                top2_margin=0.02,
                is_calibrated=True,
                is_ambiguous=False
            )
        )
        with patch.object(service.local_classifier, "classify", return_value=low_conf_local):
            with patch.object(service, "_classify_gemini", side_effect=ModelError("Gemini API 503 Unavailable")):
                payload = {
                    "description": "Uncertain machine rumbling sound.",
                    "mode": "hybrid"
                }
                resp = self.client.post("/api/v1/classify", json=payload)
                self.assertEqual(resp.status_code, 200)
                data = resp.json().get("data", {})
                self.assertEqual(data.get("category"), "Mechanical Fault")
                self.assertEqual(data.get("provider"), "local")

    # -------------------------------------------------------------------------
    # Test J: Provider Failure Propagation
    # -------------------------------------------------------------------------
    def test_provider_failure_propagation(self):
        """J. Verifies provider errors map cleanly to HTTP 502/503 PROVIDER_ERROR without leaks."""
        # 1. Configuration error (e.g. missing API key) -> 503
        config_err_result = ClassificationResult(
            category="Unknown",
            confidence=None,
            status="configuration_error",
            reason="GEMINI_API_KEY is not configured.",
            reliability="Uncertain",
            language="en",
            original_description="Motor bearing failed.",
            normalized_description="Motor bearing failed.",
            model_source="gemini"
        )
        with patch("classification.classifier.DefectClassifier.classify", return_value=config_err_result):
            payload = {"description": "Motor bearing failed.", "mode": "gemini"}
            resp = self.client.post("/api/v1/classify", json=payload)
            self.assertEqual(resp.status_code, 503)
            body = resp.json()
            self.assertFalse(body.get("success"))
            self.assertEqual(body.get("error", {}).get("code"), "PROVIDER_ERROR")

        # 2. Model service error (e.g. upstream timeout or quota) -> 502
        model_err_result = ClassificationResult(
            category="Unknown",
            confidence=None,
            status="model_error",
            reason="Upstream provider quota exceeded.",
            reliability="Uncertain",
            language="en",
            original_description="Motor bearing failed.",
            normalized_description="Motor bearing failed.",
            model_source="gemini"
        )
        with patch("classification.classifier.DefectClassifier.classify", return_value=model_err_result):
            payload = {"description": "Motor bearing failed.", "mode": "gemini"}
            resp = self.client.post("/api/v1/classify", json=payload)
            self.assertEqual(resp.status_code, 502)
            body = resp.json()
            self.assertFalse(body.get("success"))
            self.assertEqual(body.get("error", {}).get("code"), "PROVIDER_ERROR")

    # -------------------------------------------------------------------------
    # Test K: Classifier Failure Propagation
    # -------------------------------------------------------------------------
    def test_classifier_failure_propagation(self):
        """K. Verifies unexpected classifier runtime failures map to sanitized HTTP 500."""
        with patch("classification.classifier.DefectClassifier.classify", side_effect=RuntimeError("CUDA device crash")):
            payload = {"description": "Motor bearing failed.", "mode": "local"}
            resp = self.client.post("/api/v1/classify", json=payload)
            self.assertEqual(resp.status_code, 500)
            body = resp.json()
            self.assertFalse(body.get("success"))
            self.assertEqual(body.get("error", {}).get("code"), "INTERNAL_SERVER_ERROR")
            # Verify internal stack trace is not leaked
            self.assertNotIn("CUDA device crash", body.get("error", {}).get("message", ""))

    # -------------------------------------------------------------------------
    # Test L: No Supabase Writes
    # -------------------------------------------------------------------------
    def test_no_supabase_writes(self):
        """L. Verifies POST /api/v1/classify does not call save_defect_report, save_multi_defect_report, or execute DB queries."""
        with patch("persistence.repository.save_defect_report") as mock_save_single:
            with patch("persistence.repository.save_multi_defect_report") as mock_save_multi:
                # 1. Single defect request
                self.client.post("/api/v1/classify", json={"description": "Motor is making a grinding noise.", "mode": "local"})
                mock_save_single.assert_not_called()
                mock_save_multi.assert_not_called()

                # 2. Multi-defect request
                self.client.post(
                    "/api/v1/classify",
                    json={"description": "Motor is making a grinding noise and temperature sensor gives incorrect readings.", "mode": "local"}
                )
                mock_save_single.assert_not_called()
                mock_save_multi.assert_not_called()

                # 3. Unknown request
                self.client.post("/api/v1/classify", json={"description": "Something is wrong.", "mode": "local"})
                mock_save_single.assert_not_called()
                mock_save_multi.assert_not_called()

    # -------------------------------------------------------------------------
    # Test P: Stateless Classification
    # -------------------------------------------------------------------------
    def test_stateless_classification(self):
        """P. Verifies classification endpoint remains stateless across consecutive distinct requests."""
        resp1 = self.client.post("/api/v1/classify", json={"description": "Motor is making a grinding noise.", "mode": "local"})
        resp2 = self.client.post("/api/v1/classify", json={"description": "Something is wrong.", "mode": "local"})
        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(resp1.json()["data"]["category"], "Mechanical Fault")
        self.assertEqual(resp2.json()["data"]["category"], "Unknown")


    # -------------------------------------------------------------------------
    # Test M: API Response Schema Compatibility
    # -------------------------------------------------------------------------
    def test_api_response_schema_compatibility(self):
        """M. Verifies response JSON strictly validates against Pydantic ClassificationResponse schema."""
        payload = {"description": "Motor is making a grinding noise.", "mode": "local"}
        resp = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(resp.status_code, 200)

        # Validate directly with Pydantic model
        validated_response = ClassificationResponse.model_validate(resp.json())
        self.assertTrue(validated_response.success)
        self.assertEqual(validated_response.data.category, "Mechanical Fault")

    # -------------------------------------------------------------------------
    # Test N: Taxonomy Compliance
    # -------------------------------------------------------------------------
    def test_taxonomy_compliance(self):
        """N. Verifies returned classification categories strictly belong to canonical 8 categories."""
        payloads = [
            {"description": "Motor is making a grinding noise.", "mode": "local"},
            {"description": "Something is wrong.", "mode": "local"},
            {"description": "Motor is making a grinding noise and temperature sensor gives incorrect readings.", "mode": "local"}
        ]
        for p in payloads:
            resp = self.client.post("/api/v1/classify", json=p)
            self.assertEqual(resp.status_code, 200)
            data = resp.json().get("data", {})
            cat = data.get("category")
            if cat is not None:
                self.assertIn(cat, self.approved_categories)
            for child in data.get("defects", []):
                self.assertIn(child.get("category"), self.approved_categories)

    # -------------------------------------------------------------------------
    # Test O: Confidence Structure Preservation
    # -------------------------------------------------------------------------
    def test_confidence_structure_preservation(self):
        """O. Verifies calibrated probabilities are bounded within [0, 1] and qualitative tiers preserved."""
        resp = self.client.post("/api/v1/classify", json={"description": "Motor is making a grinding noise.", "mode": "local"})
        self.assertEqual(resp.status_code, 200)
        ca = resp.json().get("data", {}).get("confidence_assessment")
        self.assertIsNotNone(ca)
        prob = ca.get("calibrated_prob")
        if prob is not None:
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
        self.assertIn(ca.get("level"), ["High", "Medium", "Low", "Uncertain"])

    # -------------------------------------------------------------------------
    # Test Q: Existing Classification Endpoint Regression
    # -------------------------------------------------------------------------
    def test_existing_classification_endpoint_regression(self):
        """Q. Verifies endpoint remains backward-compatible with standard requests."""
        resp = self.client.post("/api/v1/classify", json={"description": "Conveyor belt is slipping and overheating."})
        self.assertEqual(resp.status_code, 200)
        data = resp.json().get("data", {})
        self.assertTrue(data.get("status") in ["success", "unknown"])

    # -------------------------------------------------------------------------
    # Test R: Unicode / Telugu Native Script Input
    # -------------------------------------------------------------------------
    def test_unicode_telugu_native_script(self):
        """R. Verifies native Telugu script input is accepted and classified accurately."""
        payload = {
            "description": "మోటారు గ్రైండింగ్ శబ్దం చేస్తోంది",
            "mode": "local"
        }
        resp = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json().get("data", {})
        self.assertEqual(data.get("original_text"), "మోటారు గ్రైండింగ్ శబ్దం చేస్తోంది")
        self.assertEqual(data.get("category"), "Mechanical Fault")

    # -------------------------------------------------------------------------
    # Test S: Telugu-English Code-Switched Input
    # -------------------------------------------------------------------------
    def test_telugu_english_code_switching(self):
        """S. Verifies code-switched Telugu-English input is accepted and classified accurately."""
        payload = {
            "description": "Motor grinding sound vasthondi",
            "mode": "local"
        }
        resp = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json().get("data", {})
        self.assertEqual(data.get("original_text"), "Motor grinding sound vasthondi")
        self.assertEqual(data.get("category"), "Mechanical Fault")

    # -------------------------------------------------------------------------
    # Test T: OpenAPI Classification Schema Remains Valid
    # -------------------------------------------------------------------------
    def test_openapi_classification_schema_remains_valid(self):
        """T. Verifies OpenAPI schema accurately registers POST /api/v1/classify with components."""
        schema = app.openapi()
        paths = schema.get("paths", {})
        self.assertIn("/api/v1/classify", paths)
        op = paths["/api/v1/classify"].get("post", {})
        self.assertEqual(op.get("summary"), "Classify Industrial Defect")
        self.assertIn("200", op.get("responses", {}))
        self.assertIn("422", op.get("responses", {}))
        self.assertIn("500", op.get("responses", {}))
        self.assertIn("502", op.get("responses", {}))
        self.assertIn("503", op.get("responses", {}))


if __name__ == "__main__":
    unittest.main()
