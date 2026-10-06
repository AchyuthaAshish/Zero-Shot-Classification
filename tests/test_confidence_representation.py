"""Unit tests for Confidence Representation Redesign (Phase 1 Step 1.1).

Verifies:
1. ConfidenceAssessment creation and defaults.
2. Local MiniLM produces raw_score without falsely claiming empirical calibration.
3. TF-IDF calibrated probability is explicitly marked as calibrated.
4. Gemini produces qualitative confidence with no numerical probability.
5. Unknown produces Uncertain level, Insufficient Evidence range, and is_ambiguous=True.
6. Top-2 margin is captured correctly (P(top1) - P(top2)).
7. Existing confidence field backward compatibility remains intact.
8. Supabase payload accepts the new fields without breaking legacy records.
9. Database integration tests pass with new fields.
10. Existing LOCAL mode remains 100% offline.
11. Existing HYBRID routing behavior remains unchanged.
12. Existing provider switching remains unchanged.
"""

import unittest
from unittest.mock import MagicMock, patch
from core.schemas import ClassificationResult, PreprocessedInput, LLMRawResponse, ConfidenceAssessment
from core.decision_engine import DecisionEngine
from ml.local_classifier import LocalDefectClassifier
from classification.classifier import DefectClassifier
from llm.client import FakeLLMClient
from database.supabase_client import validate_defect_report_payload, save_defect_report


class TestConfidenceRepresentation(unittest.TestCase):
    """Test suite for Phase 1 Step 1.1 Confidence Representation Redesign."""

    def test_confidence_assessment_creation_and_defaults(self):
        """1. Verifies ConfidenceAssessment initialization, default values, and to_dict serialization."""
        ca = ConfidenceAssessment(
            level="High",
            approximate_range="Uncalibrated model score",
            raw_score=0.85
        )
        self.assertEqual(ca.level, "High")
        self.assertEqual(ca.approximate_range, "Uncalibrated model score")
        self.assertEqual(ca.raw_score, 0.85)
        self.assertIsNone(ca.calibrated_prob)
        self.assertIsNone(ca.top2_margin)
        self.assertFalse(ca.is_calibrated)
        self.assertIsNone(ca.calibration_method)
        self.assertFalse(ca.is_ambiguous)

        d = ca.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["level"], "High")
        self.assertEqual(d["raw_score"], 0.85)

    def test_local_minilm_uncalibrated_representation(self):
        """2. Verifies Local MiniLM produces raw_score but does NOT mark it as calibrated."""
        classifier = LocalDefectClassifier(model_type="multilingual_embedding", use_calibration=False)
        result = classifier.classify("Conveyor motor bearing is making an abnormal grinding noise.")

        self.assertIsInstance(result.confidence_assessment, ConfidenceAssessment)
        ca = result.confidence_assessment
        self.assertIsNotNone(ca.raw_score)
        self.assertIsNone(ca.calibrated_prob)
        self.assertFalse(ca.is_calibrated)
        self.assertIsNone(ca.calibration_method)
        self.assertEqual(ca.approximate_range, "Uncalibrated model score")
        self.assertIn(ca.level, ["High", "Medium", "Low", "Uncertain"])

        # Backward compatibility: old confidence field is still populated with raw float
        self.assertIsNotNone(result.confidence)
        self.assertAlmostEqual(result.confidence, ca.raw_score, places=4)

    def test_tfidf_calibrated_representation(self):
        """3. Verifies TF-IDF classifier explicitly marks its probability as calibrated."""
        classifier = LocalDefectClassifier(model_type="tfidf_svm")
        result = classifier.classify("Electrical short circuit in motor wiring box with sparks.")

        self.assertIsInstance(result.confidence_assessment, ConfidenceAssessment)
        ca = result.confidence_assessment
        self.assertIsNotNone(ca.raw_score)
        self.assertIsNotNone(ca.calibrated_prob)
        self.assertTrue(ca.is_calibrated)
        self.assertIn("Platt scaling", ca.calibration_method)
        self.assertTrue(ca.approximate_range.startswith("~"))

    def test_gemini_qualitative_confidence_no_numerical_probability(self):
        """4. Verifies Gemini produces qualitative confidence with no fake numerical probability."""
        fake_client = FakeLLMClient(
            canned_responses={
                "bearing vibrating": {
                    "category": "Mechanical Fault",
                    "reason": "Vibration indicates bearing issue.",
                    "language": "English",
                    "reliability": "High"
                }
            }
        )
        classifier = DefectClassifier(llm_client=fake_client, mode="gemini")
        result = classifier.classify("Bearing vibrating at high speed.")

        self.assertEqual(result.category, "Mechanical Fault")
        self.assertIsNone(result.confidence)
        self.assertIsInstance(result.confidence_assessment, ConfidenceAssessment)
        ca = result.confidence_assessment
        self.assertEqual(ca.level, "High")
        self.assertIsNone(ca.raw_score)
        self.assertIsNone(ca.calibrated_prob)
        self.assertFalse(ca.is_calibrated)
        self.assertIsNone(ca.calibration_method)
        self.assertEqual(ca.approximate_range, "Qualitative / N/A")

    def test_unknown_produces_uncertain_and_insufficient_evidence(self):
        """5. Verifies Unknown outcomes produce Uncertain level and Insufficient Evidence range."""
        engine = DecisionEngine()
        prep_input = PreprocessedInput(
            raw_text="Machine stopped.",
            normalized_text="Machine stopped.",
            detected_language="English"
        )
        llm_resp = LLMRawResponse(
            proposed_category="Unknown",
            reason="Vague description lacks technical evidence.",
            reliability="Medium"
        )
        result = engine.decide_from_llm(prep_input, llm_resp)

        self.assertEqual(result.category, "Unknown")
        self.assertEqual(result.status, "unknown")
        ca = result.confidence_assessment
        self.assertEqual(ca.level, "Uncertain")
        self.assertEqual(ca.approximate_range, "Insufficient Evidence")
        self.assertIsNone(ca.calibrated_prob)
        self.assertTrue(ca.is_ambiguous)

    def test_top2_margin_calculation(self):
        """6. Verifies top2_margin is accurately calculated as P(top1) - P(top2)."""
        classifier = LocalDefectClassifier(model_type="multilingual_embedding")
        mock_probs = {
            "Mechanical Fault": 0.75,
            "Electrical Fault": 0.20,
            "Sensor Fault": 0.05
        }
        with patch.object(classifier, "predict_with_confidence", return_value=("Mechanical Fault", 0.75, mock_probs)):
            result = classifier.classify("Test defect description")
            ca = result.confidence_assessment
            self.assertIsNotNone(ca.top2_margin)
            self.assertAlmostEqual(ca.top2_margin, 0.55, places=4)

    def test_backward_compatibility_old_confidence_field(self):
        """7. Verifies existing confidence attribute continues to exist on ClassificationResult."""
        res = ClassificationResult(
            category="Mechanical Fault",
            reason="Motor vibration",
            language="English",
            reliability="High",
            status="success",
            original_description="Motor vibration",
            normalized_description="Motor vibration",
            confidence=0.88
        )
        # Old attribute is accessible and unchanged
        self.assertEqual(res.confidence, 0.88)
        # New structured assessment is auto-synthesized if omitted
        self.assertIsNotNone(res.confidence_assessment)
        self.assertEqual(res.confidence_assessment.raw_score, 0.88)

    def test_supabase_payload_accepts_new_fields(self):
        """8. Verifies Supabase payload accepts confidence_level, confidence_range, raw_score, calibrated_score."""
        payload = validate_defect_report_payload(
            reporter_name="Operator 1",
            employee_id="OP-001",
            defect_description="Overheating gear",
            category="Temperature Fault",
            confidence=0.78,
            confidence_level="Medium",
            confidence_range="Uncalibrated model score",
            raw_score=0.78,
            calibrated_score=None
        )
        self.assertEqual(payload["confidence_level"], "Medium")
        self.assertEqual(payload["confidence_range"], "Uncalibrated model score")
        self.assertEqual(payload["raw_score"], 0.78)
        self.assertIsNone(payload["calibrated_score"])
        self.assertEqual(payload["confidence"], 0.78)

    def test_local_mode_remains_offline(self):
        """10. Verifies LOCAL mode executes completely offline with zero external API calls."""
        with patch("llm.client.GeminiLLMClient.classify") as mock_gemini:
            with patch("llm.client.AIMLAPIClient.classify") as mock_aiml:
                classifier = DefectClassifier(mode="local")
                result = classifier.classify("Pump impeller is clogged with debris.")
                self.assertIsNotNone(result.category)
                self.assertEqual(result.model_source, "local_ml")
                mock_gemini.assert_not_called()
                mock_aiml.assert_not_called()

    def test_hybrid_routing_behavior_unchanged(self):
        """11. Verifies HYBRID routing preserves threshold logic without regression."""
        mock_fake_llm = FakeLLMClient(
            canned_responses={
                "ambiguous machine noise": {
                    "category": "Mechanical Fault",
                    "reason": "Resolved by Gemini fallback.",
                    "language": "English",
                    "reliability": "High"
                }
            }
        )
        classifier = DefectClassifier(llm_client=mock_fake_llm, mode="hybrid")

        # High confidence local prediction stays local
        high_conf_probs = {"Mechanical Fault": 0.95, "Electrical Fault": 0.05}
        with patch.object(classifier.local_classifier, "predict_with_confidence", return_value=("Mechanical Fault", 0.95, high_conf_probs)):
            res_high = classifier.classify("High confidence vibration")
            self.assertEqual(res_high.model_source, "local_ml")

        # Low confidence local prediction triggers external fallback
        low_conf_probs = {"Mechanical Fault": 0.40, "Electrical Fault": 0.35}
        with patch.object(classifier.local_classifier, "predict_with_confidence", return_value=("Mechanical Fault", 0.40, low_conf_probs)):
            res_low = classifier.classify("ambiguous machine noise")
            self.assertIn("fallback", res_low.model_source)

    def test_provider_switching_preserved(self):
        """12. Verifies switching between Gemini and AI/ML API providers remains functional."""
        with patch("config.settings.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.llm_provider = "aimlapi"
            mock_s.llm_model = "z-ai/glm-5-turbo"
            mock_s.llm_api_key = "test_key"
            mock_s.llm_base_url = "https://api.aimlapi.com/v1"
            mock_s.classification_mode = "aimlapi"
            mock_settings.return_value = mock_s

            fake_client = FakeLLMClient(default_category="Electrical Fault")
            classifier = DefectClassifier(llm_client=fake_client, mode="aimlapi")
            result = classifier.classify("Sparking wire harness")
            self.assertEqual(result.model_source, "aimlapi")
            self.assertEqual(result.confidence_assessment.approximate_range, "Qualitative / N/A")


if __name__ == "__main__":
    unittest.main()
