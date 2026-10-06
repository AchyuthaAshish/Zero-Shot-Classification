"""Unit and integration tests for Phase 1 Step 1.3 Unknown Detection.

Verifies:
1. Clearly valid Mechanical Fault -> not Unknown
2. Clearly valid Electrical Fault -> not Unknown
3. Clearly valid Sensor Fault -> not Unknown
4. Clearly valid Temperature Fault -> not Unknown
5. Clearly valid Software Fault -> not Unknown
6. Clearly valid Power Supply Fault -> not Unknown
7. Clearly valid Communication Fault -> not Unknown
8. Vague input -> Unknown ("Something is wrong.", "The machine has an issue.")
9. Empty / near-empty input -> Unknown (handled gracefully)
10. Non-defect / unrelated input -> Unknown ("Hello", "Need help", "What is the capital of France?")
11. Insufficient evidence -> Unknown ("The system is not working properly.", "Unit stopped.")
12. Valid specific defect with lower confidence retains specific category per documented policy
13. Multilingual input behavior remains valid (English, Telugu script, Telugu-English)
14. Gemini qualitative handling remains intact (no fabricated probabilities)
15. HYBRID mode remains functional with external fallback
16. LOCAL mode makes zero external API calls
17. Existing Step 1.1 confidence representation preserved
18. Existing Step 1.2 calibration behavior preserved
19. Taxonomy remains strictly 8 approved categories
20. No invented or unauthorized category can bypass validation
"""

import json
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

from core.schemas import ClassificationResult, PreprocessedInput, ConfidenceAssessment
from core.unknown_detector import UnknownDetector, get_unknown_detector
from ml.local_classifier import LocalDefectClassifier, get_local_classifier
from classification.classifier import DefectClassifier
from llm.client import FakeLLMClient
from taxonomy.repository import REQUIRED_APPROVED_CATEGORIES, get_taxonomy_repository
from ml.config import APPROVED_CATEGORIES, TRAINING_CSV_PATH, EVALUATION_DATASET_PATH


class TestUnknownDetection(unittest.TestCase):
    """Test suite for Phase 1 Step 1.3 Unknown Detection Layer."""

    @classmethod
    def setUpClass(cls):
        cls.detector = get_unknown_detector()
        cls.local_clf = get_local_classifier()
        cls.aux_fixture_path = Path(__file__).resolve().parent / "fixtures" / "auxiliary_unknown_test_cases.json"
        with open(cls.aux_fixture_path, "r", encoding="utf-8") as f:
            cls.aux_cases = json.load(f)

    # -------------------------------------------------------------
    # 1 - 7: Clearly valid defects across all 7 specific categories
    # -------------------------------------------------------------
    def test_clearly_valid_mechanical_fault_not_unknown(self):
        """1. Clearly valid Mechanical Fault -> not Unknown."""
        res = self.local_clf.classify("Motor producing abnormal vibration and bearing noise.")
        self.assertEqual(res.category, "Mechanical Fault")
        self.assertNotEqual(res.category, "Unknown")
        self.assertEqual(res.status, "success")

    def test_clearly_valid_electrical_fault_not_unknown(self):
        """2. Clearly valid Electrical Fault -> not Unknown."""
        res = self.local_clf.classify("Short circuit detected in 480V control panel wiring with sparking.")
        self.assertEqual(res.category, "Electrical Fault")
        self.assertNotEqual(res.category, "Unknown")
        self.assertEqual(res.status, "success")

    def test_clearly_valid_sensor_fault_not_unknown(self):
        """3. Clearly valid Sensor Fault -> not Unknown."""
        res = self.local_clf.classify("Temperature sensor reading incorrect and showing erratic calibration drift.")
        self.assertEqual(res.category, "Sensor Fault")
        self.assertNotEqual(res.category, "Unknown")
        self.assertEqual(res.status, "success")

    def test_clearly_valid_temperature_fault_not_unknown(self):
        """4. Clearly valid Temperature Fault -> not Unknown."""
        res = self.local_clf.classify("Bearing is overheating and running dangerously hot above safe temperature limits.")
        self.assertEqual(res.category, "Temperature Fault")
        self.assertNotEqual(res.category, "Unknown")
        self.assertEqual(res.status, "success")

    def test_clearly_valid_software_fault_not_unknown(self):
        """5. Clearly valid Software Fault -> not Unknown."""
        res = self.local_clf.classify("PLC firmware crashed with fatal software execution exception.")
        self.assertEqual(res.category, "Software Fault")
        self.assertNotEqual(res.category, "Unknown")
        self.assertEqual(res.status, "success")

    def test_clearly_valid_power_supply_fault_not_unknown(self):
        """6. Clearly valid Power Supply Fault -> not Unknown."""
        res = self.local_clf.classify("Auxiliary 24V power supply dead with zero output voltage and breaker tripped.")
        self.assertEqual(res.category, "Power Supply Fault")
        self.assertNotEqual(res.category, "Unknown")
        self.assertEqual(res.status, "success")

    def test_clearly_valid_communication_fault_not_unknown(self):
        """7. Clearly valid Communication Fault -> not Unknown."""
        res = self.local_clf.classify("CAN bus communication timeout with severe packet loss across industrial network.")
        self.assertEqual(res.category, "Communication Fault")
        self.assertNotEqual(res.category, "Unknown")
        self.assertEqual(res.status, "success")

    # -------------------------------------------------------------
    # 8 - 11: Vague, empty, non-defect, and insufficient evidence inputs
    # -------------------------------------------------------------
    def test_vague_input_resolves_to_unknown(self):
        """8. Vague input -> Unknown."""
        for vague_text in ["Something is wrong.", "The machine has an issue.", "Problem occurred."]:
            res = self.local_clf.classify(vague_text)
            self.assertEqual(res.category, "Unknown", f"Expected Unknown for '{vague_text}', got {res.category}")
            self.assertEqual(res.status, "unknown")
            self.assertIn("vague", res.reason.lower())

    def test_empty_and_trivial_input_resolves_to_unknown(self):
        """9. Empty / near-empty input -> Unknown."""
        # Whitespace input produces input_error status with Unknown category
        res_empty = self.local_clf.classify("   ")
        self.assertEqual(res_empty.category, "Unknown")
        self.assertEqual(res_empty.status, "input_error")

        # Single generic token
        res_single = self.local_clf.classify("test")
        self.assertEqual(res_single.category, "Unknown")
        self.assertEqual(res_single.status, "unknown")

    def test_non_defect_unrelated_input_resolves_to_unknown(self):
        """10. Non-defect / unrelated input -> Unknown."""
        for unrelated in ["Hello", "Need help", "Good morning, how are you today?", "What is the capital of France?"]:
            res = self.local_clf.classify(unrelated)
            self.assertEqual(res.category, "Unknown", f"Expected Unknown for '{unrelated}', got {res.category}")
            self.assertEqual(res.status, "unknown")
            self.assertIn("unrelated", res.reason.lower())

    def test_insufficient_evidence_resolves_to_unknown(self):
        """11. Insufficient evidence -> Unknown."""
        for text in ["The system is not working properly.", "Unit stopped.", "Machine stopped."]:
            res = self.local_clf.classify(text)
            self.assertEqual(res.category, "Unknown", f"Expected Unknown for '{text}', got {res.category}")
            self.assertEqual(res.status, "unknown")

    # -------------------------------------------------------------
    # 12: Valid specific defect with lower confidence policy
    # -------------------------------------------------------------
    def test_valid_defect_with_lower_confidence_retains_specific_category(self):
        """12. Valid specific defect with lower confidence retains category per documented policy."""
        # When explicit technical evidence is present (e.g. bearing noise), but probability is lower
        mock_probs = {cat: 0.08 for cat in APPROVED_CATEGORIES}
        mock_probs["Mechanical Fault"] = 0.44  # Lower confidence
        mock_probs["Electrical Fault"] = 0.20

        with patch.object(self.local_clf, "predict_with_confidence", return_value=("Mechanical Fault", 0.44, mock_probs)):
            res = self.local_clf.classify("Slight bearing noise detected during startup sequence.")
            # Explicit physical symptom (bearing noise) ensures category is retained, not forced into Unknown
            self.assertEqual(res.category, "Mechanical Fault")
            self.assertEqual(res.status, "success")
            self.assertTrue(res.confidence_assessment.is_ambiguous)

    # -------------------------------------------------------------
    # 13: Multilingual input behavior
    # -------------------------------------------------------------
    def test_multilingual_input_behavior(self):
        """13. Multilingual input behavior remains valid across English, Telugu, and Telugu-English."""
        # A. Telugu valid defect -> Mechanical Fault
        res_te = self.local_clf.classify("మోటార్ ఎక్కువగా వైబ్రేట్ అవుతోంది మరియు శబ్దం వస్తోంది.")
        self.assertEqual(res_te.category, "Mechanical Fault")

        # B. Telugu vague -> Unknown
        res_te_vague = self.local_clf.classify("మెషిన్‌లో ఏదో సమస్య వచ్చింది.")
        self.assertEqual(res_te_vague.category, "Unknown")
        self.assertEqual(res_te_vague.status, "unknown")

        # C. Telugu-English code-switched valid -> Mechanical Fault
        res_cs = self.local_clf.classify("Motor lo unusual sound vastundi, heavy vibration undi.")
        self.assertEqual(res_cs.category, "Mechanical Fault")

        # D. Telugu-English vague -> Unknown
        res_cs_vague = self.local_clf.classify("Machine lo edo issue undi, work avvatledu.")
        self.assertEqual(res_cs_vague.category, "Unknown")
        self.assertEqual(res_cs_vague.status, "unknown")

    # -------------------------------------------------------------
    # 14: Gemini qualitative handling
    # -------------------------------------------------------------
    def test_gemini_qualitative_handling_remains_intact(self):
        """14. Gemini qualitative handling remains intact without fake probabilities."""
        fake_client = FakeLLMClient(
            canned_responses={
                "motor vibration": {
                    "category": "Mechanical Fault",
                    "reason": "Vibration confirms mechanical fault.",
                    "language": "English",
                    "reliability": "High"
                }
            }
        )
        classifier = DefectClassifier(llm_client=fake_client, mode="gemini")
        result = classifier.classify("Motor vibration observed during peak shift.")
        self.assertEqual(result.category, "Mechanical Fault")
        self.assertIsNone(result.confidence)
        self.assertFalse(result.confidence_assessment.is_calibrated)
        self.assertEqual(result.confidence_assessment.approximate_range, "Qualitative / N/A")

    # -------------------------------------------------------------
    # 15 & 16: Hybrid mode and Local offline verification
    # -------------------------------------------------------------
    def test_hybrid_mode_and_unnecessary_calls_avoidance(self):
        """15 & 16. HYBRID mode functional and avoids unnecessary API calls for unrelated text."""
        fake_client = FakeLLMClient()
        spy_client = MagicMock(wraps=fake_client)
        classifier = DefectClassifier(llm_client=spy_client, mode="hybrid")

        # High confidence local prediction makes ZERO external calls
        res_local = classifier.classify("Conveyor motor bearing is overheating and vibrating.")
        self.assertEqual(res_local.category, "Mechanical Fault")
        spy_client.classify.assert_not_called()

        # Non-defect greeting ("Hello") resolves locally to Unknown without calling LLM
        res_greeting = classifier.classify("Hello")
        self.assertEqual(res_greeting.category, "Unknown")
        spy_client.classify.assert_not_called()

    def test_local_mode_makes_zero_external_api_calls(self):
        """16. LOCAL mode makes no external API calls under any circumstance."""
        fake_client = FakeLLMClient()
        spy_client = MagicMock(wraps=fake_client)
        classifier = DefectClassifier(llm_client=spy_client, mode="local")

        for test_text in [
            "Motor bearing vibration.",
            "Something is wrong.",
            "Hello",
            "Short circuit in panel."
        ]:
            res = classifier.classify(test_text)
            self.assertIn(res.category, REQUIRED_APPROVED_CATEGORIES)
        spy_client.classify.assert_not_called()

    # -------------------------------------------------------------
    # 17 & 18: Step 1.1 and 1.2 behavior preservation
    # -------------------------------------------------------------
    def test_step_1_1_and_step_1_2_calibration_preserved(self):
        """17 & 18. ConfidenceAssessment structure and calibration behavior preserved."""
        res = self.local_clf.classify("Centrifugal pump impeller is rattling with abnormal vibration.")
        ca = res.confidence_assessment
        self.assertIsInstance(ca, ConfidenceAssessment)
        self.assertTrue(ca.is_calibrated)
        self.assertEqual(ca.calibration_method, "Temperature Scaling")
        self.assertIsNotNone(ca.calibrated_prob)
        self.assertIsNotNone(ca.raw_score)
        self.assertIsNotNone(ca.top2_margin)

    # -------------------------------------------------------------
    # 19 & 20: Taxonomy integrity and validation rejection
    # -------------------------------------------------------------
    def test_taxonomy_integrity_strictly_eight_categories(self):
        """19. Authoritative taxonomy remains strictly 8 categories."""
        cats = get_taxonomy_repository().get_categories()
        self.assertEqual(len(cats), 8)
        self.assertEqual(set(cats), REQUIRED_APPROVED_CATEGORIES)

    def test_unapproved_invented_category_rejected(self):
        """20. No invented or unauthorized category can bypass validation."""
        fake_client = FakeLLMClient(
            canned_responses={
                "bearing failed": {
                    "category": "Hydraulic Component Failure",
                    "reason": "Invented category."
                }
            }
        )
        classifier = DefectClassifier(llm_client=fake_client, mode="gemini")
        result = classifier.classify("Hydraulic pump bearing failed.")
        self.assertEqual(result.category, "Unknown")
        self.assertEqual(result.status, "validation_error")

    # -------------------------------------------------------------
    # Auxiliary fixture dataset execution
    # -------------------------------------------------------------
    def test_auxiliary_unknown_dataset_evaluation(self):
        """Evaluates all 20 cases from auxiliary_unknown_test_cases.json."""
        for case in self.aux_cases:
            res = self.local_clf.classify(case["description"])
            self.assertIn(res.category, REQUIRED_APPROVED_CATEGORIES)
            if case["expected_category"] == "Unknown":
                self.assertEqual(
                    res.category, "Unknown",
                    f"Auxiliary case {case['id']} ('{case['description']}') expected Unknown, got {res.category}"
                )


if __name__ == "__main__":
    unittest.main()
