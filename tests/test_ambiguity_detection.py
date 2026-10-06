"""Unit and integration tests for Phase 1 Step 1.4 Ambiguity Detection.

Verifies:
1. Clearly supported single-category defect -> not ambiguous
2. Vague input -> remains Unknown, not incorrectly labeled ambiguous
3. Genuine competing evidence -> ambiguity detected
4. Strong category-specific evidence -> not ambiguous
5. Borderline top-2 local prediction (margin < 0.12) -> ambiguity detected
6. Calibrated probability behavior correctly integrated
7. Uncalibrated score behavior correctly labeled
8. Gemini qualitative-confidence behavior preserved without fake numbers
9. Hybrid behavior triggers fallback when local prediction is ambiguous
10. Local offline behavior makes zero external API calls
11. English ambiguity detected on competing defect text
12. Telugu ambiguity detected on competing Telugu signals
13. Telugu-English code-switched ambiguity detected on competing signals
14. Taxonomy remains strictly 8 approved categories
15. Existing Unknown Detection tests continue passing
16. Existing calibration tests continue passing
"""

import json
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

from core.schemas import ClassificationResult, PreprocessedInput, ConfidenceAssessment, AmbiguityAssessment
from core.ambiguity_detector import AmbiguityDetector, get_ambiguity_detector, AMBIGUITY_MARGIN_THRESHOLD
from ml.local_classifier import LocalDefectClassifier, get_local_classifier
from classification.classifier import DefectClassifier
from llm.client import FakeLLMClient
from taxonomy.repository import REQUIRED_APPROVED_CATEGORIES, get_taxonomy_repository
from ml.config import APPROVED_CATEGORIES


class TestAmbiguityDetection(unittest.TestCase):
    """Test suite for Phase 1 Step 1.4 Ambiguity Detection Layer."""

    @classmethod
    def setUpClass(cls):
        cls.detector = get_ambiguity_detector()
        cls.local_clf = get_local_classifier()
        cls.aux_fixture_path = Path(__file__).resolve().parent / "fixtures" / "auxiliary_ambiguity_test_cases.json"
        with open(cls.aux_fixture_path, "r", encoding="utf-8") as f:
            cls.aux_cases = json.load(f)

    # -------------------------------------------------------------
    # 1. Clearly supported single-category defect -> not ambiguous
    # -------------------------------------------------------------
    def test_clearly_supported_single_category_not_ambiguous(self):
        """1. Clear single-category defect is NOT flagged as ambiguous."""
        res = self.local_clf.classify("The conveyor motor is making a grinding noise.")
        self.assertEqual(res.category, "Mechanical Fault")
        self.assertIsNotNone(res.ambiguity_assessment)
        self.assertFalse(res.ambiguity_assessment.is_ambiguous)
        self.assertEqual(res.ambiguity_assessment.method, "single_category_evidence_uncontested")

    # -------------------------------------------------------------
    # 2. Vague input -> Unknown, NOT ambiguous
    # -------------------------------------------------------------
    def test_vague_input_is_unknown_and_not_ambiguous(self):
        """2. Vague input has insufficient evidence (Unknown) and is NOT ambiguous."""
        res = self.local_clf.classify("Something is wrong with the machine.")
        self.assertEqual(res.category, "Unknown")
        self.assertIsNotNone(res.ambiguity_assessment)
        self.assertFalse(res.ambiguity_assessment.is_ambiguous)
        self.assertIn("insufficient evidence", res.ambiguity_assessment.reason.lower())
        self.assertEqual(res.ambiguity_assessment.method, "insufficient_evidence_not_ambiguous")

    # -------------------------------------------------------------
    # 3. Genuine competing evidence -> ambiguity detected
    # -------------------------------------------------------------
    def test_genuine_competing_evidence_detected_as_ambiguous(self):
        """3. Competing defect evidence (e.g. thermal + network) triggers ambiguity."""
        res = self.local_clf.classify("The motor is hot and the network connection is failing.")
        self.assertIsNotNone(res.ambiguity_assessment)
        self.assertTrue(res.ambiguity_assessment.is_ambiguous)
        self.assertIn("competing evidence", res.ambiguity_assessment.reason.lower())
        self.assertEqual(res.ambiguity_assessment.method, "competing_textual_evidence")

    # -------------------------------------------------------------
    # 4. Strong category-specific evidence -> not ambiguous
    # -------------------------------------------------------------
    def test_strong_category_evidence_not_ambiguous(self):
        """4. Strong explicit domain signals uniquely prevent ambiguity."""
        res = self.local_clf.classify("Short circuit detected in 480V control panel wiring with sparking.")
        self.assertEqual(res.category, "Electrical Fault")
        self.assertFalse(res.ambiguity_assessment.is_ambiguous)
        self.assertEqual(res.ambiguity_assessment.method, "single_category_evidence_uncontested")

    # -------------------------------------------------------------
    # 5. Borderline top-2 local prediction -> ambiguity detected
    # -------------------------------------------------------------
    def test_borderline_top2_prediction_detected_as_ambiguous(self):
        """5. Narrow margin (< 0.12) between top hypotheses triggers ambiguity."""
        # Simulated close model outputs without dominant keyword match
        mock_probs = {cat: 0.05 for cat in APPROVED_CATEGORIES}
        mock_probs["Mechanical Fault"] = 0.36
        mock_probs["Electrical Fault"] = 0.31  # margin = 0.05 (< 0.12 threshold)

        with patch.object(self.local_clf, "predict_with_confidence", return_value=("Mechanical Fault", 0.36, mock_probs)):
            res = self.local_clf.classify("Apparatus operational anomaly reported by shift operator.")
            self.assertIsNotNone(res.ambiguity_assessment)
            self.assertTrue(res.ambiguity_assessment.is_ambiguous)
            self.assertEqual(res.ambiguity_assessment.method, "borderline_top2_margin")
            self.assertLess(res.ambiguity_assessment.margin, AMBIGUITY_MARGIN_THRESHOLD)

    # -------------------------------------------------------------
    # 6 & 7. Calibrated vs Uncalibrated behavior
    # -------------------------------------------------------------
    def test_calibrated_and_uncalibrated_score_integration(self):
        """6 & 7. Calibrated probabilities and uncalibrated scores correctly handled."""
        # A. Calibrated LocalDefectClassifier
        cal_clf = LocalDefectClassifier(model_type="multilingual_embedding", use_calibration=True)
        self.assertTrue(cal_clf.is_calibrated)
        res_cal = cal_clf.classify("Conveyor roller bearing is grinding and vibrating.")
        self.assertTrue(res_cal.confidence_assessment.is_calibrated)
        self.assertIsNotNone(res_cal.confidence_assessment.calibrated_prob)
        self.assertFalse(res_cal.ambiguity_assessment.is_ambiguous)

        # B. Uncalibrated LocalDefectClassifier
        uncal_clf = LocalDefectClassifier(model_type="multilingual_embedding", use_calibration=False)
        self.assertFalse(uncal_clf.is_calibrated)
        res_uncal = uncal_clf.classify("Conveyor roller bearing is grinding and vibrating.")
        self.assertFalse(res_uncal.confidence_assessment.is_calibrated)
        self.assertEqual(res_uncal.confidence_assessment.approximate_range, "Uncalibrated model score")

    # -------------------------------------------------------------
    # 8. Gemini qualitative-confidence behavior
    # -------------------------------------------------------------
    def test_gemini_qualitative_behavior_preserved(self):
        """8. Gemini qualitative reliability preserved and ambiguity assessed from text/dispute."""
        fake_client = FakeLLMClient(
            canned_responses={
                "network connection is failing": {
                    "category": "Temperature Fault",
                    "reason": "Motor overheating and network disconnected.",
                    "language": "English",
                    "reliability": "Medium"
                }
            }
        )
        classifier = DefectClassifier(llm_client=fake_client, mode="gemini")
        res = classifier.classify("The motor is hot and the network connection is failing.")

        self.assertIsNone(res.confidence)
        self.assertFalse(res.confidence_assessment.is_calibrated)
        self.assertIsNotNone(res.ambiguity_assessment)
        self.assertTrue(res.ambiguity_assessment.is_ambiguous)
        self.assertEqual(res.ambiguity_assessment.method, "competing_textual_evidence")

    # -------------------------------------------------------------
    # 9. Hybrid behavior triggers fallback on ambiguous prediction
    # -------------------------------------------------------------
    def test_hybrid_behavior_triggers_fallback_on_ambiguity(self):
        """9. Hybrid mode falls back to external LLM when local ML is ambiguous."""
        fake_client = FakeLLMClient(
            canned_responses={
                "bearing vibrating while terminal sparking": {
                    "category": "Electrical Fault",
                    "reason": "Resolved by LLM reasoning.",
                    "language": "English",
                    "reliability": "High"
                }
            }
        )
        spy_client = MagicMock(wraps=fake_client)
        classifier = DefectClassifier(llm_client=spy_client, mode="hybrid")

        # Competing signals prompt fallback
        res = classifier.classify("Motor bearing is vibrating violently while terminal block is sparking.")
        spy_client.classify.assert_called_once()
        self.assertIn("fallback", res.model_source)

    # -------------------------------------------------------------
    # 10. Local offline behavior makes zero external API calls
    # -------------------------------------------------------------
    def test_local_offline_behavior_makes_zero_external_calls(self):
        """10. Local mode computes ambiguity 100% offline with zero external API calls."""
        fake_client = FakeLLMClient()
        spy_client = MagicMock(wraps=fake_client)
        classifier = DefectClassifier(llm_client=spy_client, mode="local")

        for text in [
            "Motor bearing is vibrating.",
            "The motor is hot and the network connection is failing.",
            "Something is wrong with the machine."
        ]:
            res = classifier.classify(text)
            self.assertIn(res.category, REQUIRED_APPROVED_CATEGORIES)
            self.assertIsNotNone(res.ambiguity_assessment)

        spy_client.classify.assert_not_called()

    # -------------------------------------------------------------
    # 11 - 13. Multilingual ambiguity verification
    # -------------------------------------------------------------
    def test_multilingual_ambiguity_verification(self):
        """11, 12, 13. Ambiguity verified across English, Telugu script, and Telugu-English."""
        # 11. English competing signals
        res_en = self.local_clf.classify("Bearing vibrating and circuit sparking.")
        self.assertTrue(res_en.ambiguity_assessment.is_ambiguous)

        # 12. Telugu script competing signals (Temperature vs Communication)
        res_te = self.local_clf.classify("మోటార్ వేడెక్కింది మరియు కనెక్షన్ కట్ అయింది.")
        self.assertTrue(res_te.ambiguity_assessment.is_ambiguous)

        # 12. Telugu script uncontested signals
        res_te_clean = self.local_clf.classify("మోటార్ ఎక్కువగా వైబ్రేట్ అవుతోంది మరియు శబ్దం వస్తోంది.")
        self.assertFalse(res_te_clean.ambiguity_assessment.is_ambiguous)

        # 13. Telugu-English code-switched competing signals
        res_cs = self.local_clf.classify("Motor chala heat avtundi and wifi disconnected aindi.")
        self.assertTrue(res_cs.ambiguity_assessment.is_ambiguous)

        # 13. Telugu-English uncontested signals
        res_cs_clean = self.local_clf.classify("Motor lo unusual sound vastundi, heavy vibration undi.")
        self.assertFalse(res_cs_clean.ambiguity_assessment.is_ambiguous)

    # -------------------------------------------------------------
    # 14. Taxonomy integrity
    # -------------------------------------------------------------
    def test_taxonomy_integrity_strictly_eight_categories(self):
        """14. Taxonomy remains strictly 8 approved categories."""
        cats = get_taxonomy_repository().get_categories()
        self.assertEqual(len(cats), 8)
        self.assertEqual(set(cats), REQUIRED_APPROVED_CATEGORIES)

    # -------------------------------------------------------------
    # Auxiliary fixture dataset execution
    # -------------------------------------------------------------
    def test_auxiliary_ambiguity_dataset_evaluation(self):
        """Evaluates all cases from auxiliary_ambiguity_test_cases.json."""
        for case in self.aux_cases:
            res = self.local_clf.classify(case["description"])
            self.assertIn(res.category, REQUIRED_APPROVED_CATEGORIES)
            self.assertIsNotNone(res.ambiguity_assessment)
            self.assertEqual(
                res.ambiguity_assessment.is_ambiguous,
                case["expected_ambiguous"],
                f"Case {case['id']} ('{case['description']}') expected ambiguous={case['expected_ambiguous']}, got {res.ambiguity_assessment.is_ambiguous}"
            )


if __name__ == "__main__":
    unittest.main()
