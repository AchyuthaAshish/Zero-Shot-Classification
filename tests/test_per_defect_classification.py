"""Unit and integration test suite for Per-Defect Classification (Phase 2 Step 2.3).

Validates:
1. Single defect backward compatibility
2. Two-defect classification
3. Three-defect classification
4. Same-category multiple defects
5. Different-category multiple defects
6. Causal single defect (preserved as 1 defect)
7. Unknown input (0 defects, preserved Unknown behavior)
8. Ambiguous input (1 defect with ambiguity assessment preserved)
9. Per-defect confidence (independent calibrated probabilities and ranges)
10. Per-defect explanation (isolated per-segment grounding)
11. Segment-to-result mapping (defect_id, segment_id, text)
12. Character offset preservation (original_text[start:end] == defect.text)
13. Taxonomy enforcement (all categories strictly in canonical 8-category taxonomy)
14. LOCAL mode (100% offline classification)
15. HYBRID mode (local-first with fallback routing)
16. Gemini pathway without requiring real API credentials (via FakeLLMClient)
17. No cross-segment explanation contamination
18. Existing Phase 1 regression behavior
19. Existing Step 2.1 regression behavior
20. Existing Step 2.2 regression behavior
21. Protected dataset integrity (600 training rows, 93 evaluation cases)
"""

import json
import csv
from pathlib import Path
import unittest

from core.schemas import (
    ClassificationResult,
    ConfidenceAssessment,
    AmbiguityAssessment,
    DefectSegment,
    DefectSegmentationResult,
    MultiDefectAssessment,
    PerDefectClassification,
    MultiDefectClassificationResult
)
from core.per_defect_classifier import (
    get_per_defect_classifier,
    classify_per_defect,
    PerDefectClassifier
)
from ml.local_classifier import classify_local, get_local_classifier
from classification.classifier import DefectClassifier, get_defect_classifier
from llm.client import FakeLLMClient
from taxonomy.repository import REQUIRED_APPROVED_CATEGORIES, get_taxonomy_repository


class TestPerDefectClassification(unittest.TestCase):
    """Comprehensive test suite for Phase 2 Step 2.3 Per-Defect Classification."""

    @classmethod
    def setUpClass(cls):
        cls.classifier = get_per_defect_classifier()
        cls.taxonomy_repo = get_taxonomy_repository()
        cls.approved_categories = set(REQUIRED_APPROVED_CATEGORIES)

    # -----------------------------------------------------------------------
    # 1. Single Defect Backward Compatibility
    # -----------------------------------------------------------------------

    def test_single_defect_backward_compatibility(self):
        """Verifies single-defect input produces standard ClassificationResult and 1-defect result."""
        text = "Motor is making a grinding noise."
        result = classify_local(text)

        self.assertIsInstance(result, ClassificationResult)
        self.assertEqual(result.category, "Mechanical Fault")
        self.assertEqual(result.status, "success")
        self.assertIn("grinding", result.reason.lower())
        self.assertIsNotNone(result.confidence)

        # Check multi_defect_result attachment
        self.assertIsNotNone(result.multi_defect_result)
        self.assertFalse(result.multi_defect_result.is_multi_defect)
        self.assertEqual(result.multi_defect_result.defect_count, 1)
        self.assertEqual(len(result.per_defect_classifications), 1)

        d = result.per_defect_classifications[0]
        self.assertEqual(d.defect_id, 1)
        self.assertEqual(d.category, "Mechanical Fault")
        self.assertEqual(d.status, "success")
        self.assertEqual(d.text, text)
        self.assertEqual(d.source_start_char, 0)
        self.assertEqual(d.source_end_char, len(text))

    # -----------------------------------------------------------------------
    # 2. Two-Defect Classification
    # -----------------------------------------------------------------------

    def test_two_defect_classification(self):
        """CASE 1: Conveyor motor grinding noise + temperature sensor incorrect readings."""
        text = "The conveyor motor is making a grinding noise and the temperature sensor is giving incorrect readings."
        res = classify_per_defect(text, mode="local")

        self.assertTrue(res.is_multi_defect)
        self.assertEqual(res.defect_count, 2)
        self.assertEqual(len(res.defects), 2)
        self.assertEqual(res.overall_status, "success")

        d1, d2 = res.defects
        self.assertEqual(d1.defect_id, 1)
        self.assertEqual(d1.category, "Mechanical Fault")
        self.assertEqual(d1.reliability, "High")

        self.assertEqual(d2.defect_id, 2)
        self.assertEqual(d2.category, "Sensor Fault")
        self.assertEqual(d2.reliability, "High")

    # -----------------------------------------------------------------------
    # 3. Three-Defect Classification
    # -----------------------------------------------------------------------

    def test_three_defect_classification(self):
        """CASE 3: Machine overheating + bearing vibrating + display software crashes."""
        text = "The machine is overheating, the bearing is vibrating, and the display software crashes."
        res = classify_per_defect(text, mode="local")

        self.assertTrue(res.is_multi_defect)
        self.assertEqual(res.defect_count, 3)
        self.assertEqual(len(res.defects), 3)
        self.assertEqual(res.overall_status, "success")

        d1, d2, d3 = res.defects
        self.assertEqual(d1.defect_id, 1)
        self.assertEqual(d1.category, "Temperature Fault")

        self.assertEqual(d2.defect_id, 2)
        self.assertEqual(d2.category, "Mechanical Fault")

        self.assertEqual(d3.defect_id, 3)
        self.assertEqual(d3.category, "Software Fault")

    # -----------------------------------------------------------------------
    # 4. Same-Category Multiple Defects
    # -----------------------------------------------------------------------

    def test_same_category_multiple_defects(self):
        """CASE 4: Conveyor motor vibrating + exhaust fan loose bolt (both Mechanical Fault)."""
        text = "The conveyor motor is vibrating and the exhaust fan has a loose bolt."
        res = classify_per_defect(text, mode="local")

        self.assertTrue(res.is_multi_defect)
        self.assertEqual(res.defect_count, 2)

        d1, d2 = res.defects
        self.assertEqual(d1.category, "Mechanical Fault")
        self.assertEqual(d2.category, "Mechanical Fault")
        self.assertIn("conveyor motor", d1.text.lower())
        self.assertIn("exhaust fan", d2.text.lower())

    # -----------------------------------------------------------------------
    # 5. Different-Category Multiple Defects
    # -----------------------------------------------------------------------

    def test_different_category_multiple_defects(self):
        """CASE 2: Voltage dropping + PLC lost communication with controller."""
        text = "Voltage is dropping and the PLC lost communication with the controller."
        res = classify_per_defect(text, mode="local")

        self.assertTrue(res.is_multi_defect)
        self.assertEqual(res.defect_count, 2)

        d1, d2 = res.defects
        self.assertEqual(d1.category, "Power Supply Fault")
        self.assertEqual(d2.category, "Communication Fault")

    # -----------------------------------------------------------------------
    # 6. Causal Single Defect
    # -----------------------------------------------------------------------

    def test_causal_single_defect_preserved(self):
        """Causal explanatory connective preserves single defect without splitting."""
        text = "The motor is making a grinding noise because the bearing is damaged."
        res = classify_per_defect(text, mode="local")

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.defect_count, 1)
        self.assertEqual(len(res.defects), 1)
        self.assertEqual(res.defects[0].category, "Mechanical Fault")

    # -----------------------------------------------------------------------
    # 7. Unknown Input Handling
    # -----------------------------------------------------------------------

    def test_unknown_input_handling(self):
        """Conversational/vague input produces 0 defects and unknown status."""
        text = "Hello good morning, can you help me?"
        res = classify_per_defect(text, mode="local")

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.defect_count, 0)
        self.assertEqual(len(res.defects), 0)
        self.assertEqual(res.overall_status, "unknown")

    # -----------------------------------------------------------------------
    # 8. Ambiguous Input Handling
    # -----------------------------------------------------------------------

    def test_ambiguous_alternative_hypothesis_handling(self):
        """Disjunctive alternative ('or') is preserved as single defect with ambiguity flag."""
        text = "Drive tripped on overcurrent or motor overheat."
        res = classify_per_defect(text, mode="local")

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.defect_count, 1)
        d = res.defects[0]
        self.assertIsNotNone(d.ambiguity_assessment)
        self.assertTrue(d.ambiguity_assessment.is_ambiguous)

    # -----------------------------------------------------------------------
    # 9. Per-Defect Confidence Handling
    # -----------------------------------------------------------------------

    def test_per_defect_independent_confidence(self):
        """Verifies each defect has independent authentic confidence metrics."""
        text = "The conveyor motor is making a grinding noise and the temperature sensor is giving incorrect readings."
        res = classify_per_defect(text, mode="local")

        d1, d2 = res.defects
        self.assertIsNotNone(d1.confidence)
        self.assertIsNotNone(d2.confidence)
        self.assertTrue(d1.confidence_assessment.is_calibrated)
        self.assertTrue(d2.confidence_assessment.is_calibrated)
        self.assertIsNotNone(d1.calibrated_prob)
        self.assertIsNotNone(d2.calibrated_prob)
        self.assertGreaterEqual(d1.confidence, 0.90)
        self.assertGreaterEqual(d2.confidence, 0.90)

    # -----------------------------------------------------------------------
    # 10. Per-Defect Explanation Generation
    # -----------------------------------------------------------------------

    def test_per_defect_explanation_grounding(self):
        """Verifies explanations are grounded in each defect's segment."""
        text = "The conveyor motor is making a grinding noise and the temperature sensor is giving incorrect readings."
        res = classify_per_defect(text, mode="local")

        d1, d2 = res.defects
        # Defect 1 explanation should ground in motor / grinding
        self.assertTrue(any(term in d1.explanation.lower() for term in ["motor", "grinding", "friction", "mechanical"]))
        # Defect 2 explanation should ground in sensor / measurement
        self.assertTrue(any(term in d2.explanation.lower() for term in ["sensor", "transducer", "measurement", "reading"]))

    # -----------------------------------------------------------------------
    # 11. Segment-to-Result Mapping
    # -----------------------------------------------------------------------

    def test_segment_to_result_mapping(self):
        """Verifies segment_id, defect_id, and segment text map 1-to-1."""
        text = "Voltage is dropping and the PLC lost communication with the controller."
        res = classify_per_defect(text, mode="local")

        for idx, defect in enumerate(res.defects, start=1):
            self.assertEqual(defect.defect_id, idx)
            self.assertEqual(defect.segment_id, idx)
            self.assertIn(defect.text, text)

    # -----------------------------------------------------------------------
    # 12. Character Offset Preservation
    # -----------------------------------------------------------------------

    def test_character_offset_preservation(self):
        """Verifies exact character offsets in original text match segment text."""
        text = "The machine is overheating, the bearing is vibrating, and the display software crashes."
        res = classify_per_defect(text, mode="local")

        for defect in res.defects:
            extracted_span = text[defect.source_start_char:defect.source_end_char]
            self.assertEqual(extracted_span, defect.text)

    # -----------------------------------------------------------------------
    # 13. Taxonomy Enforcement
    # -----------------------------------------------------------------------

    def test_taxonomy_enforcement_on_all_defects(self):
        """Verifies every individual defect category belongs to canonical 8-category taxonomy."""
        cases = [
            "The conveyor motor is making a grinding noise and the temperature sensor is giving incorrect readings.",
            "Voltage is dropping and the PLC lost communication with the controller.",
            "The machine is overheating, the bearing is vibrating, and the display software crashes."
        ]
        for c in cases:
            res = classify_per_defect(c, mode="local")
            for defect in res.defects:
                self.assertIn(defect.category, self.approved_categories)

    # -----------------------------------------------------------------------
    # 14. LOCAL Mode Pipeline
    # -----------------------------------------------------------------------

    def test_local_mode_pipeline(self):
        """Verifies LOCAL mode runs 100% offline without external API client invocations."""
        text = "The conveyor motor is vibrating and the exhaust fan has a loose bolt."
        res = classify_per_defect(text, mode="local")

        self.assertEqual(res.overall_status, "success")
        for d in res.defects:
            self.assertEqual(d.classification_mode, "local")
            self.assertEqual(d.provider, "local_ml")

    # -----------------------------------------------------------------------
    # 15. HYBRID Mode Pipeline
    # -----------------------------------------------------------------------

    def test_hybrid_mode_pipeline(self):
        """Verifies HYBRID mode successfully classifies multi-defect input."""
        fake_llm = FakeLLMClient()
        clf = DefectClassifier(llm_client=fake_llm, mode="hybrid")
        text = "The conveyor motor is making a grinding noise and the temperature sensor is giving incorrect readings."

        res = classify_per_defect(text, mode="hybrid", defect_classifier=clf)
        self.assertTrue(res.is_multi_defect)
        self.assertEqual(res.defect_count, 2)
        d1, d2 = res.defects
        self.assertEqual(d1.category, "Mechanical Fault")
        self.assertEqual(d2.category, "Sensor Fault")

    # -----------------------------------------------------------------------
    # 16. Gemini Pathway Without Live API Keys (FakeLLMClient)
    # -----------------------------------------------------------------------

    def test_gemini_mode_pipeline_mock(self):
        """Verifies Gemini pathway classifies multi-defect reports using mock LLM client."""
        fake_llm = FakeLLMClient(
            canned_responses={
                "grinding noise": {
                    "category": "Mechanical Fault",
                    "reason": "Grinding noise indicates physical mechanical friction.",
                    "reliability": "High"
                },
                "temperature sensor": {
                    "category": "Sensor Fault",
                    "reason": "Temperature sensor discrepancy indicates instrumentation fault.",
                    "reliability": "High"
                }
            }
        )
        clf = DefectClassifier(llm_client=fake_llm, mode="gemini")
        text = "The conveyor motor is making a grinding noise and the temperature sensor is giving incorrect readings."

        res = classify_per_defect(text, mode="gemini", defect_classifier=clf)
        self.assertTrue(res.is_multi_defect)
        self.assertEqual(res.defect_count, 2)

        d1, d2 = res.defects
        self.assertEqual(d1.category, "Mechanical Fault")
        self.assertEqual(d2.category, "Sensor Fault")
        self.assertEqual(d1.classification_mode, "gemini")
        self.assertEqual(d2.classification_mode, "gemini")

    # -----------------------------------------------------------------------
    # 17. No Cross-Segment Explanation Contamination
    # -----------------------------------------------------------------------

    def test_no_cross_segment_contamination(self):
        """Verifies explanation of segment 1 never mentions keywords of segment 2, and vice versa."""
        text = "The conveyor motor is making a grinding noise and the temperature sensor is giving incorrect readings."
        res = classify_per_defect(text, mode="local")

        d1, d2 = res.defects
        # Defect 1 must NOT mention temperature or sensor
        self.assertNotIn("temperature", d1.explanation.lower())
        self.assertNotIn("sensor", d1.explanation.lower())

        # Defect 2 must NOT mention motor or grinding or conveyor
        self.assertNotIn("conveyor", d2.explanation.lower())
        self.assertNotIn("motor", d2.explanation.lower())
        self.assertNotIn("grinding", d2.explanation.lower())

    # -----------------------------------------------------------------------
    # 18. Phase 1 Regression Behavior
    # -----------------------------------------------------------------------

    def test_phase_1_regression(self):
        """Verifies standard Phase 1 evaluation case behaves identically."""
        text = "Main motor drive is producing excessive vibration and loud grinding noises."
        res = classify_local(text)

        self.assertEqual(res.category, "Mechanical Fault")
        self.assertEqual(res.status, "success")
        self.assertEqual(res.reliability, "High")
        self.assertFalse(res.multi_defect_result.is_multi_defect)
        self.assertEqual(res.multi_defect_result.defect_count, 1)

    # -----------------------------------------------------------------------
    # 19. Step 2.1 Regression Behavior
    # -----------------------------------------------------------------------

    def test_step_2_1_regression(self):
        """Verifies multi_defect_assessment is properly attached and populated."""
        text = "The conveyor motor is making a grinding noise and the temperature sensor is giving incorrect readings."
        res = classify_local(text)

        self.assertIsNotNone(res.multi_defect_assessment)
        self.assertTrue(res.multi_defect_assessment.is_multi_defect)
        self.assertEqual(res.multi_defect_assessment.defect_signal_count, 2)

    # -----------------------------------------------------------------------
    # 20. Step 2.2 Regression Behavior
    # -----------------------------------------------------------------------

    def test_step_2_2_regression(self):
        """Verifies segmentation_result is properly attached and populated."""
        text = "The conveyor motor is making a grinding noise and the temperature sensor is giving incorrect readings."
        res = classify_local(text)

        self.assertIsNotNone(res.segmentation_result)
        self.assertTrue(res.segmentation_result.is_multi_defect)
        self.assertEqual(res.segmentation_result.segment_count, 2)
        self.assertEqual(len(res.segmentation_result.segments), 2)

    # -----------------------------------------------------------------------
    # 21. Protected Dataset Integrity
    # -----------------------------------------------------------------------

    def test_protected_dataset_integrity(self):
        """Verifies training dataset has 600 examples and evaluation dataset has 93 cases."""
        training_path = Path("data/training/defect_training.csv")
        self.assertTrue(training_path.exists())
        with open(training_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            row_count = len(reader) - 1
            self.assertEqual(row_count, 600, "Training dataset row count was altered!")

        eval_path = Path("tests/fixtures/evaluation_cases.json")
        self.assertTrue(eval_path.exists())
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
            self.assertEqual(len(eval_data), 93, "Evaluation dataset case count was altered!")


if __name__ == "__main__":
    unittest.main()
