"""Unit and integration test suite for Multi-Defect Detection (Phase 2 Step 2.1).

Validates:
1. Clear single-defect inputs.
2. Two independent co-occurring defects across different categories.
3. Three independent co-occurring defects across different categories.
4. Connected single defect with multiple technical terms linked by causal connectives.
5. Compound equipment / multiple symptoms describing a single machine fault.
6. Distinct physical equipment items with independent symptoms within the same category.
7. Ambiguous but single-defect inputs (disjunctive 'or' / alternative hypotheses).
8. Unknown / non-defect / vague inputs (0 defect signals, preserves Unknown behavior).
9. English multi-defect cases.
10. Telugu-English code-switched multi-defect cases.
11. Native Telugu script multi-defect cases.
12. Integration into ClassificationResult and end-to-end local classification pipeline.
13. Strict 8-category taxonomy preservation.
14. Dataset integrity of protected training (600) and evaluation (93) datasets.
"""

import json
import csv
from pathlib import Path
import unittest

from core.schemas import (
    ClassificationResult,
    MultiDefectAssessment,
    DefectSignal
)
from core.multi_defect_detector import (
    get_multi_defect_detector,
    detect_multi_defect,
    MultiDefectDetector
)
from ml.local_classifier import classify_local
from taxonomy.repository import REQUIRED_APPROVED_CATEGORIES, get_taxonomy_repository


class TestMultiDefectDetection(unittest.TestCase):
    """Comprehensive test suite for Phase 2 Step 2.1 Multi-Defect Detection."""

    @classmethod
    def setUpClass(cls):
        cls.detector = get_multi_defect_detector()
        cls.taxonomy_repo = get_taxonomy_repository()
        cls.approved_categories = set(REQUIRED_APPROVED_CATEGORIES)

    # -----------------------------------------------------------------------
    # 1. Clear Single Defect Tests
    # -----------------------------------------------------------------------

    def test_clear_single_defect_mechanical(self):
        """Verifies clear single mechanical defect report."""
        text = "Motor is making a grinding noise."
        res = self.detector.detect_multi_defect(text)

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.defect_signal_count, 1)
        self.assertEqual(len(res.signals), 1)
        self.assertEqual(res.signals[0].category, "Mechanical Fault")
        self.assertEqual(res.primary_candidate_category, "Mechanical Fault")
        self.assertIn("Single defect signal identified", res.detection_reason)

    def test_clear_single_defect_electrical(self):
        """Verifies clear single electrical defect report."""
        text = "Fuse is blown in the main junction box with severe sparking."
        res = self.detector.detect_multi_defect(text)

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.defect_signal_count, 1)
        self.assertEqual(res.signals[0].category, "Electrical Fault")

    def test_clear_single_defect_sensor(self):
        """Verifies clear single sensor defect report."""
        text = "The pressure sensor is showing erratic readings."
        res = self.detector.detect_multi_defect(text)

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.defect_signal_count, 1)
        self.assertEqual(res.signals[0].category, "Sensor Fault")

    # -----------------------------------------------------------------------
    # 2. Two Independent Defects Across Different Categories
    # -----------------------------------------------------------------------

    def test_two_independent_defects_mechanical_and_sensor(self):
        """Verifies two co-occurring defects across Mechanical Fault and Sensor Fault."""
        text = "The conveyor motor is making a grinding noise and the temperature sensor is giving incorrect readings."
        res = self.detector.detect_multi_defect(text)

        self.assertTrue(res.is_multi_defect)
        self.assertEqual(res.defect_signal_count, 2)
        self.assertEqual(len(res.signals), 2)
        cats = [s.category for s in res.signals]
        self.assertIn("Mechanical Fault", cats)
        self.assertIn("Sensor Fault", cats)
        self.assertIn("Detected 2 distinct defect signals", res.detection_reason)

    def test_two_independent_defects_power_and_communication(self):
        """Verifies two co-occurring defects across Power Supply and Communication."""
        text = "Voltage is dropping and the PLC lost communication with the controller."
        res = self.detector.detect_multi_defect(text)

        self.assertTrue(res.is_multi_defect)
        self.assertEqual(res.defect_signal_count, 2)
        cats = [s.category for s in res.signals]
        self.assertTrue("Power Supply Fault" in cats or "Electrical Fault" in cats)
        self.assertIn("Communication Fault", cats)

    def test_two_independent_defects_temperature_and_software(self):
        """Verifies two co-occurring defects across Temperature and Software."""
        text = "Heat exchanger is overheating and the HMI software is frozen."
        res = self.detector.detect_multi_defect(text)

        self.assertTrue(res.is_multi_defect)
        self.assertEqual(res.defect_signal_count, 2)
        cats = [s.category for s in res.signals]
        self.assertIn("Temperature Fault", cats)
        self.assertIn("Software Fault", cats)

    # -----------------------------------------------------------------------
    # 3. Three Independent Defects
    # -----------------------------------------------------------------------

    def test_three_independent_defects(self):
        """Verifies report containing 3 distinct defects across 3 categories."""
        text = "The machine is overheating, the bearing is vibrating, and the display software crashes."
        res = self.detector.detect_multi_defect(text)

        self.assertTrue(res.is_multi_defect)
        self.assertEqual(res.defect_signal_count, 3)
        self.assertEqual(len(res.signals), 3)
        cats = [s.category for s in res.signals]
        self.assertIn("Temperature Fault", cats)
        self.assertIn("Mechanical Fault", cats)
        self.assertIn("Software Fault", cats)
        self.assertIn("Detected 3 distinct defect signals", res.detection_reason)

    # -----------------------------------------------------------------------
    # 4. Connected Single Defect with Causal Relations
    # -----------------------------------------------------------------------

    def test_connected_single_defect_with_because(self):
        """Verifies causal connective 'because' indicates single connected defect chain."""
        text = "The motor is making a grinding noise because the bearing is damaged."
        res = self.detector.detect_multi_defect(text)

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.defect_signal_count, 1)
        self.assertTrue(res.has_causal_relation)
        self.assertEqual(res.signals[0].category, "Mechanical Fault")
        self.assertIn("Connected single defect", res.detection_reason)

    def test_connected_single_defect_with_due_to(self):
        """Verifies causal connective 'due to' indicates single connected defect."""
        text = "Conveyor drive shaft has excessive vibration due to misalignment."
        res = self.detector.detect_multi_defect(text)

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.defect_signal_count, 1)
        self.assertTrue(res.has_causal_relation)
        self.assertEqual(res.signals[0].category, "Mechanical Fault")

    # -----------------------------------------------------------------------
    # 5. Compound Equipment / Multiple Symptoms for Same Machine
    # -----------------------------------------------------------------------

    def test_compound_equipment_with_single_symptom(self):
        """Verifies 'motor and pump are vibrating' is a single defect with compound subject."""
        text = "The motor and pump are vibrating."
        res = self.detector.detect_multi_defect(text)

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.defect_signal_count, 1)
        self.assertEqual(res.signals[0].category, "Mechanical Fault")

    def test_multiple_symptoms_on_same_equipment(self):
        """Verifies multiple descriptive symptoms on the same machine are single defect."""
        text = "The motor is making a grinding noise and motor is vibrating heavily."
        res = self.detector.detect_multi_defect(text)

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.defect_signal_count, 1)
        self.assertIn("Single defect with multiple descriptive symptoms", res.detection_reason)

    # -----------------------------------------------------------------------
    # 6. Distinct Equipment in Same Category (Independent Defects)
    # -----------------------------------------------------------------------

    def test_distinct_equipment_same_category_multi_defect(self):
        """Verifies two separate physical machines with independent symptoms form multi-defect."""
        text = "The conveyor motor is vibrating and the exhaust fan has a loose bolt."
        res = self.detector.detect_multi_defect(text)

        self.assertTrue(res.is_multi_defect)
        self.assertEqual(res.defect_signal_count, 2)
        self.assertIn("conveyor motor", res.detection_reason)
        self.assertIn("exhaust fan", res.detection_reason)

    # -----------------------------------------------------------------------
    # 7. Ambiguity vs Multi-Defect Distinction
    # -----------------------------------------------------------------------

    def test_ambiguous_alternative_hypotheses_with_or(self):
        """Verifies 'or' connective denotes ambiguity / alternative hypotheses, not co-occurring defects."""
        text = "Drive tripped on overcurrent or motor overheat."
        res = self.detector.detect_multi_defect(text)

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.defect_signal_count, 1)
        self.assertIn("alternative hypothetical causes (ambiguity)", res.detection_reason)

    def test_ambiguous_sensor_or_power_issue(self):
        """Verifies disjunctive hypothesis about single incident is not marked as multi-defect."""
        text = "Intermittent power loss or sensor failure causing system trip."
        res = self.detector.detect_multi_defect(text)

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.defect_signal_count, 1)
        self.assertIn("ambiguity", res.detection_reason.lower())

    # -----------------------------------------------------------------------
    # 8. Unknown / Non-Defect Inputs
    # -----------------------------------------------------------------------

    def test_unrelated_greetings_not_multi_defect(self):
        """Verifies non-defect greetings do not trigger multi-defect detection."""
        text = "Hello good morning, can you help me today?"
        res = self.detector.detect_multi_defect(text)

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.defect_signal_count, 0)
        self.assertEqual(len(res.signals), 0)
        self.assertIn("Unrelated or non-defect input", res.detection_reason)

    def test_vague_input_not_multi_defect(self):
        """Verifies vague malfunction reports without physical symptoms have 0 defect signals."""
        text = "Equipment is not working properly, please inspect."
        res = self.detector.detect_multi_defect(text)

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.defect_signal_count, 0)
        self.assertIn("Vague description lacking technical evidence", res.detection_reason)

    def test_vague_stopped_line_not_multi_defect(self):
        """Verifies 'Line 2 stopped, check reason' is not marked as multi-defect."""
        text = "Line 2 stopped, check reason."
        res = self.detector.detect_multi_defect(text)

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.defect_signal_count, 0)

    # -----------------------------------------------------------------------
    # 9. Multilingual Support: Code-Switched & Telugu Script
    # -----------------------------------------------------------------------

    def test_code_switched_multi_defect(self):
        """Verifies Telugu-English code-switched multi-defect input."""
        text = "Motor grinding noise chestundi and temperature sensor wrong reading istondi"
        res = self.detector.detect_multi_defect(text)

        self.assertTrue(res.is_multi_defect)
        self.assertEqual(res.defect_signal_count, 2)
        cats = [s.category for s in res.signals]
        self.assertIn("Mechanical Fault", cats)
        self.assertIn("Sensor Fault", cats)

    def test_telugu_script_multi_defect(self):
        """Verifies native Telugu script multi-defect input with coordinating conjunction మరియు."""
        text = "మోటార్ గ్రైండింగ్ శబ్దం వస్తోంది మరియు సెన్సార్ తప్పుడు రీడింగ్ ఇస్తోంది"
        res = self.detector.detect_multi_defect(text)

        self.assertTrue(res.is_multi_defect)
        self.assertEqual(res.defect_signal_count, 2)
        cats = [s.category for s in res.signals]
        self.assertIn("Mechanical Fault", cats)
        self.assertIn("Sensor Fault", cats)

    # -----------------------------------------------------------------------
    # 10. Pipeline Integration & Backward Compatibility
    # -----------------------------------------------------------------------

    def test_classification_result_integration(self):
        """Verifies classify_local returns a ClassificationResult with valid multi_defect_assessment."""
        text = "The conveyor motor is making a grinding noise and the temperature sensor is giving incorrect readings."
        result = classify_local(text)

        self.assertIsInstance(result, ClassificationResult)
        self.assertIsNotNone(result.multi_defect_assessment)
        self.assertTrue(result.multi_defect_assessment.is_multi_defect)
        self.assertEqual(result.multi_defect_assessment.defect_signal_count, 2)
        # Check alias property
        self.assertEqual(len(result.multi_defect_assessment.evidence_groups), 2)
        # Verify serialization
        d = result.to_dict()
        self.assertIn("multi_defect_assessment", d)
        self.assertTrue(d["multi_defect_assessment"]["is_multi_defect"])

    def test_single_defect_regression_preservation(self):
        """Verifies existing single-defect evaluation benchmark case retains 100% backward compatibility."""
        text = "Main motor drive is producing excessive vibration and loud grinding noises."
        result = classify_local(text)

        self.assertEqual(result.category, "Mechanical Fault")
        self.assertEqual(result.status, "success")
        self.assertFalse(result.multi_defect_assessment.is_multi_defect)
        self.assertEqual(result.multi_defect_assessment.defect_signal_count, 1)

    # -----------------------------------------------------------------------
    # 11. Strict Taxonomy Preservation
    # -----------------------------------------------------------------------

    def test_taxonomy_preservation_in_all_signals(self):
        """Verifies that all categories assigned in multi-defect signals strictly belong to 8 canonical categories."""
        test_inputs = [
            "Motor is making a grinding noise.",
            "The conveyor motor is making a grinding noise and the temperature sensor is giving incorrect readings.",
            "Voltage is dropping and the PLC lost communication with the controller.",
            "The machine is overheating, the bearing is vibrating, and the display software crashes.",
            "Wiring short circuit and power supply blackout.",
            "Motor grinding noise chestundi and temperature sensor wrong reading istondi",
            "మోటార్ గ్రైండింగ్ శబ్దం వస్తోంది మరియు సెన్సార్ తప్పుడు రీడింగ్ ఇస్తోంది"
        ]
        for t in test_inputs:
            assessment = self.detector.detect_multi_defect(t)
            for sig in assessment.signals:
                self.assertIn(
                    sig.category,
                    self.approved_categories,
                    f"Category '{sig.category}' from input '{t}' violates the 8-category taxonomy!"
                )

    # -----------------------------------------------------------------------
    # 12. Protected Dataset Integrity
    # -----------------------------------------------------------------------

    def test_protected_datasets_integrity(self):
        """Verifies that protected training (600) and evaluation (93) datasets remain strictly untouched."""
        root = Path(__file__).resolve().parent.parent

        # 1. Training set
        train_path = root / "data" / "training" / "defect_training.csv"
        self.assertTrue(train_path.exists())
        with open(train_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            train_rows = list(reader)
        self.assertEqual(len(train_rows), 600, "Protected training set MUST contain exactly 600 examples.")

        # 2. Held-out benchmark
        eval_path = root / "tests" / "fixtures" / "evaluation_cases.json"
        self.assertTrue(eval_path.exists())
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_cases = json.load(f)
        self.assertEqual(len(eval_cases), 93, "Protected evaluation benchmark MUST contain exactly 93 cases.")


if __name__ == "__main__":
    unittest.main()
