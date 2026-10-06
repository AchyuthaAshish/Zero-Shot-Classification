"""Unit and integration test suite for Defect Segmentation (Phase 2 Step 2.2).

Validates:
A. Clear single defect (1 segment)
B. Two independent defects (2 segments)
C. Three independent defects (3 segments)
D. Different defect categories
E. Same-category but different physical equipment
F. Causal single defect (preserved as 1 segment, NOT split)
G. Same-equipment multiple symptoms (preserved as 1 segment)
H. Compound subject (preserved as 1 segment, no partial noun fragment split)
I. Unknown / non-defect / vague inputs (0 segments)
J. Ambiguous alternative inputs ('or' preserved as 1 segment)
K. English multi-defect cases
L. Telugu-English code-switched multi-defect cases
M. Native Telugu script multi-defect cases
N. Character span correctness (original_text[start:end] == segment.text)
O. Monotonic segment ordering (start_char_i < start_char_i+1)
P. Original text preservation (no mutation/rewriting)
Q. Step 2.1 integration (reuse of MultiDefectAssessment and DefectSignal)
R. Regression against existing single-defect classification behavior
S. Protected dataset integrity (600 training, 93 evaluation)
"""

import json
import csv
from pathlib import Path
import unittest

from core.schemas import (
    ClassificationResult,
    DefectSegment,
    DefectSegmentationResult,
    MultiDefectAssessment
)
from core.defect_segmenter import (
    get_defect_segmenter,
    segment_defects,
    DefectSegmenter
)
from ml.local_classifier import classify_local
from taxonomy.repository import REQUIRED_APPROVED_CATEGORIES, get_taxonomy_repository


class TestDefectSegmentation(unittest.TestCase):
    """Comprehensive test suite for Phase 2 Step 2.2 Defect Segmentation."""

    @classmethod
    def setUpClass(cls):
        cls.segmenter = get_defect_segmenter()
        cls.taxonomy_repo = get_taxonomy_repository()
        cls.approved_categories = set(REQUIRED_APPROVED_CATEGORIES)

    # -----------------------------------------------------------------------
    # A. Clear Single Defect
    # -----------------------------------------------------------------------

    def test_single_defect_segmentation(self):
        """Verifies single defect produces exactly 1 segment covering the input."""
        text = "Motor is making a grinding noise."
        res = self.segmenter.segment_defects(text)

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.segment_count, 1)
        self.assertEqual(len(res.segments), 1)
        self.assertEqual(res.original_text, text)
        seg = res.segments[0]
        self.assertEqual(seg.segment_id, 1)
        self.assertEqual(seg.text, text)
        self.assertEqual(text[seg.start_char:seg.end_char], seg.text)

    # -----------------------------------------------------------------------
    # B. Two Independent Defects
    # -----------------------------------------------------------------------

    def test_two_independent_defects(self):
        """Verifies report with two independent defects extracts 2 clean segments."""
        text = "The conveyor motor is making a grinding noise and the temperature sensor is giving incorrect readings."
        res = self.segmenter.segment_defects(text)

        self.assertTrue(res.is_multi_defect)
        self.assertEqual(res.segment_count, 2)
        self.assertEqual(len(res.segments), 2)

        seg1, seg2 = res.segments
        self.assertEqual(seg1.text, "The conveyor motor is making a grinding noise")
        self.assertEqual(seg2.text, "the temperature sensor is giving incorrect readings")
        self.assertEqual(text[seg1.start_char:seg1.end_char], seg1.text)
        self.assertEqual(text[seg2.start_char:seg2.end_char], seg2.text)

    # -----------------------------------------------------------------------
    # C. Three Independent Defects
    # -----------------------------------------------------------------------

    def test_three_independent_defects(self):
        """Verifies report with 3 distinct defects extracts 3 clean segments in order."""
        text = "The machine is overheating, the bearing is vibrating, and the display software crashes."
        res = self.segmenter.segment_defects(text)

        self.assertTrue(res.is_multi_defect)
        self.assertEqual(res.segment_count, 3)
        self.assertEqual(len(res.segments), 3)

        seg1, seg2, seg3 = res.segments
        self.assertEqual(seg1.text, "The machine is overheating")
        self.assertEqual(seg2.text, "the bearing is vibrating")
        self.assertEqual(seg3.text, "the display software crashes")

        self.assertEqual(text[seg1.start_char:seg1.end_char], seg1.text)
        self.assertEqual(text[seg2.start_char:seg2.end_char], seg2.text)
        self.assertEqual(text[seg3.start_char:seg3.end_char], seg3.text)

    # -----------------------------------------------------------------------
    # D. Different Categories
    # -----------------------------------------------------------------------

    def test_different_categories_segmentation(self):
        """Verifies defects across Power Supply and Communication categories."""
        text = "Voltage is dropping and the PLC lost communication with the controller."
        res = self.segmenter.segment_defects(text)

        self.assertTrue(res.is_multi_defect)
        self.assertEqual(res.segment_count, 2)
        seg1, seg2 = res.segments
        self.assertEqual(seg1.text, "Voltage is dropping")
        self.assertEqual(seg2.text, "the PLC lost communication with the controller")
        self.assertEqual(text[seg1.start_char:seg1.end_char], seg1.text)
        self.assertEqual(text[seg2.start_char:seg2.end_char], seg2.text)

    # -----------------------------------------------------------------------
    # E. Same-Category but Different Physical Equipment
    # -----------------------------------------------------------------------

    def test_same_category_different_equipment(self):
        """Verifies distinct machines in the same category produce separate segments."""
        text = "The conveyor motor is vibrating and the exhaust fan has a loose bolt."
        res = self.segmenter.segment_defects(text)

        self.assertTrue(res.is_multi_defect)
        self.assertEqual(res.segment_count, 2)
        seg1, seg2 = res.segments
        self.assertEqual(seg1.text, "The conveyor motor is vibrating")
        self.assertEqual(seg2.text, "the exhaust fan has a loose bolt")
        self.assertEqual(text[seg1.start_char:seg1.end_char], seg1.text)
        self.assertEqual(text[seg2.start_char:seg2.end_char], seg2.text)

    # -----------------------------------------------------------------------
    # F. Causal Single Defect (Must NOT be split!)
    # -----------------------------------------------------------------------

    def test_causal_single_defect_with_because(self):
        """Verifies causal connective 'because' preserves the defect as ONE segment."""
        text = "The motor is making a grinding noise because the bearing is damaged."
        res = self.segmenter.segment_defects(text)

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.segment_count, 1)
        self.assertEqual(len(res.segments), 1)
        self.assertEqual(res.segments[0].text, text)
        self.assertIn("Connected single defect", res.segmentation_reason)

    def test_causal_single_defect_with_due_to(self):
        """Verifies causal connective 'due to' preserves the defect as ONE segment."""
        text = "Conveyor drive shaft has excessive vibration due to misalignment."
        res = self.segmenter.segment_defects(text)

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.segment_count, 1)
        self.assertEqual(res.segments[0].text, text)

    # -----------------------------------------------------------------------
    # G. Same-Equipment Multiple Symptoms (Must NOT be split!)
    # -----------------------------------------------------------------------

    def test_same_equipment_multiple_symptoms(self):
        """Verifies multiple descriptive symptoms for same machine stay as ONE segment."""
        text = "The motor is making a grinding noise and motor is vibrating heavily."
        res = self.segmenter.segment_defects(text)

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.segment_count, 1)
        self.assertEqual(res.segments[0].text, text)

    # -----------------------------------------------------------------------
    # H. Compound Subject (Must NOT be split into noun fragments!)
    # -----------------------------------------------------------------------

    def test_compound_subject_single_symptom(self):
        """Verifies 'motor and pump are vibrating' stays as ONE segment."""
        text = "The motor and pump are vibrating."
        res = self.segmenter.segment_defects(text)

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.segment_count, 1)
        self.assertEqual(res.segments[0].text, text)
        self.assertNotEqual(res.segments[0].text, "The motor")

    # -----------------------------------------------------------------------
    # I. Unknown / Non-Defect Inputs
    # -----------------------------------------------------------------------

    def test_unknown_greeting_produces_zero_segments(self):
        """Verifies unrelated greeting produces 0 segments."""
        text = "Hello good morning, can you help me today?"
        res = self.segmenter.segment_defects(text)

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.segment_count, 0)
        self.assertEqual(res.segments, [])

    def test_vague_input_produces_zero_segments(self):
        """Verifies vague malfunction report produces 0 segments."""
        text = "Equipment is not working properly, please inspect."
        res = self.segmenter.segment_defects(text)

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.segment_count, 0)
        self.assertEqual(res.segments, [])

    # -----------------------------------------------------------------------
    # J. Ambiguous Alternative Input ('or')
    # -----------------------------------------------------------------------

    def test_ambiguous_alternative_stays_single_segment(self):
        """Verifies disjunctive 'or' hypothesis produces 1 segment with ambiguity notice."""
        text = "Drive tripped on overcurrent or motor overheat."
        res = self.segmenter.segment_defects(text)

        self.assertFalse(res.is_multi_defect)
        self.assertEqual(res.segment_count, 1)
        self.assertEqual(res.segments[0].text, text)
        self.assertIn("ambiguity", res.segmentation_reason.lower())

    # -----------------------------------------------------------------------
    # K. English Multi-Defect
    # -----------------------------------------------------------------------

    def test_english_multi_defect(self):
        """Verifies English multi-defect report with Temperature and Software faults."""
        text = "Heat exchanger is overheating and the HMI software is frozen."
        res = self.segmenter.segment_defects(text)

        self.assertTrue(res.is_multi_defect)
        self.assertEqual(res.segment_count, 2)
        seg1, seg2 = res.segments
        self.assertEqual(seg1.text, "Heat exchanger is overheating")
        self.assertEqual(seg2.text, "the HMI software is frozen")
        self.assertEqual(text[seg1.start_char:seg1.end_char], seg1.text)
        self.assertEqual(text[seg2.start_char:seg2.end_char], seg2.text)

    # -----------------------------------------------------------------------
    # L. Telugu-English Code-Switched Multi-Defect
    # -----------------------------------------------------------------------

    def test_code_switched_multi_defect(self):
        """Verifies Telugu-English code-switched multi-defect input."""
        text = "Motor grinding noise chestundi and temperature sensor wrong reading istondi"
        res = self.segmenter.segment_defects(text)

        self.assertTrue(res.is_multi_defect)
        self.assertEqual(res.segment_count, 2)
        seg1, seg2 = res.segments
        self.assertEqual(seg1.text, "Motor grinding noise chestundi")
        self.assertEqual(seg2.text, "temperature sensor wrong reading istondi")
        self.assertEqual(text[seg1.start_char:seg1.end_char], seg1.text)
        self.assertEqual(text[seg2.start_char:seg2.end_char], seg2.text)

    # -----------------------------------------------------------------------
    # M. Native Telugu Script Example
    # -----------------------------------------------------------------------

    def test_native_telugu_script_multi_defect(self):
        """Verifies native Telugu script multi-defect with coordinating conjunction మరియు."""
        text = "మోటార్ గ్రైండింగ్ శబ్దం వస్తోంది మరియు సెన్సార్ తప్పుడు రీడింగ్ ఇస్తోంది"
        res = self.segmenter.segment_defects(text)

        self.assertTrue(res.is_multi_defect)
        self.assertEqual(res.segment_count, 2)
        seg1, seg2 = res.segments
        self.assertEqual(seg1.text, "మోటార్ గ్రైండింగ్ శబ్దం వస్తోంది")
        self.assertEqual(seg2.text, "సెన్సార్ తప్పుడు రీడింగ్ ఇస్తోంది")
        self.assertEqual(text[seg1.start_char:seg1.end_char], seg1.text)
        self.assertEqual(text[seg2.start_char:seg2.end_char], seg2.text)

    # -----------------------------------------------------------------------
    # N & P. Character Span Correctness and Original Text Preservation
    # -----------------------------------------------------------------------

    def test_character_span_correctness_and_text_preservation(self):
        """Verifies every extracted segment exactly matches original_text[start:end]."""
        test_inputs = [
            "Motor is making a grinding noise.",
            "The conveyor motor is making a grinding noise and the temperature sensor is giving incorrect readings.",
            "Voltage is dropping and the PLC lost communication with the controller.",
            "The machine is overheating, the bearing is vibrating, and the display software crashes.",
            "The conveyor motor is vibrating and the exhaust fan has a loose bolt.",
            "The motor is making a grinding noise because the bearing is damaged.",
            "Motor grinding noise chestundi and temperature sensor wrong reading istondi",
            "మోటార్ గ్రైండింగ్ శబ్దం వస్తోంది మరియు సెన్సార్ తప్పుడు రీడింగ్ ఇస్తోంది"
        ]
        for t in test_inputs:
            res = self.segmenter.segment_defects(t)
            self.assertEqual(res.original_text, t)
            for seg in res.segments:
                self.assertGreaterEqual(seg.start_char, 0)
                self.assertLessEqual(seg.end_char, len(t))
                self.assertLess(seg.start_char, seg.end_char)
                extracted_span = t[seg.start_char:seg.end_char]
                self.assertEqual(
                    extracted_span,
                    seg.text,
                    f"Span mismatch in input '{t}': expected '{seg.text}', got '{extracted_span}'"
                )

    # -----------------------------------------------------------------------
    # O. Segment Ordering (Strictly Monotonic Offsets)
    # -----------------------------------------------------------------------

    def test_segment_ordering_monotonic(self):
        """Verifies segments appear in strict left-to-right order without overlap."""
        text = "The machine is overheating, the bearing is vibrating, and the display software crashes."
        res = self.segmenter.segment_defects(text)

        self.assertEqual(len(res.segments), 3)
        for i in range(len(res.segments) - 1):
            self.assertLess(
                res.segments[i].end_char,
                res.segments[i + 1].start_char,
                f"Segment {i} and {i+1} overlap or out of order"
            )

    # -----------------------------------------------------------------------
    # Q. Step 2.1 Integration
    # -----------------------------------------------------------------------

    def test_step_2_1_detector_integration(self):
        """Verifies DefectSegmenter correctly reuses MultiDefectAssessment evidence groups."""
        text = "The conveyor motor is making a grinding noise and the temperature sensor is giving incorrect readings."
        res = self.segmenter.segment_defects(text)

        for seg in res.segments:
            self.assertIsNotNone(seg.evidence_group)
            self.assertIn(seg.evidence_group.category, self.approved_categories)
            self.assertIsNotNone(seg.linked_equipment)

    # -----------------------------------------------------------------------
    # R. Regression Against Existing Pipeline Behavior
    # -----------------------------------------------------------------------

    def test_pipeline_integration_and_single_defect_regression(self):
        """Verifies classify_local returns valid segmentation_result without breaking single-defect classification."""
        text = "Main motor drive is producing excessive vibration and loud grinding noises."
        result = classify_local(text)

        self.assertIsInstance(result, ClassificationResult)
        self.assertEqual(result.category, "Mechanical Fault")
        self.assertEqual(result.status, "success")
        self.assertIsNotNone(result.segmentation_result)
        self.assertFalse(result.segmentation_result.is_multi_defect)
        self.assertEqual(result.segmentation_result.segment_count, 1)
        self.assertEqual(len(result.segmentation_result.segments), 1)

        # Check serialization
        d = result.to_dict()
        self.assertIn("segmentation_result", d)
        self.assertFalse(d["segmentation_result"]["is_multi_defect"])
        self.assertEqual(d["segmentation_result"]["segment_count"], 1)

    def test_pipeline_integration_multi_defect(self):
        """Verifies classify_local on multi-defect input attaches full segmentation_result."""
        text = "The conveyor motor is making a grinding noise and the temperature sensor is giving incorrect readings."
        result = classify_local(text)

        self.assertIsNotNone(result.segmentation_result)
        self.assertTrue(result.segmentation_result.is_multi_defect)
        self.assertEqual(result.segmentation_result.segment_count, 2)
        self.assertEqual(len(result.segmentation_result.segments), 2)

    # -----------------------------------------------------------------------
    # S. Protected Dataset Integrity
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
