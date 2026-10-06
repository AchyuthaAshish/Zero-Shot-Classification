"""Comprehensive Test Suite for Multi-Defect Validation Layer.

Conforms to Phase 2 Step 2.4:
- All 7 canonical test cases (Section 8)
- All 15 negative/fault-injection tests (Section 9)
- Taxonomy validation (canonical 8 categories)
- Span boundary & overlap checks
- Calibrated and qualitative confidence validation
- Explanation validation
- Ambiguity and Unknown consistency
- Single-defect backward compatibility
- Integration with ClassificationResult and PerDefectClassification
- 100% offline (no real external Gemini API calls)
"""

import pytest
from core.schemas import (
    ClassificationResult,
    ConfidenceAssessment,
    AmbiguityAssessment,
    DefectSegment,
    DefectSegmentationResult,
    MultiDefectAssessment,
    DefectSignal,
    PerDefectClassification,
    MultiDefectClassificationResult,
    ValidationIssue,
    MultiDefectValidationResult
)
from core.multi_defect_validator import (
    MultiDefectValidator,
    get_multi_defect_validator,
    validate_multi_defect_result
)
from ml.local_classifier import get_local_classifier


# Helper fixture to create a valid base PerDefectClassification
def create_valid_defect(
    defect_id: int = 1,
    segment_id: int = 1,
    text: str = "conveyor motor is vibrating",
    category: str = "Mechanical Fault",
    start_char: int = 0,
    end_char: int = 27,
    level: str = "High",
    calibrated_prob: float = 0.95,
    explanation: str = "Bearing wear caused high vibration in the conveyor motor.",
    provider: str = "local_ml",
    model: str = "local_ml"
) -> PerDefectClassification:
    return PerDefectClassification(
        defect_id=defect_id,
        segment_id=segment_id,
        text=text,
        category=category,
        confidence_assessment=ConfidenceAssessment(
            level=level,
            approximate_range="~95%",
            raw_score=calibrated_prob,
            calibrated_prob=calibrated_prob,
            top2_margin=0.45,
            is_calibrated=True,
            calibration_method="Temperature Scaling"
        ),
        reliability=level,
        explanation=explanation,
        classification_mode="local",
        provider=provider,
        model=model,
        source_start_char=start_char,
        source_end_char=end_char,
        status="success",
        confidence=calibrated_prob,
        raw_score=calibrated_prob,
        calibrated_prob=calibrated_prob,
        top2_margin=0.45
    )


@pytest.fixture(scope="module")
def local_classifier():
    return get_local_classifier()


class TestCanonicalExamples:
    """Tests covering the 7 canonical test cases from Section 8."""

    def test_case_1_two_distinct_defects(self, local_classifier):
        """Case 1: Conveyor motor noise + temperature sensor readings -> 2 segments, VALID."""
        text = "The conveyor motor is making a grinding noise and the temperature sensor is giving incorrect readings."
        res = local_classifier.classify(text)

        assert res.multi_defect_classification is not None
        md = res.multi_defect_classification
        assert md.is_multi_defect is True
        assert len(md.defects) == 2
        assert md.defects[0].category == "Mechanical Fault"
        assert md.defects[1].category == "Sensor Fault"

        val = res.validation_result
        assert val is not None
        assert val.is_valid is True
        assert val.status == "VALID"
        assert len(val.errors) == 0
        assert val.checks["taxonomy_validity"] is True
        assert val.checks["segment_count_consistency"] is True
        assert val.checks["source_span_validity"] is True

    def test_case_2_power_and_communication(self, local_classifier):
        """Case 2: Voltage drop + PLC lost communication -> Power Supply Fault + Communication Fault, VALID."""
        text = "Voltage is dropping and the PLC lost communication with the controller."
        res = local_classifier.classify(text)

        md = res.multi_defect_classification
        assert md is not None
        assert md.is_multi_defect is True
        assert len(md.defects) == 2
        assert md.defects[0].category == "Power Supply Fault"
        assert md.defects[1].category == "Communication Fault"

        val = res.validation_result
        assert val is not None
        assert val.is_valid is True
        assert val.status == "VALID"
        assert len(val.errors) == 0

    def test_case_3_three_defects(self, local_classifier):
        """Case 3: Overheating + bearing vibrating + software crashes -> 3 defects, VALID."""
        text = "The machine is overheating, the bearing is vibrating, and the display software crashes."
        res = local_classifier.classify(text)

        md = res.multi_defect_classification
        assert md is not None
        assert md.is_multi_defect is True
        assert len(md.defects) == 3
        assert md.defects[0].category == "Temperature Fault"
        assert md.defects[1].category == "Mechanical Fault"
        assert md.defects[2].category == "Software Fault"

        val = res.validation_result
        assert val is not None
        assert val.is_valid is True
        assert val.status == "VALID"
        assert len(val.errors) == 0

    def test_case_4_same_category_distinct_equipment(self, local_classifier):
        """Case 4: Motor vibrating + exhaust fan loose bolt -> 2 Mechanical Faults, VALID."""
        text = "The conveyor motor is vibrating and the exhaust fan has a loose bolt."
        res = local_classifier.classify(text)

        md = res.multi_defect_classification
        assert md is not None
        assert md.is_multi_defect is True
        assert len(md.defects) == 2
        assert md.defects[0].category == "Mechanical Fault"
        assert md.defects[1].category == "Mechanical Fault"

        val = res.validation_result
        assert val is not None
        assert val.is_valid is True
        assert val.status == "VALID"
        assert len(val.errors) == 0

    def test_case_5_causal_single_defect(self, local_classifier):
        """Case 5: Grinding noise because bearing damaged -> 1 segment, NOT multi-defect, VALID."""
        text = "The motor is making a grinding noise because the bearing is damaged."
        res = local_classifier.classify(text)

        md = res.multi_defect_classification
        assert md is not None
        assert md.is_multi_defect is False
        assert len(md.defects) == 1
        assert md.defects[0].category == "Mechanical Fault"

        val = res.validation_result
        assert val is not None
        assert val.is_valid is True
        assert val.status == "VALID"
        assert len(val.errors) == 0

    def test_case_6_unknown_greeting(self, local_classifier):
        """Case 6: Non-defect greeting -> 0 segments, Unknown behavior, UNKNOWN status."""
        text = "Hello good morning, can you help me?"
        res = local_classifier.classify(text)

        assert res.category == "Unknown"
        md = res.multi_defect_classification
        assert md is not None
        assert md.is_multi_defect is False
        assert len(md.defects) == 0

        val = res.validation_result
        assert val is not None
        assert val.is_valid is True
        assert val.status == "UNKNOWN"
        assert len(val.errors) == 0

    def test_case_7_ambiguous_alternative(self, local_classifier):
        """Case 7: Drive tripped on overcurrent or motor overheat -> 1 segment, ambiguity preserved, PARTIAL/VALID."""
        text = "Drive tripped on overcurrent or motor overheat."
        res = local_classifier.classify(text)

        md = res.multi_defect_classification
        assert md is not None
        assert md.is_multi_defect is False
        assert len(md.defects) == 1

        val = res.validation_result
        assert val is not None
        assert val.is_valid is True
        assert val.status in ("VALID", "PARTIAL")
        assert len(val.errors) == 0


class TestNegativeFaultInjection:
    """Tests covering all 15 negative/fault-injection scenarios from Section 9."""

    def test_fault_1_segment_count_greater_than_classification_count(self):
        """Fault 1: 2 segments + 1 classification -> SEGMENT_CLASSIFICATION_COUNT_MISMATCH."""
        text = "Motor is overheating and pump is leaking."
        seg1 = DefectSegment(segment_id=1, text="Motor is overheating", start_char=0, end_char=20)
        seg2 = DefectSegment(segment_id=2, text="pump is leaking", start_char=25, end_char=40)
        d1 = create_valid_defect(defect_id=1, segment_id=1, text="Motor is overheating", start_char=0, end_char=20)

        seg_res = DefectSegmentationResult(original_text=text, is_multi_defect=True, segment_count=2, segments=[seg1, seg2])
        md_res = MultiDefectClassificationResult(original_text=text, is_multi_defect=True, defect_count=1, defects=[d1], segmentation_result=seg_res)

        val = validate_multi_defect_result(md_res)
        assert val.is_valid is False
        assert val.status == "INVALID"
        assert "SEGMENT_CLASSIFICATION_COUNT_MISMATCH" in val.error_codes
        assert val.checks["segment_count_consistency"] is False

    def test_fault_2_segment_count_less_than_classification_count(self):
        """Fault 2: 1 segment + 2 classifications -> SEGMENT_CLASSIFICATION_COUNT_MISMATCH."""
        text = "Motor is overheating."
        seg1 = DefectSegment(segment_id=1, text="Motor is overheating", start_char=0, end_char=20)
        d1 = create_valid_defect(defect_id=1, segment_id=1, text="Motor is overheating", start_char=0, end_char=20)
        d2 = create_valid_defect(defect_id=2, segment_id=1, text="Motor is overheating", start_char=0, end_char=20)

        seg_res = DefectSegmentationResult(original_text=text, is_multi_defect=False, segment_count=1, segments=[seg1])
        md_res = MultiDefectClassificationResult(original_text=text, is_multi_defect=False, defect_count=2, defects=[d1, d2], segmentation_result=seg_res)

        val = validate_multi_defect_result(md_res)
        assert val.is_valid is False
        assert val.status == "INVALID"
        assert "SEGMENT_CLASSIFICATION_COUNT_MISMATCH" in val.error_codes

    def test_fault_3_duplicate_segment_ids(self):
        """Fault 3: Duplicate segment IDs in segmentation -> DUPLICATE_SEGMENT_ID."""
        text = "Motor is overheating and pump is vibrating."
        seg1 = DefectSegment(segment_id=1, text="Motor is overheating", start_char=0, end_char=20)
        seg2 = DefectSegment(segment_id=1, text="pump is vibrating", start_char=25, end_char=42)  # Duplicate ID 1
        d1 = create_valid_defect(defect_id=1, segment_id=1, text="Motor is overheating", start_char=0, end_char=20)
        d2 = create_valid_defect(defect_id=2, segment_id=1, text="pump is vibrating", start_char=25, end_char=42)

        seg_res = DefectSegmentationResult(original_text=text, is_multi_defect=True, segment_count=2, segments=[seg1, seg2])
        md_res = MultiDefectClassificationResult(original_text=text, is_multi_defect=True, defect_count=2, defects=[d1, d2], segmentation_result=seg_res)

        val = validate_multi_defect_result(md_res)
        assert val.is_valid is False
        assert "DUPLICATE_SEGMENT_ID" in val.error_codes
        assert val.checks["segment_id_consistency"] is False

    def test_fault_4_duplicate_defect_ids(self):
        """Fault 4: Duplicate defect IDs -> DUPLICATE_DEFECT_ID."""
        text = "Motor is overheating and pump is vibrating."
        seg1 = DefectSegment(segment_id=1, text="Motor is overheating", start_char=0, end_char=20)
        seg2 = DefectSegment(segment_id=2, text="pump is vibrating", start_char=25, end_char=42)
        d1 = create_valid_defect(defect_id=1, segment_id=1, text="Motor is overheating", start_char=0, end_char=20)
        d2 = create_valid_defect(defect_id=1, segment_id=2, text="pump is vibrating", start_char=25, end_char=42)  # Duplicate defect_id 1

        seg_res = DefectSegmentationResult(original_text=text, is_multi_defect=True, segment_count=2, segments=[seg1, seg2])
        md_res = MultiDefectClassificationResult(original_text=text, is_multi_defect=True, defect_count=2, defects=[d1, d2], segmentation_result=seg_res)

        val = validate_multi_defect_result(md_res)
        assert val.is_valid is False
        assert "DUPLICATE_DEFECT_ID" in val.error_codes
        assert val.checks["defect_id_consistency"] is False

    def test_fault_5_unknown_segment_id(self):
        """Fault 5: Defect references a non-existent segment ID -> UNKNOWN_SEGMENT_ID."""
        text = "Motor is overheating."
        seg1 = DefectSegment(segment_id=1, text="Motor is overheating", start_char=0, end_char=20)
        d1 = create_valid_defect(defect_id=1, segment_id=99, text="Motor is overheating", start_char=0, end_char=20)  # Unknown ID 99

        seg_res = DefectSegmentationResult(original_text=text, is_multi_defect=False, segment_count=1, segments=[seg1])
        md_res = MultiDefectClassificationResult(original_text=text, is_multi_defect=False, defect_count=1, defects=[d1], segmentation_result=seg_res)

        val = validate_multi_defect_result(md_res)
        assert val.is_valid is False
        assert "UNKNOWN_SEGMENT_ID" in val.error_codes
        assert val.checks["segment_id_consistency"] is False

    def test_fault_6_invalid_taxonomy_category(self):
        """Fault 6: Unapproved category -> INVALID_TAXONOMY_CATEGORY."""
        text = "Motor has hydraulic leak."
        seg1 = DefectSegment(segment_id=1, text="Motor has hydraulic leak", start_char=0, end_char=24)
        d1 = create_valid_defect(
            defect_id=1,
            segment_id=1,
            text="Motor has hydraulic leak",
            category="Hydraulic Fault",  # NOT in 8 approved categories
            start_char=0,
            end_char=24
        )

        seg_res = DefectSegmentationResult(original_text=text, is_multi_defect=False, segment_count=1, segments=[seg1])
        md_res = MultiDefectClassificationResult(original_text=text, is_multi_defect=False, defect_count=1, defects=[d1], segmentation_result=seg_res)

        val = validate_multi_defect_result(md_res)
        assert val.is_valid is False
        assert "INVALID_TAXONOMY_CATEGORY" in val.error_codes
        assert val.checks["taxonomy_validity"] is False

    def test_fault_7_overlapping_source_spans(self):
        """Fault 7: Segments/defects overlap character spans -> OVERLAPPING_SPANS."""
        text = "The conveyor motor is making a grinding noise and vibrating heavily."
        # Overlapping spans: [4:25] and [20:45]
        seg1 = DefectSegment(segment_id=1, text="conveyor motor is mak", start_char=4, end_char=25)
        seg2 = DefectSegment(segment_id=2, text="is making a grinding noise", start_char=20, end_char=46)
        d1 = create_valid_defect(defect_id=1, segment_id=1, text="conveyor motor is mak", start_char=4, end_char=25)
        d2 = create_valid_defect(defect_id=2, segment_id=2, text="is making a grinding noise", start_char=20, end_char=46)

        seg_res = DefectSegmentationResult(original_text=text, is_multi_defect=True, segment_count=2, segments=[seg1, seg2])
        md_res = MultiDefectClassificationResult(original_text=text, is_multi_defect=True, defect_count=2, defects=[d1, d2], segmentation_result=seg_res)

        val = validate_multi_defect_result(md_res)
        assert val.is_valid is False
        assert "OVERLAPPING_SPANS" in val.error_codes
        assert val.checks["span_non_overlapping"] is False

    def test_fault_8_source_span_outside_original_text(self):
        """Fault 8: Span end exceeds original text length -> SPAN_OUT_OF_BOUNDS."""
        text = "Motor vibrating."
        seg1 = DefectSegment(segment_id=1, text="Motor vibrating.", start_char=0, end_char=50)  # len(text) is 16
        d1 = create_valid_defect(defect_id=1, segment_id=1, text="Motor vibrating.", start_char=0, end_char=50)

        seg_res = DefectSegmentationResult(original_text=text, is_multi_defect=False, segment_count=1, segments=[seg1])
        md_res = MultiDefectClassificationResult(original_text=text, is_multi_defect=False, defect_count=1, defects=[d1], segmentation_result=seg_res)

        val = validate_multi_defect_result(md_res)
        assert val.is_valid is False
        assert "SPAN_OUT_OF_BOUNDS" in val.error_codes
        assert val.checks["source_span_validity"] is False

    def test_fault_9_text_does_not_match_source_span(self):
        """Fault 9: Text slice at [start:end] does not match defect.text -> SPAN_TEXT_MISMATCH."""
        text = "Motor is vibrating and pump is leaking."
        seg1 = DefectSegment(segment_id=1, text="Different text entirely", start_char=0, end_char=18)
        d1 = create_valid_defect(defect_id=1, segment_id=1, text="Different text entirely", start_char=0, end_char=18)

        seg_res = DefectSegmentationResult(original_text=text, is_multi_defect=False, segment_count=1, segments=[seg1])
        md_res = MultiDefectClassificationResult(original_text=text, is_multi_defect=False, defect_count=1, defects=[d1], segmentation_result=seg_res)

        val = validate_multi_defect_result(md_res)
        assert val.is_valid is False
        assert "SPAN_TEXT_MISMATCH" in val.error_codes
        assert val.checks["source_span_validity"] is False

    def test_fault_10_missing_confidence_assessment(self):
        """Fault 10: Missing ConfidenceAssessment -> MISSING_CONFIDENCE_ASSESSMENT."""
        text = "Motor is vibrating."
        seg1 = DefectSegment(segment_id=1, text="Motor is vibrating.", start_char=0, end_char=19)
        d1 = create_valid_defect(defect_id=1, segment_id=1, text="Motor is vibrating.", start_char=0, end_char=19)
        d1.confidence_assessment = None

        seg_res = DefectSegmentationResult(original_text=text, is_multi_defect=False, segment_count=1, segments=[seg1])
        md_res = MultiDefectClassificationResult(original_text=text, is_multi_defect=False, defect_count=1, defects=[d1], segmentation_result=seg_res)

        val = validate_multi_defect_result(md_res)
        assert val.is_valid is False
        assert "MISSING_CONFIDENCE_ASSESSMENT" in val.error_codes
        assert val.checks["confidence_structure_validity"] is False

    def test_fault_11_calibrated_probability_greater_than_one(self):
        """Fault 11: Calibrated probability > 1.0 -> INVALID_CALIBRATED_PROBABILITY."""
        text = "Motor is vibrating."
        seg1 = DefectSegment(segment_id=1, text="Motor is vibrating.", start_char=0, end_char=19)
        d1 = create_valid_defect(defect_id=1, segment_id=1, text="Motor is vibrating.", start_char=0, end_char=19)
        d1.confidence_assessment.calibrated_prob = 1.45

        seg_res = DefectSegmentationResult(original_text=text, is_multi_defect=False, segment_count=1, segments=[seg1])
        md_res = MultiDefectClassificationResult(original_text=text, is_multi_defect=False, defect_count=1, defects=[d1], segmentation_result=seg_res)

        val = validate_multi_defect_result(md_res)
        assert val.is_valid is False
        assert "INVALID_CALIBRATED_PROBABILITY" in val.error_codes
        assert val.checks["confidence_structure_validity"] is False

    def test_fault_12_calibrated_probability_less_than_zero(self):
        """Fault 12: Calibrated probability < 0.0 -> INVALID_CALIBRATED_PROBABILITY."""
        text = "Motor is vibrating."
        seg1 = DefectSegment(segment_id=1, text="Motor is vibrating.", start_char=0, end_char=19)
        d1 = create_valid_defect(defect_id=1, segment_id=1, text="Motor is vibrating.", start_char=0, end_char=19)
        d1.confidence_assessment.calibrated_prob = -0.15

        seg_res = DefectSegmentationResult(original_text=text, is_multi_defect=False, segment_count=1, segments=[seg1])
        md_res = MultiDefectClassificationResult(original_text=text, is_multi_defect=False, defect_count=1, defects=[d1], segmentation_result=seg_res)

        val = validate_multi_defect_result(md_res)
        assert val.is_valid is False
        assert "INVALID_CALIBRATED_PROBABILITY" in val.error_codes

    def test_fault_13_missing_explanation(self):
        """Fault 13: Empty or None explanation -> MISSING_EXPLANATION."""
        text = "Motor is vibrating."
        seg1 = DefectSegment(segment_id=1, text="Motor is vibrating.", start_char=0, end_char=19)
        d1 = create_valid_defect(defect_id=1, segment_id=1, text="Motor is vibrating.", start_char=0, end_char=19)
        d1.explanation = "   "  # Whitespace only

        seg_res = DefectSegmentationResult(original_text=text, is_multi_defect=False, segment_count=1, segments=[seg1])
        md_res = MultiDefectClassificationResult(original_text=text, is_multi_defect=False, defect_count=1, defects=[d1], segmentation_result=seg_res)

        val = validate_multi_defect_result(md_res)
        assert val.is_valid is False
        assert "MISSING_EXPLANATION" in val.error_codes
        assert val.checks["explanation_validity"] is False

    def test_fault_14_inconsistent_multi_defect_flags(self):
        """Fault 14: 2 segments but is_multi_defect is False -> INCONSISTENT_MULTI_DEFECT_FLAG."""
        text = "Motor is overheating and pump is vibrating."
        seg1 = DefectSegment(segment_id=1, text="Motor is overheating", start_char=0, end_char=20)
        seg2 = DefectSegment(segment_id=2, text="pump is vibrating", start_char=25, end_char=42)
        d1 = create_valid_defect(defect_id=1, segment_id=1, text="Motor is overheating", start_char=0, end_char=20)
        d2 = create_valid_defect(defect_id=2, segment_id=2, text="pump is vibrating", start_char=25, end_char=42)

        # Contradiction: 2 segments, but is_multi_defect is set to False
        seg_res = DefectSegmentationResult(original_text=text, is_multi_defect=False, segment_count=2, segments=[seg1, seg2])
        md_res = MultiDefectClassificationResult(original_text=text, is_multi_defect=False, defect_count=2, defects=[d1, d2], segmentation_result=seg_res)

        val = validate_multi_defect_result(md_res)
        assert val.is_valid is False
        assert "INCONSISTENT_MULTI_DEFECT_FLAG" in val.error_codes
        assert val.checks["multi_defect_flag_consistency"] is False

    def test_fault_15_incorrect_segment_ordering(self):
        """Fault 15: Defect sequence does not match segment sequence -> ORDER_MISMATCH."""
        text = "Motor is overheating and pump is vibrating."
        seg1 = DefectSegment(segment_id=1, text="Motor is overheating", start_char=0, end_char=20)
        seg2 = DefectSegment(segment_id=2, text="pump is vibrating", start_char=25, end_char=42)

        # Inverted ordering: first defect points to segment 2, second defect points to segment 1
        d1 = create_valid_defect(defect_id=1, segment_id=2, text="pump is vibrating", start_char=25, end_char=42)
        d2 = create_valid_defect(defect_id=2, segment_id=1, text="Motor is overheating", start_char=0, end_char=20)

        seg_res = DefectSegmentationResult(original_text=text, is_multi_defect=True, segment_count=2, segments=[seg1, seg2])
        md_res = MultiDefectClassificationResult(original_text=text, is_multi_defect=True, defect_count=2, defects=[d1, d2], segmentation_result=seg_res)

        val = validate_multi_defect_result(md_res)
        assert val.is_valid is False
        assert "ORDER_MISMATCH" in val.error_codes
        assert val.checks["order_consistency"] is False


class TestAdditionalValidationFeatures:
    """Tests for warnings, cross-segment detection, and qualitative confidence."""

    def test_low_confidence_produces_warning_not_fatal_error(self):
        """Low confidence is a warning and yields PARTIAL status, not INVALID."""
        text = "Motor is vibrating slightly."
        seg1 = DefectSegment(segment_id=1, text="Motor is vibrating slightly.", start_char=0, end_char=28)
        d1 = create_valid_defect(
            defect_id=1,
            segment_id=1,
            text="Motor is vibrating slightly.",
            start_char=0,
            end_char=28,
            level="Low",
            calibrated_prob=0.52
        )

        seg_res = DefectSegmentationResult(original_text=text, is_multi_defect=False, segment_count=1, segments=[seg1])
        md_res = MultiDefectClassificationResult(original_text=text, is_multi_defect=False, defect_count=1, defects=[d1], segmentation_result=seg_res)

        val = validate_multi_defect_result(md_res)
        assert val.is_valid is True
        assert val.status == "PARTIAL"
        assert len(val.errors) == 0
        assert "LOW_CONFIDENCE" in val.warning_codes

    def test_short_explanation_produces_warning(self):
        """Short explanation (< 3 words) produces a warning, status is PARTIAL."""
        text = "Motor is vibrating."
        seg1 = DefectSegment(segment_id=1, text="Motor is vibrating.", start_char=0, end_char=19)
        d1 = create_valid_defect(
            defect_id=1,
            segment_id=1,
            text="Motor is vibrating.",
            start_char=0,
            end_char=19,
            explanation="Bearing issue"  # 2 words
        )

        seg_res = DefectSegmentationResult(original_text=text, is_multi_defect=False, segment_count=1, segments=[seg1])
        md_res = MultiDefectClassificationResult(original_text=text, is_multi_defect=False, defect_count=1, defects=[d1], segmentation_result=seg_res)

        val = validate_multi_defect_result(md_res)
        assert val.is_valid is True
        assert val.status == "PARTIAL"
        assert "SHORT_EXPLANATION" in val.warning_codes

    def test_gemini_qualitative_confidence_accepted_without_numeric_prob(self):
        """Gemini results with qualitative confidence (None calibrated_prob) are valid."""
        text = "Chiller unit is overheating."
        seg1 = DefectSegment(segment_id=1, text="Chiller unit is overheating.", start_char=0, end_char=28)
        d1 = PerDefectClassification(
            defect_id=1,
            segment_id=1,
            text="Chiller unit is overheating.",
            category="Temperature Fault",
            confidence_assessment=ConfidenceAssessment(
                level="High",
                approximate_range="Qualitative / N/A",
                raw_score=None,
                calibrated_prob=None,
                top2_margin=None,
                is_calibrated=False
            ),
            reliability="High",
            explanation="The chiller temperature spiked beyond normal thermal operating parameters.",
            classification_mode="gemini",
            provider="gemini",
            model="gemini-3.8-flash",
            source_start_char=0,
            source_end_char=28,
            status="success"
        )

        seg_res = DefectSegmentationResult(original_text=text, is_multi_defect=False, segment_count=1, segments=[seg1])
        md_res = MultiDefectClassificationResult(original_text=text, is_multi_defect=False, defect_count=1, defects=[d1], segmentation_result=seg_res)

        val = validate_multi_defect_result(md_res)
        assert val.is_valid is True
        assert val.status == "VALID"
        assert len(val.errors) == 0

    def test_cross_segment_explanation_contamination_detected(self):
        """Cross-segment explanation mentioning distinct equipment from another segment emits warning."""
        text = "Conveyor motor is vibrating and exhaust fan has a loose bolt."
        seg1 = DefectSegment(segment_id=1, text="Conveyor motor is vibrating", start_char=0, end_char=27, linked_equipment="conveyor motor")
        seg2 = DefectSegment(segment_id=2, text="exhaust fan has a loose bolt", start_char=32, end_char=60, linked_equipment="exhaust fan")

        # Defect 1 incorrectly explains exhaust fan instead of conveyor motor
        d1 = create_valid_defect(
            defect_id=1,
            segment_id=1,
            text="Conveyor motor is vibrating",
            start_char=0,
            end_char=27,
            explanation="The exhaust fan has significant imbalance and rattling."
        )
        d2 = create_valid_defect(
            defect_id=2,
            segment_id=2,
            text="exhaust fan has a loose bolt",
            start_char=32,
            end_char=60,
            explanation="The exhaust fan mounting hardware is loose."
        )

        seg_res = DefectSegmentationResult(original_text=text, is_multi_defect=True, segment_count=2, segments=[seg1, seg2])
        md_res = MultiDefectClassificationResult(original_text=text, is_multi_defect=True, defect_count=2, defects=[d1, d2], segmentation_result=seg_res)

        val = validate_multi_defect_result(md_res)
        assert val.is_valid is True
        assert val.status == "PARTIAL"
        assert "CROSS_SEGMENT_EXPLANATION_MISMATCH" in val.warning_codes

    def test_single_defect_backward_compatibility(self):
        """Single-defect inputs wrapped in ClassificationResult validate cleanly."""
        text = "The machine bearing is damaged and vibrating."
        clf = get_local_classifier()
        res = clf.classify(text)

        assert res.validation_result is not None
        assert res.validation_result.is_valid is True
        assert res.validation_result.status == "VALID"
        assert res.validation_result.expected_segment_count == 1
        assert res.validation_result.validated_defect_count == 1

    def test_ambiguity_mistaken_for_multi_defect_error(self):
        """Ambiguous alternative disjunction mistakenly split into multiple segments triggers error."""
        text = "Drive tripped on overcurrent or motor overheat."
        seg1 = DefectSegment(segment_id=1, text="Drive tripped on overcurrent", start_char=0, end_char=28)
        seg2 = DefectSegment(segment_id=2, text="motor overheat", start_char=32, end_char=46)
        d1 = create_valid_defect(defect_id=1, segment_id=1, text="Drive tripped on overcurrent", category="Electrical Fault", start_char=0, end_char=28)
        d2 = create_valid_defect(defect_id=2, segment_id=2, text="motor overheat", category="Temperature Fault", start_char=32, end_char=46)

        seg_res = DefectSegmentationResult(original_text=text, is_multi_defect=True, segment_count=2, segments=[seg1, seg2])
        md_res = MultiDefectClassificationResult(original_text=text, is_multi_defect=True, defect_count=2, defects=[d1, d2], segmentation_result=seg_res)

        val = validate_multi_defect_result(md_res, original_text=text)
        assert val.is_valid is False
        assert val.status == "INVALID"
        assert "AMBIGUITY_MISTAKEN_FOR_MULTI_DEFECT" in val.error_codes
