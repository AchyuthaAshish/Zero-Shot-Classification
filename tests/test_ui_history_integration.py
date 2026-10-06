"""Unit and Integration Tests for Step 2.5D Multi-Defect UI and History Integration.

Verifies:
A. Existing single-defect rendering still works.
B. Multi-defect result renders all child defects.
C. Each child category is displayed correctly.
D. Confidence information is displayed without fake probability values.
E. Unknown result remains correctly displayed.
F. Ambiguous individual defect is shown as ambiguous.
G. Multi-defect history entry displays defect_count.
H. History expansion retrieves child items.
I. Legacy single-defect history does not crash.
J. Empty history does not crash.
K. Missing child rows do not crash.
L. Taxonomy remains restricted to the existing 8 categories.
Dataset Integrity: Protected datasets (600 training, 93 evaluation) remain untouched.
"""

import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from core.schemas import (
    ClassificationResult,
    ConfidenceAssessment,
    AmbiguityAssessment,
    PerDefectClassification,
    MultiDefectClassificationResult,
    MultiDefectValidationResult
)
from taxonomy.repository import get_taxonomy_repository, REQUIRED_APPROVED_CATEGORIES
from ml.config import TRAINING_CSV_PATH, EVALUATION_DATASET_PATH
from ui.classifier_view import _render_result_card, _render_multi_defect_result
from ui.history_view import render_history_view, _render_history_detail


def _mock_columns_factory(n):
    """Dynamic columns mock factory that returns exactly n mocks for any integer or sequence."""
    count = n if isinstance(n, int) else len(n)
    return [MagicMock() for _ in range(count)]


class TestUIHistoryIntegration(unittest.TestCase):
    """Test suite for Phase 2 Step 2.5D Multi-Defect UI and History Integration."""

    def setUp(self):
        self.tax_repo = get_taxonomy_repository()

    # -------------------------------------------------------------------------
    # Test A: Existing single-defect rendering still works
    # -------------------------------------------------------------------------
    @patch("streamlit.container")
    @patch("streamlit.columns", side_effect=_mock_columns_factory)
    @patch("streamlit.markdown")
    @patch("streamlit.write")
    def test_single_defect_rendering_preserved(self, mock_write, mock_md, mock_cols, mock_container):
        """Verifies single-defect result uses original single-defect rendering workflow without error."""
        single_result = ClassificationResult(
            category="Mechanical Fault",
            reason="Motor bearing grinding noise detected.",
            language="en",
            reliability="High",
            status="success",
            original_description="Motor is making a grinding noise.",
            normalized_description="motor is making a grinding noise",
            model_source="local_ml",
            confidence=0.92,
            confidence_assessment=ConfidenceAssessment(
                level="High",
                approximate_range="~92%",
                raw_score=0.92,
                calibrated_prob=0.915,
                top2_margin=0.45,
                is_calibrated=True,
                calibration_method="Temperature Scaling"
            )
        )

        # Should render cleanly without exception
        _render_result_card(single_result)

        # Ensure container was entered and markdown was written with category
        self.assertTrue(mock_container.called)
        md_calls = [call[0][0] for call in mock_md.call_args_list if call[0]]
        self.assertTrue(any("Mechanical Fault" in str(c) for c in md_calls))
        self.assertTrue(any("Confidence Level" in str(c) for c in md_calls))

    # -------------------------------------------------------------------------
    # Test B & C: Multi-defect result renders all child defects & categories
    # -------------------------------------------------------------------------
    @patch("streamlit.container")
    @patch("streamlit.columns", side_effect=_mock_columns_factory)
    @patch("streamlit.expander")
    @patch("streamlit.markdown")
    @patch("streamlit.write")
    @patch("streamlit.info")
    @patch("streamlit.metric")
    def test_multi_defect_result_renders_all_child_defects_and_categories(
        self, mock_metric, mock_info, mock_write, mock_md, mock_exp, mock_cols, mock_cont
    ):
        """Verifies multi-defect result renders overall summary and each child defect card."""
        d1 = PerDefectClassification(
            defect_id=1,
            segment_id=1,
            text="conveyor motor grinding noise",
            category="Mechanical Fault",
            confidence_assessment=ConfidenceAssessment(
                level="High",
                approximate_range="~88%",
                raw_score=0.88,
                calibrated_prob=0.875,
                top2_margin=0.40,
                is_calibrated=True
            ),
            reliability="High",
            explanation="Bearing wear on conveyor motor.",
            classification_mode="local",
            provider="local",
            model="Local ML"
        )
        d2 = PerDefectClassification(
            defect_id=2,
            segment_id=2,
            text="temperature sensor incorrect reading",
            category="Sensor Fault",
            confidence_assessment=ConfidenceAssessment(
                level="Medium",
                approximate_range="~74%",
                raw_score=0.74,
                calibrated_prob=0.735,
                top2_margin=0.25,
                is_calibrated=True
            ),
            reliability="Medium",
            explanation="Thermocouple calibration issue.",
            classification_mode="local",
            provider="local",
            model="Local ML"
        )

        md_res = MultiDefectClassificationResult(
            original_text="The conveyor motor is making a grinding noise and the temperature sensor gives incorrect readings.",
            is_multi_defect=True,
            defect_count=2,
            defects=[d1, d2],
            overall_status="success",
            validation_result=MultiDefectValidationResult(
                is_valid=True,
                status="VALID",
                validated_defect_count=2,
                expected_segment_count=2
            )
        )

        parent_result = ClassificationResult(
            category="Mechanical Fault",
            reason="Multi-defect report",
            language="en",
            reliability="High",
            status="success",
            original_description=md_res.original_text,
            normalized_description=md_res.original_text.lower(),
            multi_defect_classification=md_res
        )

        _render_result_card(parent_result)

        # Check expander was called for both defects (Test B)
        self.assertEqual(mock_exp.call_count, 2)
        exp_titles = [call[0][0] for call in mock_exp.call_args_list if call[0]]
        # Check categories are in the expander titles (Test C)
        self.assertTrue(any("Mechanical Fault" in t for t in exp_titles))
        self.assertTrue(any("Sensor Fault" in t for t in exp_titles))

        # Check total defects metric was rendered (Step 3)
        metric_args = [call[0] for call in mock_metric.call_args_list if call[0]]
        self.assertTrue(any(a[0] == "Total Defects" and a[1] == 2 for a in metric_args))

    # -------------------------------------------------------------------------
    # Test D: Confidence information displayed without fake probability values
    # -------------------------------------------------------------------------
    @patch("streamlit.container")
    @patch("streamlit.columns", side_effect=_mock_columns_factory)
    @patch("streamlit.expander")
    @patch("streamlit.markdown")
    @patch("streamlit.write")
    @patch("streamlit.info")
    @patch("streamlit.metric")
    def test_confidence_information_without_fake_probabilities(
        self, mock_metric, mock_info, mock_write, mock_md, mock_exp, mock_cols, mock_cont
    ):
        """Verifies external/Gemini qualitative result displays Qualitative / N/A without fabricated numbers."""
        d_gemini = PerDefectClassification(
            defect_id=1,
            segment_id=1,
            text="complex intermittent communication glitch",
            category="Communication Fault",
            confidence_assessment=ConfidenceAssessment(
                level="High",
                approximate_range="Qualitative / N/A",
                raw_score=None,
                calibrated_prob=None,
                top2_margin=None,
                is_calibrated=False
            ),
            reliability="High",
            explanation="Protocol timeout observed.",
            classification_mode="gemini",
            provider="gemini",
            model="gemini-3.8-flash",
            confidence=None,
            raw_score=None,
            calibrated_prob=None,
            top2_margin=None
        )

        md_res = MultiDefectClassificationResult(
            original_text="complex intermittent communication glitch and another issue",
            is_multi_defect=True,
            defect_count=2,
            defects=[d_gemini, d_gemini],
            overall_status="success"
        )

        parent = ClassificationResult(
            category="Communication Fault",
            reason="Multi defect",
            language="en",
            reliability="High",
            status="success",
            original_description=md_res.original_text,
            normalized_description=md_res.original_text,
            multi_defect_classification=md_res
        )

        _render_result_card(parent)

        # Ensure Qualitative / N/A was displayed
        md_texts = [str(call[0][0]) for call in mock_md.call_args_list if call[0]]
        self.assertTrue(any("Qualitative / N/A" in t for t in md_texts))
        # Ensure raw score or calibrated prob is NOT fabricated with a fake float string
        self.assertFalse(any("Calibrated Prob" in t for t in md_texts))

    # -------------------------------------------------------------------------
    # Test E: Unknown result remains correctly displayed
    # -------------------------------------------------------------------------
    @patch("streamlit.container")
    @patch("streamlit.columns", side_effect=_mock_columns_factory)
    @patch("streamlit.info")
    @patch("streamlit.markdown")
    @patch("streamlit.write")
    def test_unknown_result_displays_correctly(self, mock_write, mock_md, mock_info, mock_cols, mock_cont):
        """Verifies Unknown category triggers the dedicated insufficient technical evidence warning."""
        unknown_result = ClassificationResult(
            category="Unknown",
            reason="The description does not contain sufficient technical evidence.",
            language="en",
            reliability="Low",
            status="unknown",
            original_description="Something is wrong with the machine.",
            normalized_description="something is wrong with the machine",
            model_source="local_ml",
            confidence=None,
            confidence_assessment=ConfidenceAssessment(
                level="Uncertain",
                approximate_range="Insufficient Evidence",
                raw_score=None,
                calibrated_prob=None,
                top2_margin=None,
                is_calibrated=False,
                is_ambiguous=True
            )
        )

        _render_result_card(unknown_result)

        info_calls = [str(call[0][0]) for call in mock_info.call_args_list if call[0]]
        self.assertTrue(any("sufficient technical evidence" in c for c in info_calls))

    # -------------------------------------------------------------------------
    # Test F: Ambiguous individual defect is shown as ambiguous
    # -------------------------------------------------------------------------
    @patch("streamlit.container")
    @patch("streamlit.columns", side_effect=_mock_columns_factory)
    @patch("streamlit.expander")
    @patch("streamlit.warning")
    @patch("streamlit.info")
    @patch("streamlit.markdown")
    @patch("streamlit.write")
    @patch("streamlit.metric")
    def test_ambiguous_individual_defect_displayed(
        self, mock_metric, mock_write, mock_md, mock_info, mock_warn, mock_exp, mock_cols, mock_cont
    ):
        """Verifies child defect with ambiguity displays ambiguity warning without marking entire report Unknown."""
        amb_defect = PerDefectClassification(
            defect_id=1,
            segment_id=1,
            text="vibration or electrical hum",
            category="Mechanical Fault",
            confidence_assessment=ConfidenceAssessment(
                level="Medium",
                approximate_range="~65%",
                is_ambiguous=True
            ),
            ambiguity_assessment=AmbiguityAssessment(
                is_ambiguous=True,
                reason="Competing evidence between Mechanical Fault and Electrical Fault."
            ),
            reliability="Medium",
            explanation="Uncertain acoustic pattern.",
            classification_mode="local"
        )
        clear_defect = PerDefectClassification(
            defect_id=2,
            segment_id=2,
            text="power cable sparking",
            category="Electrical Fault",
            confidence_assessment=ConfidenceAssessment(
                level="High",
                approximate_range="~90%",
                is_ambiguous=False
            ),
            reliability="High",
            explanation="Sparking detected at cable junction.",
            classification_mode="local"
        )

        md_res = MultiDefectClassificationResult(
            original_text="vibration or electrical hum and power cable sparking",
            is_multi_defect=True,
            defect_count=2,
            defects=[amb_defect, clear_defect],
            overall_status="success"
        )

        parent = ClassificationResult(
            category="Mechanical Fault",
            reason="Multi defect",
            language="en",
            reliability="Medium",
            status="success",
            original_description=md_res.original_text,
            normalized_description=md_res.original_text,
            multi_defect_classification=md_res
        )

        _render_result_card(parent)

        # Ambiguity warning should be shown specifically for the ambiguous defect
        warn_calls = [str(call[0][0]) for call in mock_warn.call_args_list if call[0]]
        self.assertTrue(any("Ambiguity Detected" in w and "Electrical Fault" in w for w in warn_calls))

    # -------------------------------------------------------------------------
    # Test G: Multi-defect history entry displays defect_count
    # -------------------------------------------------------------------------
    @patch("ui.history_view.is_supabase_configured", return_value=True)
    @patch("ui.history_view.get_defect_reports")
    @patch("streamlit.dataframe")
    @patch("streamlit.columns", side_effect=_mock_columns_factory)
    @patch("streamlit.selectbox", return_value=None)
    def test_multi_defect_history_displays_defect_count(self, mock_select, mock_cols, mock_df_render, mock_get_reports, mock_sb):
        """Verifies history overview table marks multi-defect rows with their defect_count."""
        mock_get_reports.return_value = [
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "reporter_name": "Alice",
                "employee_id": "EMP-01",
                "defect_description": "Motor grinding and sensor error",
                "category": "Mechanical Fault",
                "is_multi_defect": True,
                "defect_count": 2,
                "validation_status": "VALID",
                "confidence_level": "High",
                "created_at": "2026-03-01T10:00:00Z"
            }
        ]

        render_history_view()

        # The dataframe rendered must contain Report Type with "Multi-Defect • 2 defects"
        self.assertTrue(mock_df_render.called)
        df_passed = mock_df_render.call_args[0][0]
        self.assertIsInstance(df_passed, pd.DataFrame)
        self.assertEqual(len(df_passed), 1)
        self.assertEqual(df_passed.iloc[0]["Report Type"], "Multi-Defect • 2 defects")

    # -------------------------------------------------------------------------
    # Test H: History expansion retrieves child items
    # -------------------------------------------------------------------------
    @patch("ui.history_view.get_defect_report_items")
    @patch("streamlit.expander")
    @patch("streamlit.columns", side_effect=_mock_columns_factory)
    @patch("streamlit.markdown")
    @patch("streamlit.info")
    @patch("streamlit.write")
    def test_history_expansion_retrieves_child_items(
        self, mock_write, mock_info, mock_md, mock_cols, mock_exp, mock_get_items
    ):
        """Verifies selecting a multi-defect history report calls get_defect_report_items and displays them."""
        report_record = {
            "id": "22222222-2222-2222-2222-222222222222",
            "reporter_name": "Bob",
            "employee_id": "EMP-02",
            "defect_description": "Exchanger overheating and HMI frozen",
            "category": "Temperature Fault",
            "is_multi_defect": True,
            "defect_count": 2,
            "validation_status": "VALID"
        }

        mock_get_items.return_value = [
            {
                "defect_index": 1,
                "defect_text": "Exchanger overheating",
                "category": "Temperature Fault",
                "confidence_level": "High",
                "confidence_range": "~91%",
                "raw_score": 0.91,
                "calibrated_score": 0.905,
                "top2_margin": 0.42,
                "explanation": "High thermal dissipation issue.",
                "classification_mode": "local"
            },
            {
                "defect_index": 2,
                "defect_text": "HMI frozen",
                "category": "Software Fault",
                "confidence_level": "High",
                "confidence_range": "~89%",
                "raw_score": 0.89,
                "calibrated_score": 0.885,
                "top2_margin": 0.38,
                "explanation": "UI runtime deadlock.",
                "classification_mode": "local"
            }
        ]

        _render_history_detail(report_record)

        # get_defect_report_items must be called with the report id
        mock_get_items.assert_called_once_with("22222222-2222-2222-2222-222222222222")
        # Expander must be opened for both child items
        self.assertEqual(mock_exp.call_count, 2)

    # -------------------------------------------------------------------------
    # Test I: Legacy single-defect history does not crash
    # -------------------------------------------------------------------------
    @patch("streamlit.columns", side_effect=_mock_columns_factory)
    @patch("streamlit.markdown")
    @patch("streamlit.write")
    @patch("streamlit.info")
    def test_legacy_single_defect_history_no_crash(self, mock_info, mock_write, mock_md, mock_cols):
        """Verifies old single-defect record without multi-defect columns renders safely without crash."""
        legacy_record = {
            "id": "33333333-3333-3333-3333-333333333333",
            "reporter_name": "Charlie",
            "employee_id": "EMP-03",
            "defect_description": "Vibration on motor shaft",
            "category": "Mechanical Fault",
            "confidence": 0.88,
            "reliability": "High",
            "explanation": "Shaft misalignment.",
            # Older record without is_multi_defect, defect_count, or validation_status
        }

        try:
            _render_history_detail(legacy_record)
        except Exception as e:
            self.fail(f"_render_history_detail raised unexpected exception on legacy record: {e}")

    # -------------------------------------------------------------------------
    # Test J: Empty history does not crash
    # -------------------------------------------------------------------------
    @patch("ui.history_view.is_supabase_configured", return_value=True)
    @patch("ui.history_view.get_defect_reports", return_value=[])
    @patch("streamlit.columns", side_effect=_mock_columns_factory)
    @patch("streamlit.selectbox", return_value="All")
    @patch("streamlit.text_input", return_value="")
    @patch("streamlit.button", return_value=False)
    @patch("streamlit.info")
    def test_empty_history_does_not_crash(self, mock_info, mock_btn, mock_ti, mock_sb_cat, mock_cols, mock_get_reports, mock_sb):
        """Verifies empty database query result is handled gracefully without crash."""
        try:
            render_history_view()
        except Exception as e:
            self.fail(f"render_history_view crashed on empty history: {e}")

        # Should show info message
        info_calls = [str(call[0][0]) for call in mock_info.call_args_list if call[0]]
        self.assertTrue(any("No defect reports found" in c for c in info_calls))

    # -------------------------------------------------------------------------
    # Test K: Missing child rows do not crash
    # -------------------------------------------------------------------------
    @patch("ui.history_view.get_defect_report_items", return_value=[])
    @patch("streamlit.columns", side_effect=_mock_columns_factory)
    @patch("streamlit.markdown")
    @patch("streamlit.info")
    def test_missing_child_rows_does_not_crash(self, mock_info, mock_md, mock_cols, mock_get_items):
        """Verifies multi-defect record with missing child rows in defect_report_items renders safely."""
        partial_record = {
            "id": "44444444-4444-4444-4444-444444444444",
            "reporter_name": "David",
            "employee_id": "EMP-04",
            "defect_description": "Compound issue",
            "category": "Power Supply Fault",
            "is_multi_defect": True,
            "defect_count": 2,
            "validation_status": "VALID"
        }

        try:
            _render_history_detail(partial_record)
        except Exception as e:
            self.fail(f"_render_history_detail crashed on missing child rows: {e}")

        info_calls = [str(call[0][0]) for call in mock_info.call_args_list if call[0]]
        self.assertTrue(any("No child defect items found" in c for c in info_calls))

    # -------------------------------------------------------------------------
    # Test L: Taxonomy remains restricted to the existing 8 categories
    # -------------------------------------------------------------------------
    def test_taxonomy_restricted_to_eight_categories(self):
        """Verifies taxonomy repository strictly contains only the approved 8 categories."""
        categories = self.tax_repo.get_categories()
        self.assertEqual(len(categories), 8, "Taxonomy must strictly contain exactly 8 categories.")
        for cat in REQUIRED_APPROVED_CATEGORIES:
            self.assertIn(cat, categories)

    # -------------------------------------------------------------------------
    # Dataset Integrity Verification (Protected 600 / 93 Datasets)
    # -------------------------------------------------------------------------
    def test_protected_datasets_integrity(self):
        """Verifies protected training (600) and evaluation (93) datasets remain strictly untouched."""
        self.assertTrue(TRAINING_CSV_PATH.exists(), "Training dataset must exist.")
        df = pd.read_csv(TRAINING_CSV_PATH)
        self.assertEqual(len(df), 600, "Training dataset must contain exactly 600 rows.")

        self.assertTrue(EVALUATION_DATASET_PATH.exists(), "Evaluation dataset must exist.")
        with open(EVALUATION_DATASET_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)
        self.assertEqual(len(cases), 93, "Evaluation dataset must contain exactly 93 cases.")


if __name__ == "__main__":
    unittest.main()
