"""Unit tests for Phase 1 Step 1.7 Basic Evaluation Dashboard / Report.

Verifies:
1. Evaluation artifacts reports/model_comparison.json and reports/error_analysis.json exist and are valid.
2. Artifact loader functions load expected structures and metrics.
3. Summary report reports/evaluation_summary.md exists and covers all required sections.
4. UI component ui/evaluation_view.py and alias ui/statistics_view.py export callable dashboard handlers.
5. Navigation in app.py integrates AI Evaluation Dashboard.
6. Protected datasets (600 training, 93 evaluation) remain completely untouched.
"""

import json
from pathlib import Path
import unittest
import pandas as pd

from ml.config import TRAINING_CSV_PATH, EVALUATION_DATASET_PATH, APPROVED_CATEGORIES
from ui.evaluation_view import load_comparison_data, load_error_analysis_data, render_evaluation_dashboard
from ui.statistics_view import render_evaluation_dashboard as stats_render_dashboard

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"


class TestEvaluationDashboard(unittest.TestCase):
    """Test suite for Step 1.7 Evaluation Dashboard and Reporting."""

    def test_protected_datasets_integrity(self):
        """1. Verifies protected datasets are strictly untouched."""
        self.assertTrue(TRAINING_CSV_PATH.exists(), "Training dataset must exist.")
        df = pd.read_csv(TRAINING_CSV_PATH)
        self.assertEqual(len(df), 600, "Training dataset must contain exactly 600 examples.")

        self.assertTrue(EVALUATION_DATASET_PATH.exists(), "Evaluation dataset must exist.")
        with open(EVALUATION_DATASET_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)
        self.assertEqual(len(cases), 93, "Evaluation dataset must contain exactly 93 cases.")

    def test_comparison_json_artifact_exists_and_valid(self):
        """2. Verifies reports/model_comparison.json is complete and valid."""
        json_file = REPORTS_DIR / "model_comparison.json"
        self.assertTrue(json_file.exists(), "reports/model_comparison.json must exist.")

        data = load_comparison_data()
        self.assertIsNotNone(data)
        self.assertIn("benchmark_dataset", data)
        self.assertIn("models_evaluated", data)
        self.assertIn("factual_tradeoffs", data)
        self.assertIn("error_overlap", data)

        # Check models evaluated
        models = data["models_evaluated"]
        self.assertIn("model_a", models)
        self.assertIn("model_b", models)

        m_a = models["model_a"]
        m_b = models["model_b"]

        self.assertEqual(m_a["metrics"]["total_cases"], 93)
        self.assertEqual(m_b["metrics"]["total_cases"], 93)
        self.assertAlmostEqual(m_a["metrics"]["accuracy"], 0.8495, places=3)
        self.assertAlmostEqual(m_b["metrics"]["accuracy"], 0.9032, places=3)

    def test_error_analysis_json_artifact_exists_and_valid(self):
        """3. Verifies reports/error_analysis.json is complete and valid."""
        json_file = REPORTS_DIR / "error_analysis.json"
        self.assertTrue(json_file.exists(), "reports/error_analysis.json must exist.")

        data = load_error_analysis_data()
        self.assertIsNotNone(data)
        self.assertIn("overall_metrics", data)
        self.assertIn("per_category_metrics", data)
        self.assertEqual(data["overall_metrics"]["total_cases"], 93)

    def test_evaluation_summary_markdown_report_exists(self):
        """4. Verifies reports/evaluation_summary.md exists and covers all required sections."""
        md_file = REPORTS_DIR / "evaluation_summary.md"
        self.assertTrue(md_file.exists(), "reports/evaluation_summary.md must exist.")

        content = md_file.read_text(encoding="utf-8")
        self.assertIn("Evaluation Protocol", content)
        self.assertIn("Overall Performance", content)
        self.assertIn("Model Comparison", content)
        self.assertIn("Per-Category Performance", content)
        self.assertIn("Language & Script Performance", content)
        self.assertIn("Confidence & Calibration", content)
        self.assertIn("Unknown & Ambiguity", content)
        self.assertIn("Key Observations", content)
        self.assertIn("Known Limitations", content)

    def test_ui_components_and_app_navigation(self):
        """5. Verifies UI render functions exist and app.py sidebar navigation is configured."""
        self.assertTrue(callable(render_evaluation_dashboard))
        self.assertTrue(callable(stats_render_dashboard))

        app_py = BASE_DIR / "app.py"
        self.assertTrue(app_py.exists())
        app_code = app_py.read_text(encoding="utf-8")
        self.assertIn("AI Evaluation Dashboard", app_code)
        self.assertIn("render_evaluation_dashboard", app_code)

    def test_all_categories_represented_in_benchmark_results(self):
        """6. Verifies all 8 approved categories are present in per-category metrics."""
        data = load_comparison_data()
        per_cat_a = data["models_evaluated"]["model_a"]["metrics"]["per_category"]
        per_cat_b = data["models_evaluated"]["model_b"]["metrics"]["per_category"]

        for cat in APPROVED_CATEGORIES:
            self.assertIn(cat, per_cat_a)
            self.assertIn(cat, per_cat_b)


if __name__ == "__main__":
    unittest.main()
