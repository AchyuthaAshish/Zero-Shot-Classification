"""Unit tests for Confidence Calibration (Phase 1 Step 1.2).

Verifies:
1. Calibration model can be trained on internal data.
2. Calibration uses only permitted training/internal-validation data (600 examples).
3. 93-case held-out dataset is strictly NOT used for calibration or tuning.
4. Calibrated probabilities are valid floats.
5. Calibrated probabilities are strictly within [0, 1].
6. Multiclass probabilities sum approximately to 1.0 across all classes.
7. ConfidenceAssessment correctly stores calibrated_prob.
8. is_calibrated is True only after valid calibration.
9. calibration_method is populated correctly ("Temperature Scaling").
10. Raw score remains available alongside calibrated probability.
11. Gemini external provider remains purely qualitative (no fake probability).
12. Unknown behavior remains intact (level="Uncertain", is_calibrated=False).
13. HYBRID mode still functions properly with calibrated probabilities.
14. Reliability diagram artifact and calibration metrics exist on disk.
15. Backward compatibility with existing ClassificationResult.confidence.
"""

import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from core.schemas import ClassificationResult, PreprocessedInput, ConfidenceAssessment
from ml.calibration import MulticlassTemperatureScaler, calculate_ece, multiclass_brier_score
from ml.local_classifier import LocalDefectClassifier, get_local_classifier
from classification.classifier import DefectClassifier
from llm.client import FakeLLMClient
from ml.config import EMBEDDING_MODEL_DIR, APPROVED_CATEGORIES, TRAINING_CSV_PATH, EVALUATION_DATASET_PATH


class TestConfidenceCalibration(unittest.TestCase):
    """Test suite for Phase 1 Step 1.2 Confidence Calibration."""

    @classmethod
    def setUpClass(cls):
        cls.train_csv = TRAINING_CSV_PATH
        cls.eval_json = EVALUATION_DATASET_PATH

    def test_protected_datasets_integrity(self):
        """1 & 2 & 3. Verifies protected datasets are intact and holdout is strictly untouched."""
        # 1. Verify 600 training examples exist
        self.assertTrue(self.train_csv.exists(), "Training dataset must exist at data/training/defect_training.csv.")
        import pandas as pd
        df = pd.read_csv(self.train_csv)
        self.assertEqual(len(df), 600, "Training dataset must contain exactly 600 examples.")

        # 2. Verify 93 evaluation cases exist
        self.assertTrue(self.eval_json.exists(), "Evaluation dataset must exist.")
        with open(self.eval_json, "r", encoding="utf-8") as f:
            eval_cases = json.load(f)
        self.assertEqual(len(eval_cases), 93, "Evaluation dataset must contain exactly 93 cases.")

        # 3. Verify metadata documents only internal data was used
        meta_file = EMBEDDING_MODEL_DIR / "metadata.json"
        self.assertTrue(meta_file.exists(), "Model metadata file must exist.")
        with open(meta_file, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
        self.assertEqual(meta_data.get("training_dataset_size"), 600)
        self.assertEqual(meta_data.get("train_examples"), 480)
        self.assertEqual(meta_data.get("validation_examples"), 120)

    def test_temperature_scaler_mathematical_properties(self):
        """4, 5, 6. Verifies MulticlassTemperatureScaler satisfies probability axioms."""
        # Synthetic mock base estimator
        class MockEstimator:
            def __init__(self):
                self.classes_ = np.array(APPROVED_CATEGORIES)
            def decision_function(self, X):
                # Deterministic pseudo-logits for 5 samples, 8 classes
                np.random.seed(42)
                return np.random.randn(len(X), len(self.classes_))
            def predict_proba(self, X):
                logits = self.decision_function(X)
                exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                return exp / np.sum(exp, axis=1, keepdims=True)

        mock_base = MockEstimator()
        scaler = MulticlassTemperatureScaler(base_estimator=mock_base, temperature=0.5)

        X_dummy = np.zeros((5, 10))
        calibrated_probs = scaler.predict_proba(X_dummy)
        raw_probs = scaler.predict_raw_proba(X_dummy)

        # 4 & 5. Valid probabilities within [0, 1]
        self.assertEqual(calibrated_probs.shape, (5, 8))
        self.assertTrue(np.all(calibrated_probs >= 0.0), "All probabilities must be >= 0.")
        self.assertTrue(np.all(calibrated_probs <= 1.0), "All probabilities must be <= 1.")

        # 6. Sum to 1.0 across classes
        row_sums = np.sum(calibrated_probs, axis=1)
        np.testing.assert_allclose(row_sums, np.ones(5), atol=1e-5, err_msg="Probabilities must sum to 1.")

        # Ranking preservation: argmax must be identical between raw and calibrated
        raw_preds = np.argmax(raw_probs, axis=1)
        calibrated_preds = np.argmax(calibrated_probs, axis=1)
        np.testing.assert_array_equal(raw_preds, calibrated_preds, "Temperature scaling must preserve argmax ranking.")

    def test_calibrated_classifier_artifact_loading(self):
        """7, 8, 9, 10. Verifies LocalDefectClassifier loads calibrated model and populates ConfidenceAssessment."""
        classifier = LocalDefectClassifier(model_type="multilingual_embedding", use_calibration=True)
        self.assertTrue(classifier.is_calibrated)
        self.assertEqual(classifier.calibration_method, "Temperature Scaling")

        result = classifier.classify("Motor bearing is overheating and vibrating excessively.")

        self.assertIsInstance(result.confidence_assessment, ConfidenceAssessment)
        ca = result.confidence_assessment

        # 7. ConfidenceAssessment correctly stores calibrated_prob
        self.assertIsNotNone(ca.calibrated_prob)
        self.assertGreaterEqual(ca.calibrated_prob, 0.0)
        self.assertLessEqual(ca.calibrated_prob, 1.0)

        # 8. is_calibrated is True
        self.assertTrue(ca.is_calibrated)

        # 9. calibration_method populated correctly
        self.assertEqual(ca.calibration_method, "Temperature Scaling")

        # 10. Raw score remains available alongside calibrated prob
        self.assertIsNotNone(ca.raw_score)
        self.assertGreaterEqual(ca.raw_score, 0.0)
        self.assertLessEqual(ca.raw_score, 1.0)

        # Approximate range shows calibrated percentage
        self.assertTrue(ca.approximate_range.startswith("~"))

        # Backward compatibility: result.confidence matches calibrated_prob
        self.assertEqual(result.confidence, ca.calibrated_prob)

    def test_gemini_remains_purely_qualitative(self):
        """11. Verifies Gemini external provider never fabricates statistical calibration."""
        fake_client = FakeLLMClient(
            canned_responses={
                "bearing overheating": {
                    "category": "Temperature Fault",
                    "reason": "Overheating bearing indicates thermal breakdown.",
                    "language": "English",
                    "reliability": "High"
                }
            }
        )
        classifier = DefectClassifier(llm_client=fake_client, mode="gemini")
        result = classifier.classify("Bearing overheating during peak load.")

        self.assertEqual(result.category, "Temperature Fault")
        self.assertIsNone(result.confidence)
        ca = result.confidence_assessment
        self.assertEqual(ca.level, "High")
        self.assertIsNone(ca.raw_score)
        self.assertIsNone(ca.calibrated_prob)
        self.assertFalse(ca.is_calibrated)
        self.assertIsNone(ca.calibration_method)
        self.assertEqual(ca.approximate_range, "Qualitative / N/A")

    def test_unknown_behavior_preserved(self):
        """12. Verifies Unknown outcomes preserve Uncertain level and are never marked as calibrated."""
        classifier = LocalDefectClassifier(model_type="multilingual_embedding")
        # Preprocessing failure on whitespace-only input
        result = classifier.classify("   ")
        self.assertEqual(result.category, "Unknown")
        self.assertEqual(result.status, "input_error")
        ca = result.confidence_assessment
        self.assertEqual(ca.level, "Uncertain")
        self.assertEqual(ca.approximate_range, "Insufficient Evidence")
        self.assertIsNone(ca.calibrated_prob)
        self.assertFalse(ca.is_calibrated)
        self.assertTrue(ca.is_ambiguous)

    def test_hybrid_mode_with_calibration(self):
        """13. Verifies HYBRID routing continues to function seamlessly with calibrated probabilities."""
        fake_client = FakeLLMClient(
            canned_responses={
                "unclear vibration": {
                    "category": "Mechanical Fault",
                    "reason": "LLM classified low confidence vibration.",
                    "language": "English",
                    "reliability": "High"
                }
            }
        )
        classifier = DefectClassifier(llm_client=fake_client, mode="hybrid")

        # Case A: High confidence local prediction -> returns local ML
        high_conf_probs = {cat: 0.01 for cat in APPROVED_CATEGORIES}
        high_conf_probs["Mechanical Fault"] = 0.92
        with patch.object(classifier.local_classifier, "predict_with_confidence", return_value=("Mechanical Fault", 0.92, high_conf_probs)):
            result_local = classifier.classify("Clear mechanical gear wear.")
            self.assertEqual(result_local.category, "Mechanical Fault")
            self.assertEqual(result_local.model_source, "local_ml")
            self.assertEqual(result_local.confidence, 0.92)

        # Case B: Low confidence local prediction (< 0.70) -> falls back to LLM
        low_conf_probs = {cat: 0.125 for cat in APPROVED_CATEGORIES}
        with patch.object(classifier.local_classifier, "predict_with_confidence", return_value=("Mechanical Fault", 0.35, low_conf_probs)):
            result_fallback = classifier.classify("unclear vibration")
            self.assertEqual(result_fallback.category, "Mechanical Fault")
            self.assertIn("fallback", result_fallback.model_source)

    def test_calibration_artifacts_exist(self):
        """14. Verifies reliability diagram and calibration metrics artifacts were generated and saved."""
        rel_diag = EMBEDDING_MODEL_DIR / "reliability_diagram.md"
        cal_metrics = EMBEDDING_MODEL_DIR / "calibration_metrics.json"
        cal_model = EMBEDDING_MODEL_DIR / "calibrated_classifier.joblib"

        self.assertTrue(rel_diag.exists(), "Reliability diagram markdown artifact must exist.")
        self.assertTrue(cal_metrics.exists(), "Calibration metrics JSON artifact must exist.")
        self.assertTrue(cal_model.exists(), "Calibrated model joblib artifact must exist.")

        with open(cal_metrics, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("oof_5fold_metrics", data)
        oof = data["oof_5fold_metrics"]
        self.assertIn("Temperature Scaling", oof)
        self.assertIn("Uncalibrated MiniLM", oof)
        ts_metrics = oof["Temperature Scaling"]
        uncal_metrics = oof["Uncalibrated MiniLM"]

        self.assertLess(ts_metrics["ece"], uncal_metrics["ece"], "ECE must improve after calibration.")
        self.assertLess(ts_metrics["brier_score"], uncal_metrics["brier_score"], "Brier score must improve after calibration.")
        self.assertLess(ts_metrics["log_loss"], uncal_metrics["log_loss"], "Log loss must improve after calibration.")

    def test_metric_calculation_functions(self):
        """15. Verifies ECE and Brier score calculation functions."""
        # Simulated prediction probabilities
        probs = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.2, 0.8]
        ])
        y_true = np.array([0, 0, 1, 1])
        brier = multiclass_brier_score(y_true, probs, n_classes=2)
        self.assertIsInstance(brier, float)
        self.assertGreaterEqual(brier, 0.0)

        ece_score, bin_details = calculate_ece(y_true, probs, n_bins=5)
        self.assertIsInstance(ece_score, float)
        self.assertGreaterEqual(ece_score, 0.0)
        self.assertLessEqual(ece_score, 1.0)
        self.assertEqual(len(bin_details), 5)


if __name__ == "__main__":
    unittest.main()
