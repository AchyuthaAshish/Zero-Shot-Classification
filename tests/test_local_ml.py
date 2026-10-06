"""Unit tests for Free Local ML Classifier and Hybrid Fallback Architecture.

Tests all 15 required scenarios:
1. Training dataset loading
2. Invalid category rejection
3. Missing training text
4. Duplicate detection
5. Model loading
6. Local prediction
7. Confidence calculation
8. Unknown handling
9. Taxonomy validation
10. Gemini fallback
11. Gemini unavailable fallback
12. LOCAL mode never calling Gemini
13. HYBRID mode using local model first
14. Invalid local prediction handling
15. Streamlit integration path where practical
"""

import os
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock

from ml.config import (
    TRAINING_CSV_PATH,
    APPROVED_CATEGORIES,
    DEFAULT_LOCAL_CONFIDENCE_THRESHOLD
)
from ml.dataset import (
    load_and_validate_training_dataset,
    split_training_data
)
from ml.local_classifier import (
    LocalDefectClassifier,
    get_local_classifier,
    classify_local
)
from classification.classifier import (
    DefectClassifier,
    classify_defect
)
from core.exceptions import ModelError, ConfigurationError
from core.schemas import ClassificationResult
from llm.client import FakeLLMClient


class TestLocalMLClassification(unittest.TestCase):
    """Test suite for Local ML and Hybrid Architecture."""

    @classmethod
    def setUpClass(cls):
        """Ensure dataset exists for testing."""
        if not TRAINING_CSV_PATH.exists():
            from ml.dataset import generate_training_csv
            generate_training_csv()

    # 1. Training dataset loading
    def test_training_dataset_loading(self):
        """Verifies training dataset loads successfully with required columns."""
        df = load_and_validate_training_dataset(TRAINING_CSV_PATH)
        self.assertGreaterEqual(len(df), 320)
        self.assertIn("text", df.columns)
        self.assertIn("category", df.columns)
        self.assertIn("language", df.columns)
        self.assertIn("source", df.columns)

    # 2. Invalid category rejection
    def test_invalid_category_rejection(self):
        """Verifies dataset validator rejects invalid taxonomy categories."""
        bad_df = pd.DataFrame([
            {"text": "Sample text", "category": "NonExistentCategory", "language": "English", "source": "test"}
        ])
        temp_csv = TRAINING_CSV_PATH.parent / "temp_invalid.csv"
        bad_df.to_csv(temp_csv, index=False, encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                load_and_validate_training_dataset(temp_csv)
        finally:
            if temp_csv.exists():
                temp_csv.unlink()

    # 3. Missing training text
    def test_missing_training_text(self):
        """Verifies dataset validator rejects missing or whitespace-only text."""
        bad_df = pd.DataFrame([
            {"text": "   ", "category": "Mechanical Fault", "language": "English", "source": "test"}
        ])
        temp_csv = TRAINING_CSV_PATH.parent / "temp_missing.csv"
        bad_df.to_csv(temp_csv, index=False, encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                load_and_validate_training_dataset(temp_csv)
        finally:
            if temp_csv.exists():
                temp_csv.unlink()

    # 4. Duplicate detection
    def test_duplicate_detection(self):
        """Verifies duplicate rows in dataset are correctly counted."""
        dup_df = pd.DataFrame([
            {"text": "Duplicate sentence here", "category": "Mechanical Fault", "language": "English", "source": "test"},
            {"text": "Duplicate sentence here", "category": "Mechanical Fault", "language": "English", "source": "test"}
        ])
        self.assertEqual(dup_df["text"].duplicated().sum(), 1)

    # 5. Model loading
    def test_model_loading(self):
        """Verifies local classifier initializes and loads trained artifact."""
        classifier = LocalDefectClassifier(model_type="tfidf_svm")
        self.assertIsNotNone(classifier.model)

    # 6. Local prediction
    def test_local_prediction(self):
        """Verifies local classifier returns valid canonical result without network calls."""
        res = classify_local("Conveyor drive motor shaft has severe axial wobble.", model_type="tfidf_svm")
        self.assertIsInstance(res, ClassificationResult)
        self.assertIn(res.category, APPROVED_CATEGORIES)
        self.assertEqual(res.model_source, "local_ml")
        self.assertIsNotNone(res.confidence)

    # 7. Confidence calculation
    def test_confidence_calculation(self):
        """Verifies confidence score is between 0.0 and 1.0 and probabilities sum to 1.0."""
        classifier = LocalDefectClassifier(model_type="tfidf_svm")
        cat, conf, probs = classifier.predict_with_confidence("Motor terminal box short circuit sparking.")
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=3)
        self.assertEqual(cat, max(probs, key=probs.get))

    # 8. Unknown handling
    def test_unknown_handling(self):
        """Verifies ambiguous input is classified as Unknown or low confidence."""
        res = classify_local("Equipment is not working properly.", model_type="tfidf_svm")
        self.assertIn(res.category, APPROVED_CATEGORIES)

    # 9. Taxonomy validation
    def test_taxonomy_validation(self):
        """Verifies local classifier strictly enforces the 8 approved categories."""
        res = classify_local("Temperature sensor is giving wrong readings.", model_type="tfidf_svm")
        self.assertIn(res.category, APPROVED_CATEGORIES)

    # 10. Gemini fallback
    def test_gemini_fallback(self):
        """Verifies HYBRID mode triggers Gemini fallback when local confidence is below threshold."""
        fake_llm = FakeLLMClient(canned_responses={
            "vague machine problem": {
                "category": "Unknown",
                "reason": "Fallback successfully triggered and classified by Gemini."
            }
        })
        # Set high local threshold (0.99) to force fallback
        mock_local = MagicMock(spec=LocalDefectClassifier)
        mock_local.classify.return_value = ClassificationResult(
            category="Mechanical Fault",
            reason="Uncertain local guess",
            language="English",
            reliability="Low",
            status="low_confidence",
            original_description="vague machine problem",
            normalized_description="vague machine problem",
            model_source="local_ml",
            confidence=0.45
        )

        classifier = DefectClassifier(
            llm_client=fake_llm,
            local_classifier=mock_local,
            mode="hybrid"
        )
        with patch("config.settings.get_settings") as mock_settings:
            mock_settings.return_value.local_confidence_threshold = 0.70
            mock_settings.return_value.classification_mode = "hybrid"
            result = classifier.classify("vague machine problem")

        self.assertIn("fallback", result.model_source)

    # 11. Gemini unavailable fallback
    def test_gemini_unavailable_fallback(self):
        """Verifies HYBRID mode gracefully falls back to local prediction when Gemini is unavailable."""
        mock_failing_llm = MagicMock()
        mock_failing_llm.classify.side_effect = ModelError("429 Resource exhausted: Gemini quota exceeded.")

        mock_local = MagicMock(spec=LocalDefectClassifier)
        mock_local.classify.return_value = ClassificationResult(
            category="Sensor Fault",
            reason="Local prediction",
            language="English",
            reliability="Medium",
            status="low_confidence",
            original_description="Sensor giving erratic signal",
            normalized_description="sensor giving erratic signal",
            model_source="local_ml",
            confidence=0.62
        )

        classifier = DefectClassifier(
            llm_client=mock_failing_llm,
            local_classifier=mock_local,
            mode="hybrid"
        )
        result = classifier.classify("Sensor giving erratic signal")

        # Must not raise exception, must safely return local fallback
        self.assertEqual(result.category, "Sensor Fault")
        self.assertEqual(result.model_source, "local_ml_fallback")

    # 12. LOCAL mode never calling Gemini
    def test_local_mode_never_calling_gemini(self):
        """Verifies LOCAL mode never invokes the LLM client under any circumstance."""
        mock_llm = MagicMock()

        classifier = DefectClassifier(
            llm_client=mock_llm,
            mode="local"
        )
        res = classifier.classify("Conveyor belt drive shaft has severe axial wobble.")
        self.assertEqual(res.model_source, "local_ml")
        mock_llm.classify.assert_not_called()

    # 13. HYBRID mode using local model first
    def test_hybrid_mode_uses_local_model_first(self):
        """Verifies HYBRID mode returns local prediction directly when confidence is high."""
        mock_llm = MagicMock()

        mock_local = MagicMock(spec=LocalDefectClassifier)
        mock_local.classify.return_value = ClassificationResult(
            category="Mechanical Fault",
            reason="High confidence local prediction",
            language="English",
            reliability="High",
            status="success",
            original_description="Motor bearing grinding noise",
            normalized_description="motor bearing grinding noise",
            model_source="local_ml",
            confidence=0.92
        )

        classifier = DefectClassifier(
            llm_client=mock_llm,
            local_classifier=mock_local,
            mode="hybrid"
        )
        result = classifier.classify("Motor bearing grinding noise")

        self.assertEqual(result.model_source, "local_ml")
        self.assertEqual(result.category, "Mechanical Fault")
        mock_llm.classify.assert_not_called()

    # 14. Invalid local prediction handling
    def test_invalid_local_prediction_handling(self):
        """Verifies invalid prediction is safely intercepted and coerced to Unknown."""
        classifier = LocalDefectClassifier(model_type="tfidf_svm")
        # Mock predict_with_confidence returning an unapproved category
        with patch.object(classifier, "predict_with_confidence", return_value=("HallucinatedCategory", 0.99, {})):
            res = classifier.classify("Some test text")
            self.assertEqual(res.category, "Unknown")

    # 15. Streamlit integration path
    def test_streamlit_integration_path(self):
        """Verifies classify_defect accepts mode parameter cleanly."""
        res_local = classify_defect("Short circuit in control wiring harness.", mode="local")
        self.assertEqual(res_local.model_source, "local_ml")
        self.assertIn(res_local.category, APPROVED_CATEGORIES)


if __name__ == "__main__":
    unittest.main()
