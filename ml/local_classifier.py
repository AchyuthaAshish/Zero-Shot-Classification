"""Local Defect Classifier.

Provides offline defect classification using either:
1. Primary: Multilingual Sentence Transformer + LogisticRegression
2. Baseline: TF-IDF + LinearSVC

Conforms strictly to 8-category taxonomy, applies confidence thresholding,
and produces canonical ClassificationResult objects without any API calls.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import joblib
import numpy as np

from core.schemas import ClassificationResult, PreprocessedInput, ConfidenceAssessment
from core.exceptions import ConfigurationError, ModelError, TaxonomyError
from preprocessing.text_processor import get_text_processor, TextProcessor
from taxonomy.repository import get_taxonomy_repository, TaxonomyRepository
from ml.config import (
    EMBEDDING_MODEL_DIR,
    TFIDF_SVM_MODEL_DIR,
    DEFAULT_LOCAL_CONFIDENCE_THRESHOLD,
    APPROVED_CATEGORIES
)
from ml.embedding_model import encode_texts


class LocalDefectClassifier:
    """Offline local ML classifier for industrial defect descriptions."""

    def __init__(
        self,
        model_type: str = "multilingual_embedding",  # "multilingual_embedding" or "tfidf_svm"
        confidence_threshold: float = DEFAULT_LOCAL_CONFIDENCE_THRESHOLD,
        text_processor: Optional[TextProcessor] = None,
        taxonomy_repo: Optional[TaxonomyRepository] = None,
        use_calibration: bool = True
    ):
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.text_processor = text_processor or get_text_processor()
        self.taxonomy_repo = taxonomy_repo or get_taxonomy_repository()
        self.use_calibration = use_calibration
        self.model = None
        self.is_calibrated = False
        self.calibration_method = None
        self._last_raw_score = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads trained local model artifact."""
        if self.model_type == "multilingual_embedding":
            calibrated_path = EMBEDDING_MODEL_DIR / "calibrated_classifier.joblib"
            model_path = EMBEDDING_MODEL_DIR / "classifier.joblib"
            if self.use_calibration and calibrated_path.exists():
                self.model = joblib.load(calibrated_path)
                self.is_calibrated = True
                self.calibration_method = getattr(self.model, "calibration_method", "Temperature Scaling")
            elif model_path.exists():
                self.model = joblib.load(model_path)
                self.is_calibrated = False
                self.calibration_method = None
            else:
                # Check if baseline is available
                baseline_path = TFIDF_SVM_MODEL_DIR / "model.joblib"
                if baseline_path.exists():
                    self.model_type = "tfidf_svm"
                    self.model = joblib.load(baseline_path)
                    self.is_calibrated = True
                    self.calibration_method = "Platt scaling (CalibratedClassifierCV)"
                    return
                raise ModelError(f"Trained local embedding model not found at {model_path}. Please run training first.")
        elif self.model_type == "tfidf_svm":
            model_path = TFIDF_SVM_MODEL_DIR / "model.joblib"
            if not model_path.exists():
                raise ModelError(f"Trained baseline model not found at {model_path}. Please run training first.")
            self.model = joblib.load(model_path)
            self.is_calibrated = True
            self.calibration_method = "Platt scaling (CalibratedClassifierCV)"
        else:
            raise ConfigurationError(f"Unknown local model type: {self.model_type}")

        # Ensure scikit-learn cross-version compatibility for LogisticRegression
        if self.model is not None:
            if not hasattr(self.model, "multi_class"):
                setattr(self.model, "multi_class", "auto")
            if hasattr(self.model, "base_estimator") and not hasattr(self.model.base_estimator, "multi_class"):
                setattr(self.model.base_estimator, "multi_class", "auto")

    def predict_with_confidence(self, normalized_text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Returns (predicted_category, confidence_score, all_class_probabilities).
        """
        if self.model is None:
            self._load_model()

        if self.model_type == "multilingual_embedding":
            # 1. Generate multilingual embedding
            embedding = encode_texts([normalized_text], batch_size=1)
            probs = self.model.predict_proba(embedding)[0]
            classes = self.model.classes_
            best_idx = int(np.argmax(probs))

            if hasattr(self.model, "predict_raw_proba"):
                raw_probs = self.model.predict_raw_proba(embedding)[0]
                self._last_raw_score = float(raw_probs[best_idx])
            else:
                self._last_raw_score = float(probs[best_idx])
        else:
            # TF-IDF pipeline
            probs = self.model.predict_proba([normalized_text])[0]
            classes = self.model.classes_
            best_idx = int(np.argmax(probs))
            self._last_raw_score = float(probs[best_idx])

        class_prob_map = {cls_name: float(prob) for cls_name, prob in zip(classes, probs)}
        predicted_category = str(classes[best_idx])
        confidence = float(probs[best_idx])

        return predicted_category, confidence, class_prob_map

    def classify(self, raw_defect_text: Optional[str], _is_subsegment: bool = False) -> ClassificationResult:
        """
        Classifies defect text offline and returns canonical ClassificationResult.
        """
        safe_raw_text = raw_defect_text or ""

        # 1. Preprocessing and validation
        try:
            processed = self.text_processor.preprocess(raw_defect_text)
        except Exception as e:
            return ClassificationResult(
                category="Unknown",
                reason=f"Input processing failed: {e}",
                language="Unknown",
                reliability="Low",
                status="input_error",
                original_description=safe_raw_text,
                normalized_description=safe_raw_text,
                error_message=str(e),
                model_source="local_ml",
                confidence=0.0,
                confidence_assessment=ConfidenceAssessment(
                    level="Uncertain",
                    approximate_range="Insufficient Evidence",
                    raw_score=None,
                    calibrated_prob=None,
                    top2_margin=None,
                    is_calibrated=False,
                    calibration_method=None,
                    is_ambiguous=True
                )
            )

        # 2. Local model inference
        try:
            pred_cat, conf, prob_map = self.predict_with_confidence(processed.normalized_text)
        except Exception as e:
            return ClassificationResult(
                category="Unknown",
                reason=f"Local model inference error: {e}",
                language=processed.detected_language,
                reliability="Low",
                status="model_error",
                original_description=safe_raw_text,
                normalized_description=processed.normalized_text,
                error_message=str(e),
                model_source="local_ml",
                confidence=0.0,
                confidence_assessment=ConfidenceAssessment(
                    level="Uncertain",
                    approximate_range="Insufficient Evidence",
                    raw_score=None,
                    calibrated_prob=None,
                    top2_margin=None,
                    is_calibrated=False,
                    calibration_method=None,
                    is_ambiguous=True
                )
            )

        # Margin calculation: top2_margin = P(top1) - P(top2)
        top2_margin = None
        if prob_map and len(prob_map) >= 2:
            sorted_probs = sorted(prob_map.values(), reverse=True)
            top2_margin = round(float(sorted_probs[0] - sorted_probs[1]), 4)

        # 3. Strict Taxonomy Validation
        if not self.taxonomy_repo.is_valid_category(pred_cat):
            pred_cat = "Unknown"
            conf = 0.0

        # 4. Confidence Threshold Assessment
        is_high_confidence = conf >= self.confidence_threshold
        if conf >= 0.85:
            reliability = "High"
        elif conf >= 0.70:
            reliability = "Medium"
        else:
            reliability = "Low"

        # Determine status
        if pred_cat == "Unknown":
            status = "unknown"
        elif is_high_confidence:
            status = "success"
        else:
            status = "low_confidence"

        # Retrieve raw score if available
        raw_score = getattr(self, "_last_raw_score", None)
        if raw_score is None:
            raw_score = conf

        # Construct ConfidenceAssessment
        if pred_cat == "Unknown":
            ca = ConfidenceAssessment(
                level="Uncertain",
                approximate_range="Insufficient Evidence",
                raw_score=round(raw_score, 4) if raw_score > 0.0 else None,
                calibrated_prob=None,
                top2_margin=top2_margin,
                is_calibrated=False,
                calibration_method=None,
                is_ambiguous=True
            )
        elif self.is_calibrated:
            ca = ConfidenceAssessment(
                level=reliability,
                approximate_range=f"~{round(conf * 100)}%",
                raw_score=round(raw_score, 4),
                calibrated_prob=round(conf, 4),
                top2_margin=top2_margin,
                is_calibrated=True,
                calibration_method=self.calibration_method or "Temperature Scaling",
                is_ambiguous=bool(top2_margin is not None and top2_margin < 0.15) or (reliability == "Low")
            )
        else:
            ca = ConfidenceAssessment(
                level=reliability,
                approximate_range="Uncalibrated model score",
                raw_score=round(raw_score, 4),
                calibrated_prob=None,
                top2_margin=top2_margin,
                is_calibrated=False,
                calibration_method=None,
                is_ambiguous=bool(top2_margin is not None and top2_margin < 0.15) or (reliability == "Low")
            )

        # 5. Unknown Detection & Evidence Verification Layer (Phase 1 Step 1.3)
        from core.unknown_detector import get_unknown_detector
        ud_result = get_unknown_detector().evaluate_unknown(
            raw_or_preprocessed=processed,
            candidate_category=pred_cat,
            confidence=conf,
            confidence_assessment=ca,
            top2_margin=top2_margin,
            class_probabilities=prob_map,
            model_source="local_ml"
        )

        final_cat = ud_result.final_category
        final_status = ud_result.status
        final_ca = ud_result.confidence_assessment

        from ml.explanation import generate_local_explanation
        if final_cat == "Unknown":
            final_conf = None
            reliability = "Low" if final_status.endswith("_error") else "Medium"
            reason = ud_result.reason
        else:
            final_conf = round(conf, 4)
            reliability = final_ca.level
            reason = generate_local_explanation(processed.normalized_text, final_cat, conf, raw_text=safe_raw_text)

        # 6. Ambiguity Detection Layer (Phase 1 Step 1.4)
        from core.ambiguity_detector import get_ambiguity_detector
        ambiguity_assessment = get_ambiguity_detector().evaluate_ambiguity(
            raw_or_preprocessed=processed,
            candidate_category=final_cat,
            confidence=conf,
            confidence_assessment=final_ca,
            top2_margin=top2_margin,
            class_probabilities=prob_map,
            model_source="local_ml"
        )

        multi_defect_assessment = None
        segmentation_result = None
        multi_defect_classification = None

        if not _is_subsegment:
            # 7. Multi-Defect Detection Layer (Phase 2 Step 2.1)
            from core.multi_defect_detector import get_multi_defect_detector
            multi_defect_assessment = get_multi_defect_detector().detect_multi_defect(
                raw_or_preprocessed=processed,
                candidate_category=final_cat
            )

            # 8. Defect Segmentation Layer (Phase 2 Step 2.2)
            from core.defect_segmenter import get_defect_segmenter
            segmentation_result = get_defect_segmenter().segment_defects(
                raw_or_preprocessed=processed,
                multi_defect_assessment=multi_defect_assessment
            )

            # 9. Per-Defect Classification Layer (Phase 2 Step 2.3)
            if segmentation_result.is_multi_defect and segmentation_result.segment_count >= 2:
                from core.per_defect_classifier import get_per_defect_classifier
                multi_defect_classification = get_per_defect_classifier().classify_segments(
                    segments=segmentation_result.segments,
                    original_text=safe_raw_text,
                    mode="local",
                    multi_defect_assessment=multi_defect_assessment,
                    segmentation_result=segmentation_result,
                    local_classifier=self
                )

        return ClassificationResult(
            category=final_cat,
            reason=reason,
            language=processed.detected_language,
            reliability=reliability,
            status=final_status,
            original_description=safe_raw_text,
            normalized_description=processed.normalized_text,
            error_message=None if final_status not in ("model_error", "validation_error") else ud_result.reason,
            model_source="local_ml",
            confidence=final_conf,
            confidence_assessment=final_ca,
            ambiguity_assessment=ambiguity_assessment,
            multi_defect_assessment=multi_defect_assessment,
            segmentation_result=segmentation_result,
            multi_defect_classification=multi_defect_classification
        )


_local_classifier_instance: Optional[LocalDefectClassifier] = None


def get_local_classifier(
    model_type: str = "multilingual_embedding",
    use_calibration: bool = True,
    force_reload: bool = False
) -> LocalDefectClassifier:
    """Returns singleton LocalDefectClassifier instance."""
    global _local_classifier_instance
    if (
        _local_classifier_instance is None
        or force_reload
        or getattr(_local_classifier_instance, "model_type", None) != model_type
        or getattr(_local_classifier_instance, "use_calibration", None) != use_calibration
    ):
        _local_classifier_instance = LocalDefectClassifier(model_type=model_type, use_calibration=use_calibration)
    return _local_classifier_instance


def classify_local(
    description: Optional[str],
    model_type: str = "multilingual_embedding",
    use_calibration: bool = True
) -> ClassificationResult:
    """
    Public convenience API for free offline local classification.
    """
    classifier = get_local_classifier(model_type=model_type, use_calibration=use_calibration)
    return classifier.classify(description)
