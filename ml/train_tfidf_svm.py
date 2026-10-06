"""Baseline ML Training: TF-IDF Vectorizer + LinearSVC.

Trains a traditional machine learning baseline on the training dataset.
Measures actual training and validation times.
Saves model artifact and metadata to models/tfidf_svm/.
"""

import time
import json
from datetime import datetime
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

from ml.config import (
    TFIDF_SVM_MODEL_DIR,
    RANDOM_SEED,
    DEFAULT_LOCAL_CONFIDENCE_THRESHOLD,
    APPROVED_CATEGORIES
)
from ml.dataset import load_and_validate_training_dataset, split_training_data


def train_tfidf_svm_model(save_artifacts: bool = True):
    """Trains TF-IDF + Calibrated LinearSVC pipeline and measures timings."""
    print("=" * 60)
    print("      TRAINING BASELINE: TF-IDF + LinearSVC")
    print("=" * 60)

    # 1. Dataset loading and validation
    t0_data = time.perf_counter()
    df = load_and_validate_training_dataset()
    train_df, val_df = split_training_data(df)
    data_prep_time = time.perf_counter() - t0_data

    X_train, y_train = train_df["text"].tolist(), train_df["category"].tolist()
    X_val, y_val = val_df["text"].tolist(), val_df["category"].tolist()

    # 2. Pipeline setup: TF-IDF + Calibrated LinearSVC (for probabilities)
    base_svc = LinearSVC(random_state=RANDOM_SEED, dual="auto", max_iter=2000)
    calibrated_svc = CalibratedClassifierCV(estimator=base_svc, cv=3)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", calibrated_svc)
    ])

    # 3. Model training
    t0_train = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_time = time.perf_counter() - t0_train

    # 4. Validation evaluation
    t0_val = time.perf_counter()
    val_preds = pipeline.predict(X_val)
    val_probs = pipeline.predict_proba(X_val)
    val_time = time.perf_counter() - t0_val

    val_acc = accuracy_score(y_val, val_preds)
    val_f1_macro = f1_score(y_val, val_preds, average="macro", zero_division=0)
    report = classification_report(y_val, val_preds, zero_division=0, output_dict=True)

    # 5. Measure single prediction latency
    t0_pred = time.perf_counter()
    sample_text = ["Conveyor motor is making a loud abnormal noise."]
    for _ in range(50):
        _ = pipeline.predict_proba(sample_text)
    avg_pred_latency_ms = ((time.perf_counter() - t0_pred) / 50) * 1000

    total_time = data_prep_time + training_time + val_time

    # 6. Save artifacts
    if save_artifacts:
        TFIDF_SVM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_file = TFIDF_SVM_MODEL_DIR / "model.joblib"
        metadata_file = TFIDF_SVM_MODEL_DIR / "metadata.json"

        joblib.dump(pipeline, model_file)

        metadata = {
            "model_type": "TF-IDF + LinearSVC (Calibrated)",
            "pipeline": ["TfidfVectorizer(ngram_range=(1,2))", "CalibratedClassifierCV(LinearSVC)"],
            "training_dataset_size": len(df),
            "train_examples": len(train_df),
            "validation_examples": len(val_df),
            "class_distribution": df["category"].value_counts().to_dict(),
            "random_seed": RANDOM_SEED,
            "confidence_threshold": DEFAULT_LOCAL_CONFIDENCE_THRESHOLD,
            "training_timestamp": datetime.utcnow().isoformat() + "Z",
            "validation_metrics": {
                "accuracy": round(float(val_acc), 4),
                "macro_f1": round(float(val_f1_macro), 4)
            },
            "timing_measurements": {
                "dataset_prep_time_seconds": round(float(data_prep_time), 4),
                "training_time_seconds": round(float(training_time), 4),
                "validation_time_seconds": round(float(val_time), 4),
                "total_time_seconds": round(float(total_time), 4),
                "avg_prediction_latency_ms": round(float(avg_pred_latency_ms), 3)
            }
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    # 7. Print results
    print("\n" + "=" * 60)
    print("           TF-IDF + LinearSVC RESULTS")
    print("=" * 60)
    print(f"Validation Accuracy   : {val_acc:.4f} ({val_acc * 100:.2f}%)")
    print(f"Validation Macro F1   : {val_f1_macro:.4f}")
    print(f"Data Prep Time        : {data_prep_time:.4f}s")
    print(f"Training Time         : {training_time:.4f}s")
    print(f"Validation Time       : {val_time:.4f}s")
    print(f"Total Time            : {total_time:.4f}s")
    print(f"Avg Single Latency    : {avg_pred_latency_ms:.2f}ms")
    print("=" * 60)

    return {
        "pipeline": pipeline,
        "accuracy": val_acc,
        "macro_f1": val_f1_macro,
        "report": report,
        "timings": {
            "data_prep": data_prep_time,
            "training": training_time,
            "validation": val_time,
            "total": total_time,
            "single_latency_ms": avg_pred_latency_ms
        }
    }


if __name__ == "__main__":
    train_tfidf_svm_model()
