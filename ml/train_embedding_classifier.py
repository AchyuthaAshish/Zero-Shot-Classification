"""Primary Local ML Training: Multilingual Sentence Transformer + LogisticRegression.

Uses frozen pretrained 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
to generate multilingual sentence embeddings and trains scikit-learn LogisticRegression.
Measures actual execution times, evaluates on validation data, and persists artifacts to
models/multilingual_embedding_classifier/.
"""

import time
import json
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, log_loss

from ml.config import (
    EMBEDDING_MODEL_DIR,
    PRETRAINED_MODEL_NAME,
    RANDOM_SEED,
    DEFAULT_LOCAL_CONFIDENCE_THRESHOLD,
    APPROVED_CATEGORIES
)
from ml.dataset import load_and_validate_training_dataset, split_training_data
from ml.embedding_model import get_embedding_model, encode_texts


def train_multilingual_embedding_classifier(save_artifacts: bool = True):
    """
    Trains LogisticRegression on frozen Multilingual MiniLM sentence embeddings.
    Records actual timings, validation metrics, and persists artifacts.
    """
    print("=" * 60)
    print("  TRAINING PRIMARY MODEL: Multilingual MiniLM + LogisticRegression")
    print("=" * 60)

    # 1. Dataset loading and validation
    t0_data = time.perf_counter()
    df = load_and_validate_training_dataset()
    train_df, val_df = split_training_data(df)
    data_prep_time = time.perf_counter() - t0_data

    X_train_raw = train_df["text"].tolist()
    y_train = train_df["category"].tolist()
    X_val_raw = val_df["text"].tolist()
    y_val = val_df["category"].tolist()

    # 2. Embedding generation (Frozen Sentence Transformer)
    print(f"\nLoading pretrained SentenceTransformer '{PRETRAINED_MODEL_NAME}'...")
    t0_load_emb = time.perf_counter()
    _ = get_embedding_model()
    model_load_time = time.perf_counter() - t0_load_emb
    print(f"Pretrained model loaded in {model_load_time:.2f}s.")

    print(f"Generating embeddings for {len(X_train_raw)} training examples...")
    t0_emb = time.perf_counter()
    X_train_emb = encode_texts(X_train_raw, batch_size=32, show_progress=False)
    emb_gen_time_train = time.perf_counter() - t0_emb

    print(f"Generating embeddings for {len(X_val_raw)} validation examples...")
    t0_val_emb = time.perf_counter()
    X_val_emb = encode_texts(X_val_raw, batch_size=32, show_progress=False)
    emb_gen_time_val = time.perf_counter() - t0_val_emb
    total_emb_time = emb_gen_time_train + emb_gen_time_val

    # 3. Train scikit-learn LogisticRegression classifier
    print("Fitting LogisticRegression classifier...")
    clf = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
        C=2.0,
        solver="lbfgs"
    )
    t0_train = time.perf_counter()
    clf.fit(X_train_emb, y_train)
    clf_train_time = time.perf_counter() - t0_train

    # 4. Calibration & Validation evaluation
    from ml.calibration import (
        MulticlassTemperatureScaler,
        fit_temperature_scaling,
        calculate_ece,
        multiclass_brier_score,
        evaluate_oof_calibration,
        format_reliability_diagram_markdown
    )

    t0_val = time.perf_counter()
    label_to_idx = {cat: i for i, cat in enumerate(clf.classes_)}
    y_val_indices = np.array([label_to_idx[cat] for cat in y_val])

    val_logits = clf.decision_function(X_val_emb)
    val_raw_probs = clf.predict_proba(X_val_emb)

    # Learn temperature on internal validation split
    opt_T = fit_temperature_scaling(val_logits, y_val_indices)
    calibrated_clf = MulticlassTemperatureScaler(base_estimator=clf, temperature=opt_T)
    val_cal_probs = calibrated_clf.predict_proba(X_val_emb)
    val_eval_time = time.perf_counter() - t0_val

    val_preds = calibrated_clf.predict(X_val_emb)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1_macro = f1_score(y_val, val_preds, average="macro", zero_division=0)
    report = classification_report(y_val, val_preds, zero_division=0, output_dict=True)

    # Calculate calibration metrics on validation split
    raw_ece, raw_bins = calculate_ece(y_val_indices, val_raw_probs)
    cal_ece, cal_bins = calculate_ece(y_val_indices, val_cal_probs)
    raw_brier = multiclass_brier_score(y_val_indices, val_raw_probs)
    cal_brier = multiclass_brier_score(y_val_indices, val_cal_probs)
    raw_ll = float(log_loss(y_val_indices, val_raw_probs))
    cal_ll = float(log_loss(y_val_indices, val_cal_probs))

    # Also run rigorous 5-fold Out-Of-Fold (OOF) cross-validation on full 600 training examples
    print("\nRunning Stratified 5-Fold Cross-Validation calibration evaluation on training data...")
    oof_eval = evaluate_oof_calibration(
        embeddings=np.vstack([X_train_emb, X_val_emb]),
        labels=y_train + y_val,
        n_splits=5,
        seed=RANDOM_SEED
    )
    oof_metrics = oof_eval["metrics"]
    mean_oof_temp = oof_eval["mean_temperature"]

    # 5. Measure single prediction latency (embedding + calibrated inference)
    sample_text = ["Conveyor motor is making a loud abnormal noise."]
    t0_latency = time.perf_counter()
    for _ in range(20):
        emb_sample = encode_texts(sample_text, batch_size=1)
        _ = calibrated_clf.predict_proba(emb_sample)
    avg_pred_latency_ms = ((time.perf_counter() - t0_latency) / 20) * 1000

    total_training_time = data_prep_time + total_emb_time + clf_train_time + val_eval_time

    # 6. Save artifacts
    if save_artifacts:
        EMBEDDING_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        classifier_file = EMBEDDING_MODEL_DIR / "classifier.joblib"
        calibrated_file = EMBEDDING_MODEL_DIR / "calibrated_classifier.joblib"
        metadata_file = EMBEDDING_MODEL_DIR / "metadata.json"
        cal_metrics_file = EMBEDDING_MODEL_DIR / "calibration_metrics.json"
        rel_diagram_file = EMBEDDING_MODEL_DIR / "reliability_diagram.md"

        # Save both uncalibrated and calibrated estimator
        joblib.dump(clf, classifier_file)
        joblib.dump(calibrated_clf, calibrated_file)

        calibration_summary = {
            "calibration_method": "Temperature Scaling",
            "temperature_validation_split": round(float(opt_T), 4),
            "temperature_5fold_mean": round(float(mean_oof_temp), 4),
            "fold_temperatures": oof_eval["fold_temperatures"],
            "validation_split_metrics": {
                "before_calibration": {
                    "ece": round(float(raw_ece), 4),
                    "brier_score": round(float(raw_brier), 4),
                    "log_loss": round(float(raw_ll), 4)
                },
                "after_calibration": {
                    "ece": round(float(cal_ece), 4),
                    "brier_score": round(float(cal_brier), 4),
                    "log_loss": round(float(cal_ll), 4)
                }
            },
            "oof_5fold_metrics": oof_metrics
        }

        with open(cal_metrics_file, "w", encoding="utf-8") as f:
            json.dump(calibration_summary, f, indent=2)

        # Generate markdown reliability diagram
        rel_md = (
            "# Reliability Diagram: Multilingual MiniLM Defect Classifier\n\n"
            "Evaluated using Stratified 5-Fold Cross-Validation on the 600-example training dataset.\n"
            "*Note: The 93-case held-out benchmark is strictly excluded.*\n\n"
            "## Summary Metrics\n\n"
            "| Model Configuration | ECE | Brier Score | Log Loss | Accuracy | Macro F1 |\n"
            "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
            f"| Uncalibrated MiniLM | {oof_metrics['Uncalibrated MiniLM']['ece']:.4f} | {oof_metrics['Uncalibrated MiniLM']['brier_score']:.4f} | {oof_metrics['Uncalibrated MiniLM']['log_loss']:.4f} | {oof_metrics['Uncalibrated MiniLM']['accuracy']:.4f} | {oof_metrics['Uncalibrated MiniLM']['macro_f1']:.4f} |\n"
            f"| Temperature Scaling (T={mean_oof_temp:.4f}) | {oof_metrics['Temperature Scaling']['ece']:.4f} | {oof_metrics['Temperature Scaling']['brier_score']:.4f} | {oof_metrics['Temperature Scaling']['log_loss']:.4f} | {oof_metrics['Temperature Scaling']['accuracy']:.4f} | {oof_metrics['Temperature Scaling']['macro_f1']:.4f} |\n"
            f"| Platt Scaling (CalibratedClassifierCV) | {oof_metrics['Platt Scaling (CalibratedClassifierCV)']['ece']:.4f} | {oof_metrics['Platt Scaling (CalibratedClassifierCV)']['brier_score']:.4f} | {oof_metrics['Platt Scaling (CalibratedClassifierCV)']['log_loss']:.4f} | {oof_metrics['Platt Scaling (CalibratedClassifierCV)']['accuracy']:.4f} | {oof_metrics['Platt Scaling (CalibratedClassifierCV)']['macro_f1']:.4f} |\n\n"
            "## Reliability Table: After Temperature Scaling\n\n"
            + format_reliability_diagram_markdown(oof_metrics["Temperature Scaling"]["bin_diagnostics"])
            + "\n\n## Reliability Table: Before Calibration (Uncalibrated MiniLM)\n\n"
            + format_reliability_diagram_markdown(oof_metrics["Uncalibrated MiniLM"]["bin_diagnostics"])
        )
        with open(rel_diagram_file, "w", encoding="utf-8") as f:
            f.write(rel_md)

        metadata = {
            "model_name": "Multilingual MiniLM + LogisticRegression (Temperature Scaled)",
            "pretrained_encoder": PRETRAINED_MODEL_NAME,
            "classifier_type": "LogisticRegression(max_iter=1000, C=2.0)",
            "calibration_method": "Temperature Scaling",
            "temperature": round(float(opt_T), 4),
            "training_dataset_size": len(df),
            "train_examples": len(train_df),
            "validation_examples": len(val_df),
            "classes": APPROVED_CATEGORIES,
            "class_distribution": df["category"].value_counts().to_dict(),
            "random_seed": RANDOM_SEED,
            "confidence_threshold": DEFAULT_LOCAL_CONFIDENCE_THRESHOLD,
            "training_timestamp": datetime.utcnow().isoformat() + "Z",
            "validation_metrics": {
                "accuracy": round(float(val_acc), 4),
                "macro_f1": round(float(val_f1_macro), 4),
                "ece_before": round(float(raw_ece), 4),
                "ece_after": round(float(cal_ece), 4)
            },
            "timing_measurements": {
                "dataset_prep_time_seconds": round(float(data_prep_time), 4),
                "model_loading_time_seconds": round(float(model_load_time), 4),
                "embedding_generation_time_seconds": round(float(total_emb_time), 4),
                "classifier_training_time_seconds": round(float(clf_train_time), 4),
                "validation_evaluation_time_seconds": round(float(val_eval_time), 4),
                "total_training_time_seconds": round(float(total_training_time), 4),
                "avg_single_prediction_latency_ms": round(float(avg_pred_latency_ms), 3)
            }
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    # 7. Print summary report
    print("\n" + "=" * 60)
    print("      MULTILINGUAL EMBEDDING CLASSIFIER RESULTS")
    print("=" * 60)
    print(f"Validation Accuracy   : {val_acc:.4f} ({val_acc * 100:.2f}%)")
    print(f"Validation Macro F1   : {val_f1_macro:.4f}")
    print(f"Learned Temperature  : {opt_T:.4f}")
    print(f"ECE (Raw -> Calibrated): {raw_ece:.4f} -> {cal_ece:.4f}")
    print(f"Log Loss (Raw -> Cal) : {raw_ll:.4f} -> {cal_ll:.4f}")
    print(f"Dataset Prep Time     : {data_prep_time:.4f}s")
    print(f"Embedding Gen Time    : {total_emb_time:.4f}s")
    print(f"Classifier Train Time : {clf_train_time:.4f}s")
    print(f"Validation Time       : {val_eval_time:.4f}s")
    print(f"Total Time            : {total_training_time:.4f}s")
    print(f"Model Load Time       : {model_load_time:.4f}s")
    print(f"Avg Single Latency    : {avg_pred_latency_ms:.2f}ms")
    print("=" * 60)

    return {
        "classifier": calibrated_clf,
        "base_classifier": clf,
        "temperature": opt_T,
        "accuracy": val_acc,
        "macro_f1": val_f1_macro,
        "report": report
    }


if __name__ == "__main__":
    train_multilingual_embedding_classifier()
