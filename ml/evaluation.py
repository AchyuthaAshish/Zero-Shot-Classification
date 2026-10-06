"""Held-Out Benchmark Evaluation Pipeline for Local ML Models.

Evaluates:
A. TF-IDF + LinearSVC Baseline
B. Multilingual Sentence Transformer + LogisticRegression

Uses the 93-case held-out benchmark (tests/fixtures/evaluation_cases.json).
CONSUMES ZERO GEMINI API CALLS.
Computes Accuracy, Macro F1, Per-Category, Language breakdowns, and Latency.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

from ml.config import (
    EVALUATION_DATASET_PATH,
    EMBEDDING_MODEL_DIR,
    TFIDF_SVM_MODEL_DIR,
    APPROVED_CATEGORIES
)
from ml.embedding_model import encode_texts
from preprocessing.text_processor import get_text_processor


def load_evaluation_cases() -> List[Dict[str, Any]]:
    """Loads the 93-case held-out benchmark."""
    if not EVALUATION_DATASET_PATH.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {EVALUATION_DATASET_PATH}")

    with open(EVALUATION_DATASET_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    return cases


def evaluate_model_on_benchmark(
    model_name: str,
    predict_fn
) -> Dict[str, Any]:
    """
    Evaluates a prediction function on the 93 held-out test cases.
    Computes overall, per-category, language, and boundary metrics.
    """
    cases = load_evaluation_cases()
    text_proc = get_text_processor()

    y_true: List[str] = []
    y_pred: List[str] = []
    latencies: List[float] = []

    # Case metadata trackers
    case_results: List[Dict[str, Any]] = []

    for item in cases:
        cid = item["id"]
        raw_text = item["description"]
        expected = item["expected_category"]
        lang = item.get("language_form", "English")
        case_type = item.get("case_type", "direct")

        # Normalize text
        processed = text_proc.preprocess(raw_text)

        # Measure latency per single inference
        t0 = time.perf_counter()
        pred, conf = predict_fn(processed.normalized_text)
        lat = (time.perf_counter() - t0) * 1000  # ms
        latencies.append(lat)

        y_true.append(expected)
        y_pred.append(pred)

        case_results.append({
            "id": cid,
            "text": raw_text,
            "expected": expected,
            "predicted": pred,
            "confidence": conf,
            "language": lang,
            "case_type": case_type,
            "correct": (pred == expected),
            "latency_ms": lat
        })

    # Overall Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    avg_latency = float(np.mean(latencies))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=APPROVED_CATEGORIES)

    # Per-category metrics
    report = classification_report(
        y_true,
        y_pred,
        labels=APPROVED_CATEGORIES,
        zero_division=0,
        output_dict=True
    )

    # Language breakdown
    lang_breakdown: Dict[str, Dict[str, Any]] = {}
    for lang_group in ["English", "Telugu", "Telugu-English"]:
        group_cases = [c for c in case_results if c["language"] == lang_group]
        if group_cases:
            g_true = [c["expected"] for c in group_cases]
            g_pred = [c["predicted"] for c in group_cases]
            lang_breakdown[lang_group] = {
                "count": len(group_cases),
                "accuracy": round(float(accuracy_score(g_true, g_pred)), 4),
                "macro_f1": round(float(f1_score(g_true, g_pred, average="macro", zero_division=0)), 4),
                "correct": sum(1 for c in group_cases if c["correct"])
            }

    # Unknown detection metrics
    unknown_cases = [c for c in case_results if c["expected"] == "Unknown"]
    unknown_acc = (
        sum(1 for c in unknown_cases if c["correct"]) / len(unknown_cases)
        if unknown_cases else 0.0
    )

    # Boundary case metrics
    boundary_cases = [c for c in case_results if c["case_type"] == "boundary"]
    boundary_acc = (
        sum(1 for c in boundary_cases if c["correct"]) / len(boundary_cases)
        if boundary_cases else 0.0
    )

    return {
        "model_name": model_name,
        "total_cases": len(cases),
        "overall_accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "avg_latency_ms": round(float(avg_latency), 2),
        "unknown_accuracy": round(float(unknown_acc), 4),
        "unknown_count": len(unknown_cases),
        "boundary_accuracy": round(float(boundary_acc), 4),
        "boundary_count": len(boundary_cases),
        "language_breakdown": lang_breakdown,
        "per_category": {
            cat: {
                "precision": round(float(report[cat]["precision"]), 4),
                "recall": round(float(report[cat]["recall"]), 4),
                "f1": round(float(report[cat]["f1-score"]), 4),
                "support": int(report[cat]["support"])
            }
            for cat in APPROVED_CATEGORIES if cat in report
        },
        "confusion_matrix": cm.tolist(),
        "case_results": case_results
    }


def evaluate_both_local_models() -> Dict[str, Any]:
    """
    Evaluates both TF-IDF + LinearSVC and Multilingual MiniLM + LogisticRegression
    on the 93 held-out test cases. Consumes ZERO API calls.
    """
    print("=" * 70)
    print("   EVALUATING LOCAL MODELS ON 93-CASE HELD-OUT BENCHMARK")
    print("               (ZERO GEMINI API CALLS)")
    print("=" * 70)

    # --- 1. Evaluate Baseline: TF-IDF + LinearSVC ---
    svm_path = TFIDF_SVM_MODEL_DIR / "model.joblib"
    if not svm_path.exists():
        raise FileNotFoundError(f"Baseline model not found at {svm_path}. Train it first.")
    svm_pipeline = joblib.load(svm_path)

    def svm_predict(text: str) -> Tuple[str, float]:
        probs = svm_pipeline.predict_proba([text])[0]
        best_idx = int(np.argmax(probs))
        return str(svm_pipeline.classes_[best_idx]), float(probs[best_idx])

    print("\nEvaluating Model 1: TF-IDF + LinearSVC Baseline...")
    svm_results = evaluate_model_on_benchmark("TF-IDF + LinearSVC", svm_predict)

    # --- 2. Evaluate Primary: Multilingual MiniLM + LogisticRegression ---
    emb_path = EMBEDDING_MODEL_DIR / "classifier.joblib"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding classifier not found at {emb_path}. Train it first.")
    emb_clf = joblib.load(emb_path)
    if not hasattr(emb_clf, "multi_class"):
        emb_clf.multi_class = "auto"

    def emb_predict(text: str) -> Tuple[str, float]:
        emb = encode_texts([text], batch_size=1)
        probs = emb_clf.predict_proba(emb)[0]
        best_idx = int(np.argmax(probs))
        return str(emb_clf.classes_[best_idx]), float(probs[best_idx])

    print("Evaluating Model 2: Multilingual MiniLM + LogisticRegression...")
    emb_results = evaluate_model_on_benchmark("Multilingual MiniLM + LogisticRegression", emb_predict)

    # --- 3. Print Comparison Report ---
    print("\n" + "=" * 70)
    print("                    MODEL COMPARISON REPORT")
    print("=" * 70)
    print(f"{'Metric':<28} | {'TF-IDF + LinearSVC':<20} | {'Multilingual MiniLM':<20}")
    print("-" * 75)
    print(f"{'Overall Accuracy':<28} | {svm_results['overall_accuracy']*100:>18.2f}% | {emb_results['overall_accuracy']*100:>18.2f}%")
    print(f"{'Macro F1-Score':<28} | {svm_results['macro_f1']:>19.4f} | {emb_results['macro_f1']:>19.4f}")
    print(f"{'Macro Precision':<28} | {svm_results['macro_precision']:>19.4f} | {emb_results['macro_precision']:>19.4f}")
    print(f"{'Macro Recall':<28} | {svm_results['macro_recall']:>19.4f} | {emb_results['macro_recall']:>19.4f}")
    print(f"{'Average Latency (ms)':<28} | {svm_results['avg_latency_ms']:>17.2f}ms | {emb_results['avg_latency_ms']:>17.2f}ms")
    print(f"{'Unknown Class Accuracy':<28} | {svm_results['unknown_accuracy']*100:>18.2f}% | {emb_results['unknown_accuracy']*100:>18.2f}%")
    print(f"{'Boundary Cases Accuracy':<28} | {svm_results['boundary_accuracy']*100:>18.2f}% | {emb_results['boundary_accuracy']*100:>18.2f}%")

    print("\n" + "-" * 75)
    print("LANGUAGE BREAKDOWN (Accuracy):")
    print("-" * 75)
    for lang in ["English", "Telugu", "Telugu-English"]:
        svm_l = svm_results["language_breakdown"].get(lang, {}).get("accuracy", 0.0)
        emb_l = emb_results["language_breakdown"].get(lang, {}).get("accuracy", 0.0)
        cnt = svm_results["language_breakdown"].get(lang, {}).get("count", 0)
        print(f"  - {lang:<22} ({cnt:02d} cases) : TF-IDF={svm_l*100:>5.1f}%  |  MiniLM={emb_l*100:>5.1f}%")

    print("=" * 75)

    return {
        "svm_baseline": svm_results,
        "multilingual_minilm": emb_results
    }


if __name__ == "__main__":
    evaluate_both_local_models()
