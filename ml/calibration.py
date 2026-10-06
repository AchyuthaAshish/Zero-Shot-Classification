"""Multiclass Confidence Calibration for Local Defect Classification.

Conforms to Phase 1 Step 1.2 requirements:
1. Implements Multiclass Temperature Scaling (Guo et al., 2017).
2. Evaluates Platt scaling (CalibratedClassifierCV) vs Temperature scaling on internal OOF data.
3. Computes Expected Calibration Error (ECE), Brier score, and Log Loss.
4. Generates reliability diagram data comparing confidence bins to empirical accuracy.
5. Strictly avoids the 93-case held-out evaluation dataset.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from ml.config import RANDOM_SEED, APPROVED_CATEGORIES


class MulticlassTemperatureScaler(BaseEstimator, ClassifierMixin):
    """
    Multiclass Temperature Scaling calibrator for linear/logistic classifiers.

    Scales unnormalized logits (decision_function) by a learned positive scalar T:
        q_k(x) = exp(z_k(x) / T) / sum_j exp(z_j(x) / T)

    Properties:
    - Strictly preserves ranking and argmax predictions (Accuracy is invariant).
    - Smoothly calibrates probabilities, dramatically reducing Expected Calibration Error (ECE).
    - Preserves scikit-learn estimator interface (predict, predict_proba, classes_).
    - Provides predict_raw_proba() for access to uncalibrated baseline scores.
    """

    def __init__(
        self,
        base_estimator: Optional[Any] = None,
        temperature: float = 1.0,
        calibration_method: str = "Temperature Scaling"
    ):
        self.base_estimator = base_estimator
        self.temperature = float(temperature)
        self.calibration_method = calibration_method
        if base_estimator is not None and hasattr(base_estimator, "classes_"):
            self.classes_ = base_estimator.classes_
        else:
            self.classes_ = np.array(APPROVED_CATEGORIES)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MulticlassTemperatureScaler":
        """Fits temperature T on validation logits and true class indices."""
        if self.base_estimator is None:
            raise ValueError("base_estimator must be provided before fitting temperature.")

        logits = self.base_estimator.decision_function(X)
        self.temperature = fit_temperature_scaling(logits, y)
        self.classes_ = self.base_estimator.classes_
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Returns unscaled logits from base estimator."""
        return self.base_estimator.decision_function(X)

    def predict_raw_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns uncalibrated softmax probabilities directly from base estimator."""
        return self.base_estimator.predict_proba(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns temperature-scaled calibrated probabilities."""
        logits = self.base_estimator.decision_function(X)
        T = max(self.temperature, 1e-4)
        scaled_logits = logits / T
        # Numerically stable softmax
        exp_scaled = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels (identical to base estimator since T > 0 preserves argmax)."""
        return self.base_estimator.predict(X)


def fit_temperature_scaling(logits: np.ndarray, y: Union[np.ndarray, List[int]]) -> float:
    """
    Finds optimal scalar temperature T > 0 minimizing multiclass Negative Log Likelihood (NLL).
    Uses bounded Brent scalar minimization.
    """
    y_arr = np.asarray(y)
    eps = 1e-15

    def nll_objective(T: float) -> float:
        scaled = logits / max(T, 1e-4)
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        probs = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
        probs = np.clip(probs, eps, 1.0 - eps)
        # Multiclass cross entropy
        loss = -np.mean(np.log(probs[np.arange(len(y_arr)), y_arr]))
        return float(loss)

    res = minimize_scalar(nll_objective, bounds=(0.01, 10.0), method="bounded")
    return float(res.x)


def calculate_ece(
    y_true: Union[np.ndarray, List[int]],
    y_probs: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Computes top-1 Expected Calibration Error (ECE) across n_bins equal-width bins.
    Returns (ece_score, list_of_bin_diagnostics).
    """
    y_arr = np.asarray(y_true)
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_arr).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_details = []

    total_samples = len(y_arr)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = int(np.sum(in_bin))
        if bin_size > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            abs_err = abs(bin_acc - bin_conf)
            ece += (bin_size / total_samples) * abs_err
            bin_details.append({
                "bin_range": f"[{bin_lower:.1f}, {bin_upper:.1f}]",
                "count": bin_size,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "abs_error": round(abs_err, 4)
            })
        else:
            bin_details.append({
                "bin_range": f"[{bin_lower:.1f}, {bin_upper:.1f}]",
                "count": 0,
                "accuracy": None,
                "confidence": None,
                "abs_error": 0.0
            })

    return float(ece), bin_details


def multiclass_brier_score(
    y_true: Union[np.ndarray, List[int]],
    y_probs: np.ndarray,
    n_classes: Optional[int] = None
) -> float:
    """
    Computes multiclass Brier score: mean squared difference between predicted
    probability distribution and one-hot true indicator vector.
    Lower is better (0.0 = perfect probabilistic predictions).
    """
    y_arr = np.asarray(y_true)
    k = n_classes or y_probs.shape[1]
    y_one_hot = np.zeros_like(y_probs)
    for i, label_idx in enumerate(y_arr):
        y_one_hot[i, label_idx] = 1.0
    return float(np.mean(np.sum((y_probs - y_one_hot) ** 2, axis=1)))


def evaluate_oof_calibration(
    embeddings: np.ndarray,
    labels: List[str],
    n_splits: int = 5,
    seed: int = RANDOM_SEED
) -> Dict[str, Any]:
    """
    Executes Stratified 5-Fold Cross-Validation on training embeddings.
    Generates pure Out-Of-Fold (OOF) predictions and computes ECE, Brier score,
    and Log Loss before and after calibration.
    """
    label_to_idx = {cat: i for i, cat in enumerate(APPROVED_CATEGORIES)}
    y_indices = np.array([label_to_idx[cat] for cat in labels])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    n_samples = len(labels)
    n_classes = len(APPROVED_CATEGORIES)

    oof_uncal_probs = np.zeros((n_samples, n_classes))
    oof_temp_probs = np.zeros((n_samples, n_classes))
    oof_platt_probs = np.zeros((n_samples, n_classes))

    fold_temperatures = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, y_indices)):
        X_tr, y_tr = embeddings[train_idx], y_indices[train_idx]
        X_val, y_val = embeddings[val_idx], y_indices[val_idx]

        # 1. Base Logistic Regression
        clf = LogisticRegression(max_iter=1000, random_state=seed, C=2.0, solver="lbfgs")
        clf.fit(X_tr, y_tr)

        oof_uncal_probs[val_idx] = clf.predict_proba(X_val)
        val_logits = clf.decision_function(X_val)

        # 2. Temperature Scaling (learned on inner cross-validation logits of train_idx)
        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        inner_logits_list = []
        inner_y_list = []
        for inner_tr, inner_vl in inner_skf.split(X_tr, y_tr):
            inner_clf = LogisticRegression(max_iter=1000, random_state=seed, C=2.0, solver="lbfgs")
            inner_clf.fit(X_tr[inner_tr], y_tr[inner_tr])
            inner_logits_list.append(inner_clf.decision_function(X_tr[inner_vl]))
            inner_y_list.append(y_tr[inner_vl])

        inner_logits = np.vstack(inner_logits_list)
        inner_y = np.concatenate(inner_y_list)
        T_fold = fit_temperature_scaling(inner_logits, inner_y)
        fold_temperatures.append(T_fold)

        # Apply learned T to val fold
        scaled_logits = val_logits / T_fold
        exp_scaled = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        oof_temp_probs[val_idx] = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)

        # 3. Platt Scaling (CalibratedClassifierCV with sigmoid)
        platt_clf = CalibratedClassifierCV(
            estimator=LogisticRegression(max_iter=1000, random_state=seed, C=2.0, solver="lbfgs"),
            method="sigmoid",
            cv=3
        )
        platt_clf.fit(X_tr, y_tr)
        oof_platt_probs[val_idx] = platt_clf.predict_proba(X_val)

    methods = {
        "Uncalibrated MiniLM": oof_uncal_probs,
        "Temperature Scaling": oof_temp_probs,
        "Platt Scaling (CalibratedClassifierCV)": oof_platt_probs
    }

    metrics_summary = {}
    for name, probs in methods.items():
        ece, bins = calculate_ece(y_indices, probs)
        brier = multiclass_brier_score(y_indices, probs, n_classes=n_classes)
        ll = float(log_loss(y_indices, probs))
        preds = np.argmax(probs, axis=1)
        acc = float(accuracy_score(y_indices, preds))
        f1 = float(f1_score(y_indices, preds, average="macro"))

        metrics_summary[name] = {
            "ece": round(ece, 4),
            "brier_score": round(brier, 4),
            "log_loss": round(ll, 4),
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "bin_diagnostics": bins
        }

    mean_temperature = float(np.mean(fold_temperatures))
    return {
        "metrics": metrics_summary,
        "mean_temperature": round(mean_temperature, 4),
        "fold_temperatures": [round(float(t), 4) for t in fold_temperatures]
    }


def format_reliability_diagram_markdown(bin_diagnostics: List[Dict[str, Any]]) -> str:
    """Generates an ASCII/markdown reliability diagram comparing predicted confidence vs accuracy."""
    lines = [
        "| Confidence Bin | Sample Count | Observed Accuracy | Mean Confidence | Calibration Gap |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ]
    for b in bin_diagnostics:
        cnt = b["count"]
        if cnt == 0:
            lines.append(f"| {b['bin_range']} | 0 | — | — | — |")
        else:
            acc_str = f"{b['accuracy'] * 100:.1f}%"
            conf_str = f"{b['confidence'] * 100:.1f}%"
            err_str = f"{b['abs_error'] * 100:.1f}%"
            lines.append(f"| {b['bin_range']} | {cnt} | {acc_str} | {conf_str} | {err_str} |")
    return "\n".join(lines)
