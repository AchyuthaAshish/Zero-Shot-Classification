"""Diagnostic tool for inspecting Local ML classifier probability distributions.

Shows top-N predicted categories and confidence scores for any input description.
For development and offline analysis.
"""

import sys
from typing import Optional, List, Tuple
import numpy as np

# Ensure Windows terminal UTF-8 output
if hasattr(sys, "stdout") and sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ml.local_classifier import get_local_classifier


def inspect_probabilities(
    text: str,
    model_type: str = "multilingual_embedding",
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Returns sorted list of (category, probability) tuples for the input text.
    """
    classifier = get_local_classifier(model_type=model_type)
    pred_cat, conf, prob_map = classifier.predict_with_confidence(text)
    sorted_probs = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
    return sorted_probs[:top_n]


def print_diagnostic_report(text: str, top_n: int = 5) -> None:
    """Prints a formatted probability distribution report."""
    print("=" * 65)
    print("           LOCAL ML PROBABILITY DIAGNOSTIC REPORT")
    print("=" * 65)
    print(f"Input: \"{text}\"\n")

    for m_type, label in [
        ("multilingual_embedding", "Primary: Multilingual MiniLM + LogisticRegression"),
        ("tfidf_svm", "Baseline: TF-IDF + LinearSVC")
    ]:
        print(f"--- {label} ---")
        try:
            top_classes = inspect_probabilities(text, model_type=m_type, top_n=top_n)
            for rank, (cat, prob) in enumerate(top_classes, 1):
                pct = prob * 100
                bar = "█" * int(pct // 5)
                print(f"  {rank}. {cat:<24} : {pct:>5.1f}%  {bar}")
        except Exception as e:
            print(f"  Error inspecting model: {e}")
        print()
    print("=" * 65)


if __name__ == "__main__":
    test_input = sys.argv[1] if len(sys.argv) > 1 else "The conveyor motor is making a grinding noise."
    print_diagnostic_report(test_input)
