"""Test runner for user-specified defect descriptions."""

import sys
from classification.classifier import classify_defect
from ml.diagnostics import inspect_probabilities

if hasattr(sys, "stdout") and sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

test_cases = [
    "The conveyor motor is making a grinding noise.",
    "The conveyor drive is producing a harsh rattling sound while running.",
    "The motor bearing produces a continuous grinding sound.",
    "The conveyor shaft appears to be misaligned.",
    "The temperature sensor is showing incorrect values.",
    "The PLC has lost communication with the controller."
]

print("=" * 75)
print("             USER-SPECIFIED INPUTS TEST REPORT (LOCAL MODE)")
print("=" * 75)

for text in test_cases:
    res = classify_defect(text, mode="local")
    probs = inspect_probabilities(text, model_type="multilingual_embedding", top_n=3)
    conf_pct = round(res.confidence * 100) if res.confidence is not None else 0

    print(f"\nInput: \"{text}\"")
    print("-" * 75)
    print(f"Predicted Category : {res.category}")
    print(f"Confidence         : {conf_pct}% ({res.confidence:.4f})")
    print(f"Reliability        : {res.reliability}")
    print(f"Model Source       : {res.model_source}")
    print("Top-3 Probabilities:")
    for rank, (cat, p) in enumerate(probs, 1):
        print(f"  {rank}. {cat:<24} — {p*100:>5.2f}%")
    print("Explanation:")
    print(f"  {res.reason}")

print("\n" + "=" * 75)
