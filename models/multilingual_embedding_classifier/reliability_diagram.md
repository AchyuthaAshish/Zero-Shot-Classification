# Reliability Diagram: Multilingual MiniLM Defect Classifier

Evaluated using Stratified 5-Fold Cross-Validation on the 600-example training dataset.
*Note: The 93-case held-out benchmark is strictly excluded.*

## Summary Metrics

| Model Configuration | ECE | Brier Score | Log Loss | Accuracy | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Uncalibrated MiniLM | 0.2544 | 0.3133 | 0.7182 | 0.8383 | 0.8387 |
| Temperature Scaling (T=0.3592) | 0.0154 | 0.2147 | 0.4548 | 0.8383 | 0.8387 |
| Platt Scaling (CalibratedClassifierCV) | 0.2144 | 0.3049 | 0.6748 | 0.8250 | 0.8250 |

## Reliability Table: After Temperature Scaling

| Confidence Bin | Sample Count | Observed Accuracy | Mean Confidence | Calibration Gap |
| :--- | :---: | :---: | :---: | :---: |
| [0.0, 0.1] | 0 | — | — | — |
| [0.1, 0.2] | 0 | — | — | — |
| [0.2, 0.3] | 27 | 29.6% | 26.1% | 3.6% |
| [0.3, 0.4] | 36 | 30.6% | 34.9% | 4.3% |
| [0.4, 0.5] | 23 | 43.5% | 45.9% | 2.4% |
| [0.5, 0.6] | 36 | 55.6% | 53.9% | 1.6% |
| [0.6, 0.7] | 25 | 72.0% | 64.6% | 7.4% |
| [0.7, 0.8] | 26 | 88.5% | 75.8% | 12.6% |
| [0.8, 0.9] | 44 | 84.1% | 84.7% | 0.6% |
| [0.9, 1.0] | 383 | 98.2% | 98.2% | 0.1% |

## Reliability Table: Before Calibration (Uncalibrated MiniLM)

| Confidence Bin | Sample Count | Observed Accuracy | Mean Confidence | Calibration Gap |
| :--- | :---: | :---: | :---: | :---: |
| [0.0, 0.1] | 0 | — | — | — |
| [0.1, 0.2] | 41 | 26.8% | 18.3% | 8.5% |
| [0.2, 0.3] | 61 | 45.9% | 23.8% | 22.1% |
| [0.3, 0.4] | 53 | 73.6% | 36.4% | 37.2% |
| [0.4, 0.5] | 66 | 81.8% | 44.5% | 37.3% |
| [0.5, 0.6] | 74 | 93.2% | 55.3% | 38.0% |
| [0.6, 0.7] | 82 | 96.3% | 65.6% | 30.8% |
| [0.7, 0.8] | 86 | 100.0% | 75.7% | 24.3% |
| [0.8, 0.9] | 98 | 100.0% | 85.4% | 14.6% |
| [0.9, 1.0] | 39 | 100.0% | 92.9% | 7.1% |