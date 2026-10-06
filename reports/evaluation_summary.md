# Executive Evaluation Summary: Industrial Defect Intelligence Platform

**Phase:** Phase 1 — AI Reliability & Evaluation (Step 1.7 Final Summary)  
**Date:** 2026-09-29  
**Platform Objective:** Multi-lingual industrial machine defect classification under real-world factory uncertainty, offline operation, and calibrated confidence guarantees.  

---

## 1. Evaluation Protocol

All empirical evaluations were conducted under strict zero-leakage conditions:
- **Protected Benchmark**: Held-out benchmark dataset of **93 test cases** ([`tests/fixtures/evaluation_cases.json`](file:///c:/Users/achyu/Documents/AIML-II%20Project/tests/fixtures/evaluation_cases.json)). The dataset was never used for model training, hyperparameter selection, calibration fitting, or decision threshold tuning.
- **Protected Training Set**: Balanced training dataset of **600 examples** ([`data/training/defect_training.csv`](file:///c:/Users/achyu/Documents/AIML-II%20Project/data/training/defect_training.csv)), 75 examples per approved class.
- **Taxonomy Enforcement**: Authoritative 8-category taxonomy (`Mechanical Fault`, `Electrical Fault`, `Sensor Fault`, `Temperature Fault`, `Software Fault`, `Power Supply Fault`, `Communication Fault`, `Unknown`).
- **Offline Protocol**: All quantitative benchmarks were executed 100% offline with zero external API calls.
- **Preprocessing Standard**: Identical character normalization and tokenization via [`TextProcessor`](file:///c:/Users/achyu/Documents/AIML-II%20Project/preprocessing/text_processor.py).

---

## 2. Overall Performance

Across the 93 held-out evaluation cases, the local models achieved:

| Metric | Primary (Multilingual MiniLM + LogReg) | Baseline (TF-IDF + LinearSVC) | Delta (Baseline - Primary) |
| :--- | :---: | :---: | :---: |
| **Total Test Cases** | 93 | 93 | 0 |
| **Total Errors** | 14 / 93 | 9 / 93 | -5 errors |
| **Overall Accuracy** | **84.95%** (79/93) | **90.32%** (84/93) | **+5.37%** |
| **Macro Precision** | 0.8821 | 0.9194 | +0.0373 |
| **Macro Recall** | 0.8467 | 0.9000 | +0.0533 |
| **Macro F1-Score** | **0.8545** | **0.9036** | **+0.0491** |
| **Taxonomy Compliance** | 100% (93/93) | 100% (93/93) | 0.00% |

*Note: These metrics characterize performance on the curated held-out benchmark and should not be construed as guaranteeing identical accuracy across arbitrary factory environments.*

---

## 3. Model Comparison & Architectural Trade-offs

| Performance & Operational Dimension | Model A: Multilingual MiniLM + LogReg | Model B: TF-IDF + LinearSVC | Cloud LLM (Gemini 3.8 Flash) |
| :--- | :--- | :--- | :--- |
| **Architecture** | Dense embeddings (384-d) + LogisticRegression | Sparse n-grams (1-2) + LinearSVC | 1M+ context frontier LLM |
| **Calibration Method** | Multiclass Temperature Scaling ($T=0.3362$) | Platt Scaling (`CalibratedClassifierCV`) | Qualitative linguistic confidence |
| **Expected Calibration Error (ECE)** | **0.0850** (Superior calibration) | 0.1766 (Coarser calibration) | N/A (No statistical probabilities) |
| **Multiclass Brier Score** | 0.2421 | **0.2081** | N/A |
| **Single-Inference Latency** | 9.35 ms $\pm$ 0.74 ms | **1.38 ms $\pm$ 0.17 ms** (~6.8x faster) | 400 - 1200 ms |
| **Cold-Start Load Time** | ~15.9 s (PyTorch/Transformers init) | **11.58 ms** (Instant joblib) | Instant (HTTP client init) |
| **Model Disk Footprint** | 37.3 KB (+ 457.5 MB transformer cache) | **837.2 KB** total | 0 MB (Remote API) |
| **RAM Requirement** | ~500 MB | **< 15 MB** | < 5 MB |
| **Offline Operation** | 100% Offline | 100% Offline | Requires Internet connection |
| **Deployment Role** | Semantic generalist with calibrated confidence | High-speed edge keyword classifier | Fallback oracle for low-confidence inputs |

---

## 4. Per-Category Performance Breakdown

Measured across the 8 authoritative taxonomy categories:

| Category | Support Cases | MiniLM Precision | MiniLM Recall | MiniLM F1 | TF-IDF Precision | TF-IDF Recall | TF-IDF F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mechanical Fault** | 14 | 0.8000 | 0.8571 | 0.8276 | 0.7778 | 1.0000 | 0.8750 |
| **Electrical Fault** | 12 | 0.8750 | 0.5833 | 0.7000 | 0.8462 | 0.9167 | 0.8800 |
| **Sensor Fault** | 12 | 1.0000 | 0.8333 | 0.9091 | 1.0000 | 0.7500 | 0.8571 |
| **Temperature Fault** | 10 | 0.9000 | 0.9000 | 0.9000 | 0.8889 | 0.8000 | 0.8421 |
| **Software Fault** | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Power Supply Fault**| 10 | 0.8000 | 0.8000 | 0.8000 | 1.0000 | 0.8000 | 0.8889 |
| **Communication Fault**| 10 | 1.0000 | 0.8000 | 0.8889 | 0.9091 | 1.0000 | 0.9524 |
| **Unknown** | 15 | 0.6818 | 1.0000 | 0.8108 | 0.9333 | 0.9333 | 0.9333 |

---

## 5. Language & Script Performance

The held-out benchmark evaluates three linguistic presentations:

| Language Form | Total Cases | MiniLM Accuracy | TF-IDF Accuracy | Delta (TF-IDF - MiniLM) |
| :--- | :---: | :---: | :---: | :---: |
| **English** | 70 | 88.57% (62/70) | 90.00% (63/70) | +1.43% |
| **Telugu-English Code-Switched** | 12 | **100.00%** (12/12) | **100.00%** (12/12) | 0.00% |
| **Native Telugu Script** | 11 | 45.45% (5/11) | **81.82%** (9/11) | **+36.37%** |

**Linguistic Insights**:
- **Code-Switched Robustness**: Transliterated Telugu descriptions containing English technical keywords (e.g. *'motor bearing lo sound ostundhi'*) yielded 100% accuracy across both models.
- **Native Telugu Script Divergence**: TF-IDF outperformed MiniLM on native Telugu script because character n-grams directly matched Telugu equipment stems present in the training set. MiniLM's frozen sentence embeddings mapped Telugu phrasing close to the decision boundary, where the Unknown detector intervened.

---

## 6. Confidence & Calibration

- **Calibration ECE**: MiniLM with Temperature Scaling achieved an Expected Calibration Error of **0.0850**, down 61.7% from raw logits (0.2220). Probabilities correlate directly with empirical accuracy across bins.
- **Confidence Separation**:
  - MiniLM: Mean confidence on correct cases = **0.9418**; mean confidence on error cases = **0.7867** ($\Delta \mu = +0.1551$).
  - TF-IDF: Mean confidence on correct cases = **0.7228**; mean confidence on error cases = **0.5789** ($\Delta \mu = +0.1439$).
- **Overconfidence Safeguard**: Overconfident errors ($\ge 0.85$ confidence on incorrect prediction) were limited to 2 cases for MiniLM and 0 cases for TF-IDF.
- **Distinction of Signals**:
  - `calibrated_prob`: Posterior probability scaled to match empirical empirical rates.
  - `raw_score`: Unscaled logit/softmax output preserved for auditability.
  - `top2_margin`: Top-1 probability minus Top-2 probability ($P_1 - P_2$).

---

## 7. Unknown & Ambiguity Detection

The platform enforces a strict conceptual and operational boundary between two failure modes:

| Dimension | Unknown Detection | Ambiguity Detection |
| :--- | :--- | :--- |
| **Core Definition** | **Insufficient or unrelated evidence** (no legitimate defect signals). | **Competing evidence** across 2+ valid categories. |
| **Example Input** | *"The weather is nice today"*, *"machine weird"* | *"Motor overheating and vibration causing breaker trip"* |
| **System Action** | Assigns `category="Unknown"`, `reliability="Uncertain"`. | Assigns top candidate, sets `is_ambiguous=True`, identifies competing category. |
| **Held-Out Benchmark Result** | **15/15 cases correctly recognized** (100% recall). | Triggered whenever margin $< 0.15$ or multi-signal conflict detected. |
| **Hybrid Mode Routing** | Bypasses external API to avoid wasteful spend. | Automatically routes to Cloud LLM for second opinion. |

---

## 8. Key Observations

1. **Complementary Strengths**: TF-IDF + LinearSVC provides sub-2ms lexical keyword matching with an 837 KB footprint, while MiniLM + LogisticRegression provides semantic flexibility on paraphrases and well-calibrated probabilities.
2. **High Agreement**: Both models agreed on **75 / 93 cases** (80.6%), with both models simultaneously classifying 73 cases correctly and only failing on 3 shared cases.
3. **Calibrated Routing**: The combination of temperature-scaled probabilities and top-2 margins enables the Hybrid engine to reliably flag low-confidence or contested cases without generating false alarms on clean single-category reports.

---

## 9. Known Limitations

1. **Sample Size Constraints**: The 93-case benchmark provides a rigorous multi-lingual testbed, but sub-cohorts (such as 11 native Telugu cases) have higher statistical variance.
2. **Lexical Overlap in Baseline**: TF-IDF's strong native Telugu script performance relies on character n-gram overlap with the 600 training examples. Unseen dialectal Telugu words may cause out-of-vocabulary degradation.
3. **Qualitative Cloud LLM**: Gemini 3.8 Flash provides strong reasoning over complex inputs, but its outputs are qualitative and cannot be evaluated on the same statistical probability calibration scale as local models.

---

## 10. Phase 1 Verification Summary

Phase 1 (AI Reliability & Evaluation) is completely implemented and verified:
- [x] Step 1.1: Confidence Representation Redesign (`ConfidenceAssessment`, margin, status separation)
- [x] Step 1.2: Confidence Calibration (Temperature Scaling, ECE = 0.0850, Brier score = 0.2421)
- [x] Step 1.3: Unknown Detection (Evidence verification layer, 15/15 benchmark unknowns recognized)
- [x] Step 1.4: Ambiguity Detection (Explainable competing evidence layer, margin & keyword signals)
- [x] Step 1.5: Systematic Error Analysis (`reports/error_analysis.json`, `reports/error_analysis.md`)
- [x] Step 1.6: Model Comparison (`reports/model_comparison.json`, `reports/model_comparison.md`)
- [x] Step 1.7: Basic Evaluation Dashboard & Summary (`ui/evaluation_view.py`, `reports/evaluation_summary.md`)

**Total Automated Tests:** 141/141 passing cleanly.
