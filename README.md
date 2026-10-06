# Industrial Defect Classifier

A multi-lingual industrial defect classification system supporting English, Telugu, and Telugu-English code-switched descriptions. Combines a **Free Local ML Classifier** with an optional **Gemini Zero-Shot Fallback** under strict deterministic taxonomy validation.

---

## Architecture Overview

```text
User Input
    ↓
Local ML Classifier (Multilingual Sentence Transformer + Logistic Regression)
    ↓
Confidence & Validity Check
    ↓
High confidence (>= 0.70)?
    ├── YES → Local Prediction (100% Free, Offline, Zero API Calls)
    │
    └── NO  → Gemini Zero-Shot Fallback
                 ↓
          Strict Taxonomy Validation
                 ↓
          Final Canonical Result
```

If Gemini is unavailable, rate-limited, quota-exhausted, or no API key is configured, the system gracefully falls back to local inference without crashing.

---

## Why Local ML Was Added

1. **Zero API Cost & Unlimited Usage**: Local inference runs completely offline on CPU without consuming Gemini API quota.
2. **High Speed & Low Latency**: Local predictions execute in ~2-15 ms compared to 1000-2500 ms for cloud API requests.
3. **Offline Reliability**: Enables complete demo and operational functionality in air-gapped industrial environments or during API outages.
4. **Hybrid Safety**: Gemini remains active as an intelligent fallback for edge cases, ambiguous inputs, or low-confidence predictions.

---

## Models Implemented

1. **Primary Model: Multilingual MiniLM + LogisticRegression**
   - **Encoder**: Frozen `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` generating 384-dimensional dense semantic embeddings.
   - **Classifier**: `scikit-learn` `LogisticRegression` with calibrated probability estimation across the 8 approved categories.
   - **Features**: Native support for English, Telugu script, and Telugu-English code-switched technical phrasing.
2. **Baseline Model: TF-IDF + LinearSVC**
   - Traditional text classification pipeline (`TfidfVectorizer(ngram_range=(1,2))` + `CalibratedClassifierCV(LinearSVC)`).

---

## Approved Taxonomy Categories (Strict 8)

1. `Mechanical Fault`
2. `Electrical Fault`
3. `Sensor Fault`
4. `Temperature Fault`
5. `Software Fault`
6. `Power Supply Fault`
7. `Communication Fault`
8. `Unknown`

---

## Quick Start

### 1. Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Training Local ML Models

**Generate training dataset & train Primary Multilingual Model:**
```powershell
.\.venv\Scripts\python.exe -m ml.train_embedding_classifier
```

**Train Baseline TF-IDF + LinearSVC:**
```powershell
.\.venv\Scripts\python.exe -m ml.train_tfidf_svm
```

Both models persist artifacts and metadata under:
- `models/multilingual_embedding_classifier/`
- `models/tfidf_svm/`

### 3. Running Offline Benchmark Evaluation

Evaluate both local models on the 93-case held-out benchmark without making any Gemini API calls:

```powershell
.\.venv\Scripts\python.exe -m ml.evaluation
```

### 4. Running the Streamlit App

**Run in HYBRID Mode (Default: Local ML first, Gemini fallback):**
```powershell
$env:CLASSIFICATION_MODE="HYBRID"
.\.venv\Scripts\streamlit.exe run app.py
```

**Run in 100% Free Offline LOCAL Mode (Zero API calls, no API key needed):**
```powershell
$env:CLASSIFICATION_MODE="LOCAL"
.\.venv\Scripts\streamlit.exe run app.py
```

**Run in Pure GEMINI Zero-Shot Mode:**
```powershell
$env:CLASSIFICATION_MODE="GEMINI"
.\.venv\Scripts\streamlit.exe run app.py
```

### 5. Running Automated Tests

Run the complete test suite (66 tests covering both Local ML and Zero-Shot Gemini architectures):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

---

## Configuration Options

Configure through `.env` or system environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `CLASSIFICATION_MODE` | `HYBRID` | Routing mode: `HYBRID`, `LOCAL`, or `GEMINI` |
| `LOCAL_CONFIDENCE_THRESHOLD` | `0.70` | Engineering threshold for local acceptance |
| `LLM_PROVIDER` | `gemini` | LLM provider (`gemini` or `fake`) |
| `LLM_MODEL` | `gemini-3.8-flash` | Gemini model name |
| `LLM_API_KEY` | *(None)* | Google Gemini API key (optional in LOCAL mode) |

---

## Local Inference via Python

```python
from classification.classifier import classify_defect

# Offline local inference (no API key required)
result = classify_defect("Motor lo unusual sound vastundi and shaft vibrate avtundi.", mode="local")
print(result)

# Output:
# Category     : Mechanical Fault
# Language     : Telugu-English (Code-Switched)
# Reliability  : High
# Status       : SUCCESS
# Model Source : local_ml
# Confidence   : 0.88
```
