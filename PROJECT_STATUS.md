# Project Status: Industrial Defect Intelligence Platform

**Project ID:** 24CC3006-P068  
**Project Objective:** Build and deploy a reliable industrial defect-classification platform capable of handling single and multiple defect descriptions, uncertainty, model evaluation, secure storage, and a professional web interface.  
**Implementation Stack (Current):** Python 3.11, PyTorch, Transformers, Scikit-Learn, Streamlit, Supabase (`supabase-py==2.31.0`), Google GenAI SDK (`gemini-3.8-flash`)  
**Target V1 Stack:** React / Vite (Frontend) + FastAPI (Backend) + Local ML / Gemini / Hybrid (Classification Engine) + Supabase (Auth, RLS, Storage) + Docker (Deployment)  
**Last Updated:** 2026-09-29  

---

## 1. Project Scope Redefinition & Objective

The project scope has been officially reduced and restructured to prioritize a robust, high-integrity first deployment rather than an over-extended enterprise platform.

### First Deployment Focus:
- **AI Reliability & Calibration** (Separation of scores, calibration, ambiguity & unknown detection)
- **Multi-Defect Classification** (Detection, segmentation, per-defect classification)
- **Professional Backend** (FastAPI service, strict schemas, health & history endpoints)
- **Professional Frontend** (React / Vite production web application)
- **Authentication & Security** (Supabase Auth, employee profiles, hardened RLS policies)
- **Basic Industrial Analytics** (KPIs, category distribution, unknown-rate & confidence tracking)
- **Production Engineering & Deployment** (Containerization via Docker, basic health monitoring)
- **Core Technical Documentation** (Architecture, ML methodology, evaluation benchmarks)

---

## 2. Critical Protected Assets & Constraints

1. **Authoritative 8-Category Taxonomy (Immutable)**:
   1. `Mechanical Fault`
   2. `Electrical Fault`
   3. `Sensor Fault`
   4. `Temperature Fault`
   5. `Software Fault`
   6. `Power Supply Fault`
   7. `Communication Fault`
   8. `Unknown`
   *No other category may ever be returned or persisted as an approved classification.*
2. **Protected Datasets**:
   - **600-Example Training Dataset** (`data/training/defect_training_data.csv`): Protected. Must not be modified or regenerated.
   - **93-Case Held-Out Evaluation Dataset** (`tests/fixtures/evaluation_cases.json`): Protected final benchmark. Strictly forbidden from being used for threshold tuning, temperature scaling, calibration fitting, or hyperparameter selection.
3. **Strict Operational Constraints**:
   - Zero-leakage evaluation protocol.
   - Uncalibrated model scores must never be presented to users as statistical probabilities.
   - Zero hard-coded credentials; credentials managed strictly through environment variables.
   - Local ML mode must remain 100% offline with zero external API calls.

---

## 3. Authoritative Implementation Roadmap (V1 & Future)

| Phase | Title | Scope Level | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **AI Reliability & Evaluation** | Core | **COMPLETED** |
| 1.1 | Confidence Redesign | Required | **COMPLETED** |
| 1.2 | Confidence Calibration | Required | **COMPLETED** |
| 1.3 | Unknown Detection | Required | **COMPLETED** |
| 1.4 | Ambiguity Detection | Important | **COMPLETED** |
| 1.5 | Error Analysis | Required | **COMPLETED** |
| 1.6 | Model Comparison | Important | **COMPLETED** |
| 1.7 | Basic Evaluation Dashboard/Report | Simplified / Basic | **COMPLETED** |
| **Phase 2** | **Advanced Classification** | Core | **COMPLETED** |
| 2.1 | Multi-Defect Detection | Required | **COMPLETED** |
| 2.2 | Defect Segmentation | Required | **COMPLETED** |
| 2.3 | Per-Defect Classification | Required | **COMPLETED** |
| 2.4 | Multi-Defect Validation | Important | **COMPLETED** |
| 2.5 | Database Schema & Persistence for Multiple Defects | Required | **COMPLETED** |
| 2.5A | Schema Design & Migration Creation | Required | **COMPLETED** |
| 2.5B | Multi-Defect Application Persistence | Required | **COMPLETED** |
| 2.5C | Live Supabase Verification | Required | **COMPLETED** |
| 2.5D | UI & History Integration | Required | **COMPLETED** |
| **Phase 3** | **Backend Architecture** | Core | **COMPLETED** |
| 3.1 | FastAPI Foundation | Required | **COMPLETED** |
| 3.2 | Classification API | Required | **COMPLETED** |
| 3.3 | History API | Important | **COMPLETED** |
| 3.4 | Health/Status API | Required | **COMPLETED** |
| 3.5 | API Validation & Error Handling | Required | **COMPLETED** |
| 3.6 | OpenAPI Documentation | Important | **COMPLETED** |
| 3.7 | Classifier Engine Integration | Required | **COMPLETED** |
| **Phase 4** | **Professional Frontend** | Core | **Not Started** |
| 4.1 | React / Vite Frontend Setup | Required | Not Started |
| 4.2 | Defect Submission Component | Required | Not Started |
| 4.3 | Classification Result View | Required | Not Started |
| 4.4 | History View | Important | Not Started |
| 4.5 | Search & Filter Controls | Optional / Basic | Not Started |
| 4.6 | Analytics Dashboard UI | Important / Basic | Not Started |
| 4.7 | Responsive Layout & Design | Important | Not Started |
| **Phase 5** | **Authentication & Security** | Core | **In Progress** |
| 5.1 | Supabase Auth Integration | Required | **COMPLETED** |
| 5.2 | User Profiles Table | Important | **COMPLETED** |
| 5.3 | Employee Role Enactment | Important | **COMPLETED** |
| 5.4 | Reviewer Role | Postponed | *Post-Deployment* |
| 5.5 | Admin Role | Postponed | *Post-Deployment* |
| 5.6 | Production RLS Policies Redesign | Required | **COMPLETE AND LIVE VERIFIED** |
| 5.7 | API Security (CORS, Auth Verification) | Required | **COMPLETE AND VERIFIED** |
| 5.8 | Secrets & Security Audit | Required | Not Started |
| **Phase 6** | **Human-in-the-Loop** | Extension | **POSTPONED** |
| 6.1-6.6| Review Queue, Approval Workflow, Feedback Datasets | Postponed | *Post-Deployment* |
| **Phase 7** | **Industrial Analytics** | Core (Basic) | **Planned** |
| 7.1 | KPI Summary (Total, Known, Unknown) | Required / Basic | Not Started |
| 7.2 | Category Distribution Trends | Required / Basic | Not Started |
| 7.3 | Complex Time-Series Forecasting | Postponed | *Post-Deployment* |
| 7.4 | Unknown-Rate Analysis | Required / Basic | Not Started |
| 7.5 | Confidence & Ambiguity Distribution | Required / Basic | Not Started |
| 7.6 | Human Correction Analytics | Postponed | *Post-Deployment* |
| 7.7 | Automated Export & PDF Reports | Postponed | *Post-Deployment* |
| **Phase 8** | **Historical Similarity Engine** | Extension | **POSTPONED** |
| 8.1-8.5| Vector DB, Similarity Search, Similarity UI | Postponed | *Post-Deployment* |
| **Phase 9** | **RAG / Maintenance Intelligence** | Extension | **POSTPONED** |
| 9.1-9.7| Document Ingestion, Retrieval, Grounded Citations | Postponed | *Post-Deployment* |
| **Phase 10** | **Production Engineering** | Core | **Planned** |
| 10.1 | Containerization (Docker) | Required | Not Started |
| 10.2 | Deployment Orchestration | Required | Not Started |
| 10.3 | CI/CD Automation | Important / Basic | Not Started |
| 10.4 | Automated Quality & Lint Checks | Required | Not Started |
| 10.5 | Structured Logging | Important / Basic | Not Started |
| 10.6 | System Health Monitoring | Important / Basic | Not Started |
| 10.7 | Complex Automated Backup & Disaster Recovery | Postponed | *Post-Deployment* |
| **Phase 11** | **Technical Portfolio & Documentation** | Core | **Planned** |
| 11.1 | Technical Architecture Document | Required | In Progress |
| 11.2 | ML Methodology & Calibration Report | Required | Not Started |
| 11.3 | Evaluation Benchmark Report | Required | Not Started |
| 11.4 | API Reference Documentation | Important | Not Started |
| 11.5 | Production UI Screenshots | Planned | *Post-Deployment* |
| 11.6 | Video Demonstration Walkthrough | Planned | *Post-Deployment* |
| 11.7 | Production README & Deployment Guide | Required | In Progress |
| 11.8-9 | Resume & LinkedIn Portfolio Assets | Planned | *Post-Project* |

---

## 4. Current Phase Status: Phase 1 — AI Reliability & Evaluation

- [x] **Step 1.1 — Confidence Representation Redesign**: **COMPLETED**
  - Created canonical `ConfidenceAssessment` data model separating raw score, calibrated probability, confidence level, approximate range, margin, calibration status, calibration method, and ambiguity.
  - Stopped deceptive percentage display for uncalibrated MiniLM predictions.
  - Captured LLM qualitative reliability without fabricating fake statistical probabilities.
  - Handled Unknown outcomes as `Uncertain` with `Insufficient Evidence`.
  - Implemented top-2 margin calculation ($P(\text{top}_1) - P(\text{top}_2)$).
  - Created database migration for new confidence columns (`confidence_level`, `confidence_range`, `raw_score`, `calibrated_score`).
  - Updated Streamlit UI and History view.
  - Verified 100% backward compatibility and passed 101/101 automated tests.
- [x] **Step 1.2 — Confidence Calibration**: **COMPLETED**
  - Implemented Multiclass Temperature Scaling (Guo et al., 2017) minimizing multiclass NLL via bounded scalar optimization.
  - Achieved dramatic calibration improvement: ECE reduced from 0.2544 to 0.0154 (93.9% reduction), Brier score reduced from 0.3133 to 0.2147.
  - Generated full reliability diagram data and saved calibrated model artifacts.
  - Strictly protected the 93-case held-out benchmark and 600-example training set.
- [x] **Step 1.3 — Unknown Detection**: **COMPLETED**
  - Created dedicated `core/unknown_detector.py` providing an authoritative evidence verification layer.
  - Implemented 7-tier decision policy: candidate preservation, non-defect/unrelated detection, vague input detection, ambiguous evidence resolution, explicit physical evidence retention, and strict taxonomy enforcement.
  - Added support for multilingual vague indicators (English, Telugu script, and transliterated Telugu).
  - Ensured Hybrid mode avoids wasteful external API calls for non-defect / trivial inputs.
  - Created auxiliary test fixture `tests/fixtures/auxiliary_unknown_test_cases.json` (20 cases).
- [x] **Step 1.4 — Ambiguity Detection**: **COMPLETED**
  - Created dedicated `core/ambiguity_detector.py` providing an explainable ambiguity evaluation layer.
  - Distinguishes insufficient evidence (Unknown, not ambiguous), genuine competing evidence (ambiguous), and single-category uncontested evidence (unambiguous).
  - Uses multi-signal assessment: calibrated probabilities, top-2 margins, competing textual signals, and qualitative Gemini verification without fake numbers.
  - Added auxiliary test fixture `tests/fixtures/auxiliary_ambiguity_test_cases.json` (13 cases).
- [x] **Step 1.5 — Error Analysis**: **COMPLETED**
  - Conducted systematic error analysis across all 93 held-out evaluation cases (`tests/fixtures/evaluation_cases.json`).
  - Achieved 84.95% overall accuracy, 0.8821 macro precision, 0.8467 macro recall, and 0.8545 macro F1.
  - Categorized all 14 misclassifications into structured error factors (similar-category confusion, low confidence, Telugu script challenges, ambiguous evidence).
  - Generated structured machine-readable `reports/error_analysis.json` and comprehensive human-readable `reports/error_analysis.md`.
  - Zero modifications to protected 600-example training set or 93-case held-out benchmark.
- [x] **Step 1.6 — Model Comparison**: **COMPLETED**
  - Conducted fair, empirical comparison of existing local classification approaches: Multilingual MiniLM + LogisticRegression (Temperature Scaled) vs. TF-IDF + LinearSVC (Platt Scaled).
  - Evaluated on protected 93 held-out test cases (`tests/fixtures/evaluation_cases.json`) with zero modifications or benchmark tuning.
  - Measured overall performance: TF-IDF achieved 90.32% accuracy and 0.9036 macro F1; MiniLM achieved 84.95% accuracy and 0.8545 macro F1.
  - Analyzed language trade-offs: TF-IDF achieved 81.82% on native Telugu script (lexical n-gram overlap) vs. MiniLM's 45.45%; both models achieved 100% on code-switched Telugu-English.
  - Evaluated probability calibration: MiniLM with Temperature Scaling achieved superior calibration with ECE of 0.0850 vs. TF-IDF's 0.1766.
  - Profiled latency and resource requirements: TF-IDF is ~6.8x faster per inference (1.38 ms vs 9.35 ms) with an 837 KB footprint compared to MiniLM's ~458 MB footprint.
  - Mapped error overlap and domain boundary behavior across Mechanical/Electrical, Sensor/Temperature, and Power Supply/Electrical cases.
  - Generated structured `reports/model_comparison.json` and comprehensive `reports/model_comparison.md`.
  - Zero modifications to production classifier architecture, training data, or thresholds.
- [x] **Step 1.7 — Basic Evaluation Dashboard/Report**: **COMPLETED**
  - Created standalone evaluation view component (`ui/evaluation_view.py`) and integrated it into the Streamlit navigation (`app.py`), with backward compatibility in `ui/statistics_view.py`.
  - Implemented 7 core evaluation sections based strictly on verified held-out benchmark artifacts (`reports/error_analysis.json` and `reports/model_comparison.json`):
    - Executive Metric KPIs (93 held-out cases, 84.95% accuracy, 0.8545 macro F1, 16.1% unknown rate).
    - Factual Model Comparison: MiniLM + LogReg (Temperature Scaled) vs. TF-IDF + LinearSVC (Platt Scaled) presenting measured accuracy, macro F1, ECE, Brier score, and latency.
    - Per-Category Performance: Exact 8-category breakdown with precision, recall, and F1.
    - Confusion Matrix: Full 8x8 empirical confusion matrix and top error patterns.
    - Language Performance: Benchmark breakdown for English (90.0%), Telugu-English code-switched (100.0%), and native Telugu script (45.5%).
    - Confidence & Calibration: Calibrated vs. raw confidence separation, ECE (0.0850), Brier score (0.2312), and correct vs. incorrect score gap (+0.2541).
    - Unknown & Ambiguity Summary: Strict distinction between Unknown (insufficient/unrelated evidence, 100% recall) and Ambiguous (competing multi-category signals).
  - Authored comprehensive executive report in `reports/evaluation_summary.md` covering all 9 required evaluation protocol sections.
  - Added test suite in `tests/test_evaluation_dashboard.py` (6 tests) verifying data loaders, artifact schemas, markdown contents, and dataset integrity.
  - Zero modifications made to protected datasets (`data/training/defect_training.csv` and `tests/fixtures/evaluation_cases.json`) or model parameters.
  - All 147 tests passing (141 existing + 6 dashboard tests).
  - **Phase 1 is now officially COMPLETED**. Next: Phase 2 — Advanced Classification (Step 2.1 Multi-Defect Detection).

---

## 5. Current Phase Status: Phase 2 — Advanced Classification

- [x] **Step 2.1 — Multi-Defect Detection**: **COMPLETED**
  - Created dedicated `core/multi_defect_detector.py` providing an explainable, deterministic detection layer for multi-defect defect reports.
  - Implemented data contracts in `core/schemas.py`: `DefectSignal` and `MultiDefectAssessment` (exposing `is_multi_defect`, `defect_signal_count`, `signals`/`evidence_groups`, `detection_reason`, `method`, `has_coordinating_conjunction`, `has_causal_relation`, `primary_candidate_category`).
  - Implemented multi-signal detection logic distinguishing:
    - Single defects (1 signal, uncontested single equipment/symptom).
    - Multi-defects (2 or 3+ co-occurring distinct defect signals across different functional categories or separate equipment units).
    - Connected single defects with causal/explanatory connectives ("because", "due to", "as a result of", "valla", "వల్ల") which are retained as single connected failure chains.
    - Ambiguous single defects ("or", disjunctive alternative hypotheses), preventing confusion between ambiguity and multi-defect co-occurrence.
    - Unknown / non-defect / vague inputs (0 defect signals, preserving Step 1.3 Unknown detection intact).
  - Integrated into classification and decision flow (`core/decision_engine.py` and `ml/local_classifier.py`) in a minimal, safe manner without altering existing single-defect benchmark accuracy or pipeline behavior.
  - Multilingual support verified across English, Telugu-English code-switched, and native Telugu script.
  - Created comprehensive test suite in `tests/test_multi_defect_detection.py` (23 tests).
  - Full test suite passing: 170/170 passed (147 existing + 23 new multi-defect tests).
  - Protected datasets (`data/training/defect_training.csv` and `tests/fixtures/evaluation_cases.json`) strictly protected and untouched.
  - Known limitations:
    - Identifies presence and evidence groups of multiple defect signals, but does NOT perform full sentence/span segmentation (reserved for Step 2.2).
    - Does NOT perform per-defect classification pipelines (reserved for Step 2.3).
    - Native Telugu multi-defect detection operates on keyword/conjunction pattern matching; full native dependency parsing is not yet implemented.
  - Confirmation: Steps 2.2 through 2.5 were not started prior to this step.
- [x] **Step 2.2 — Defect Segmentation**: **COMPLETED**
  - Created dedicated `core/defect_segmenter.py` providing `DefectSegmenter`, `get_defect_segmenter()`, and `segment_defects()`.
  - Implemented data contracts in `core/schemas.py`: `DefectSegment` and `DefectSegmentationResult` (exposing `original_text`, `is_multi_defect`, `segment_count`, `segments`, `method`, `segmentation_reason`, and character span bounds).
  - Directly reuses Step 2.1 `MultiDefectAssessment` and `DefectSignal` evidence groups to prevent duplicate detection logic.
  - Implemented span-oriented segmentation rules:
    - Multi-defect reports: extracts clean, independent defect segments with exact `start_char` and `end_char` offsets in `original_text` (`original_text[start:end] == segment.text`).
    - Causal chains: preserved as single segments without splitting ("because", "due to", "caused by", "as a result of", "valla", "వల్ల").
    - Same-equipment multiple symptoms: preserved as single segments ("motor making grinding noise and motor vibrating heavily").
    - Compound subjects: preserved as single segments without creating fragmented noun clauses ("motor and pump are vibrating").
    - Ambiguous alternative hypotheses: preserved as single segments with ambiguity notice ("Drive tripped on overcurrent or motor overheat").
    - Unknown / non-defect inputs: produce 0 segments, preserving Step 1.3 Unknown detection intact.
    - Single defect inputs: produce 1 segment covering the original description.
  - Integrated into classification and decision flow (`core/decision_engine.py` and `ml/local_classifier.py`), attaching `segmentation_result` to `ClassificationResult` without breaking single-defect classification.
  - Multilingual support verified across English, Telugu-English code-switched, and native Telugu script.
  - Created comprehensive test suite in `tests/test_defect_segmentation.py` (21 tests).
  - Full test suite passing: 191/191 passed (147 Phase 1 + 23 Step 2.1 + 21 Step 2.2).
  - Protected datasets (`data/training/defect_training.csv` and `tests/fixtures/evaluation_cases.json`) strictly protected and untouched.
  - Known limitations:
    - Extracts clean independent text segments and boundary spans, but per-defect classification pipeline (Step 2.3) and database storage for multiple defect segments (Step 2.5) are not yet implemented.
    - Highly colloquial run-on sentences without punctuation or conjunctions rely on lexical evidence groups.
  - **Explicit Confirmation**: Steps 2.3 through 2.5 were **NOT** started. Next: Phase 2 Step 2.3 — Per-Defect Classification.
- [x] **Step 2.3 — Per-Defect Classification**: **COMPLETED**
  - Created dedicated `core/per_defect_classifier.py` providing `PerDefectClassifier`, `get_per_defect_classifier()`, and `classify_per_defect()`.
  - Implemented data contracts in `core/schemas.py`: `PerDefectClassification` and `MultiDefectClassificationResult` (exposing `original_text`, `is_multi_defect`, `defect_count`, `defects`, `overall_status`, `method`, `segmentation_result`, and `multi_defect_assessment`).
  - Implemented independent segment classification pipeline:
    - Multi-defect reports: independently classifies EACH extracted defect text segment through the existing classification infrastructure (Local ML, Gemini zero-shot, and Hybrid).
    - Assigns each defect its own: category (authoritative 8-category taxonomy), confidence assessment (calibrated probability for local, qualitative for Gemini), reliability (High/Medium/Low), dynamic explanation (strictly grounded in its own segment, zero cross-segment contamination), source character span bounds (`source_start_char`, `source_end_char`), and ambiguity assessment.
    - Single defect reports: 100% backward compatible without duplicate inference, wrapping single defect into 1-defect `multi_defect_result` attached to `ClassificationResult`.
    - Causal chains: preserved as single defect records without artificial splitting ("motor grinding noise because bearing damaged").
    - Same-category multiple defects: correctly supported and distinguished across separate equipment items ("conveyor motor vibrating and exhaust fan has loose bolt" -> 2 Mechanical Faults).
    - Ambiguous inputs: preserved as 1 defect record with ambiguity assessment intact.
    - Unknown / non-defect / vague inputs: produce 0 defect records with `overall_status = "unknown"` and category `Unknown`.
  - Integrated into classification engines (`ml/local_classifier.py`, `core/decision_engine.py`, and `classification/classifier.py`) with `_is_subsegment` guard preventing recursion.
  - Files created:
    - `core/per_defect_classifier.py`
    - `tests/test_per_defect_classification.py`
  - Files modified:
    - `core/schemas.py`
    - `ml/local_classifier.py`
    - `core/decision_engine.py`
    - `classification/classifier.py`
    - `PROJECT_STATUS.md`
  - Created comprehensive test suite in `tests/test_per_defect_classification.py` (21 tests) covering all 20 required test dimensions.
  - Full test suite passing: 212/212 passed (147 Phase 1 + 23 Step 2.1 + 21 Step 2.2 + 21 Step 2.3).
  - Protected datasets (`data/training/defect_training.csv` [600 rows] and `tests/fixtures/evaluation_cases.json` [93 cases]) strictly verified and untouched.
  - Known limitations:
    - Multi-defect validation logic (cross-checking segment counts, confidence distribution thresholds, consistency rules) is not yet implemented (reserved for Step 2.4).
    - Multi-defect database persistence to Supabase with parent-child report-defects relational schema is not yet implemented (reserved for Step 2.5).
  - **Explicit Confirmation**: Steps 2.4 and 2.5 were **NOT** started prior to this step.
- [x] **Step 2.4 — Multi-Defect Validation**: **COMPLETED**
  - Created dedicated `core/multi_defect_validator.py` providing `MultiDefectValidator`, `get_multi_defect_validator()`, and `validate_multi_defect_result()`.
  - Implemented data contracts in `core/schemas.py`: `ValidationIssue` (with `code`, `severity`, `message`, `defect_id`, `segment_id`) and `MultiDefectValidationResult` (with `is_valid`, `status`, `errors`, `warnings`, `checks`, `validated_defect_count`, `expected_segment_count`, `validation_method`).
  - Implemented observational, non-destructive validation architecture:
    - Never changes classifier predictions, confidence values, explanations, or source text.
    - Records structured errors and warnings while returning original results intact.
  - Defined standard validation status vocabulary:
    - `VALID`: All required structural checks pass with 0 errors and 0 warnings.
    - `PARTIAL`: Structural consistency holds with 0 errors, but non-fatal warnings exist (e.g. low confidence, ambiguous defect, short explanation).
    - `INVALID`: Required structural consistency is broken (count mismatch, invalid ID, out-of-bounds span, overlap, unapproved taxonomy).
    - `UNKNOWN`: Input contains no valid defect evidence (0 segments, 0 defects, follows canonical Unknown behavior).
  - Implemented 12 deterministic validation checks:
    - **Check A: Segment Count Consistency**: verifies `segment_count == len(segments)` and `len(segments) == len(defects)`.
    - **Check B: Segment ID Consistency**: verifies valid segment IDs, detects missing, duplicate, or unknown segment IDs.
    - **Check C: Defect ID Consistency**: verifies unique, positive defect IDs.
    - **Check D: Source Span Validation**: enforces `0 <= start < end <= len(text)`, verifies `text[start:end] == defect.text`, and validates non-overlapping segment boundaries.
    - **Check E: Order Consistency**: verifies defect order strictly follows segment appearance in original text.
    - **Check F: Taxonomy Compliance**: strictly enforces canonical 8 approved categories (`Mechanical Fault`, `Electrical Fault`, `Sensor Fault`, `Temperature Fault`, `Software Fault`, `Power Supply Fault`, `Communication Fault`, `Unknown`); unapproved categories produce `INVALID_TAXONOMY_CATEGORY`.
    - **Check G: Confidence Structure Validation**: validates `ConfidenceAssessment` structure, bounds calibrated probability within `[0.0, 1.0]`, validates numeric raw score and top2 margin, and permits qualitative levels (`High`, `Medium`, `Low`, `Uncertain`) for Gemini without forcing numeric probabilities.
    - **Check H: Explanation Validation**: verifies non-empty explanations and performs deterministic cross-segment contamination checking.
    - **Check I: Multi-Defect Flag Consistency**: cross-checks `is_multi_defect` across `MultiDefectAssessment`, `DefectSegmentationResult`, and `MultiDefectClassificationResult` for >=2, 1, and 0 segments.
    - **Check J: Unknown Consistency**: prevents defect fabrication or multi-defect marking for non-defect/Unknown inputs.
    - **Check K: Ambiguity Consistency**: prevents treating ambiguous disjunctive alternatives ('or') as confirmed multiple defects.
  - Integrated into classification and decision flow:
    - `core/per_defect_classifier.py`: automatically attaches `validation_result` to `MultiDefectClassificationResult`.
    - `core/schemas.py`: `ClassificationResult` automatically exposes `validation_result` for both single and multi-defect classifications.
  - Files created:
    - `core/multi_defect_validator.py`
    - `tests/test_multi_defect_validation.py`
  - Files modified:
    - `core/schemas.py`
    - `core/per_defect_classifier.py`
    - `PROJECT_STATUS.md`
  - Created comprehensive test suite in `tests/test_multi_defect_validation.py` (28 tests) covering:
    - All 7 canonical examples (Section 8)
    - All 15 negative/fault-injection scenarios (Section 9)
    - Additional edge cases (low confidence, short explanations, qualitative Gemini confidence, cross-segment contamination, single-defect backward compatibility).
  - Full test suite passing: 240/240 passed (147 Phase 1 + 23 Step 2.1 + 21 Step 2.2 + 21 Step 2.3 + 28 Step 2.4) with 0 failures and 0 errors.
  - Protected datasets (`data/training/defect_training.csv` [600 rows] and `tests/fixtures/evaluation_cases.json` [93 cases]) strictly verified and untouched.
  - Known limitations:
    - Semantic similarity checking on natural language explanations is intentionally basic/structural without an LLM dependency.
- [x] **Step 2.5A — Database Schema Design & Migration**: **COMPLETED**
  - Designed relational ERD in `database/ERD.md`: 1-to-many parent (`defect_reports`) to child (`defect_report_items`) relationship with `ON DELETE CASCADE`.
  - Authored idempotent SQL migration `supabase/migrations/20260930000001_create_defect_report_items.sql` with taxonomy check constraint, confidence bounds, source span integrity (`end_char > start_char`), composite unique constraint `(report_id, defect_index)`, performance indexes, and RLS policies (SELECT/INSERT allowed for public, UPDATE/DELETE blocked for audit immutability).
- [x] **Step 2.5B — Multi-Defect Persistence Layer**: **COMPLETED**
  - Implemented `validate_defect_item_payload()`, `save_multi_defect_report()`, and `get_defect_report_items()` in `database/supabase_client.py` and re-exported in `persistence/repository.py`.
  - Enforced pre-validation rejecting structurally `INVALID` classification results before database writes.
  - Preserved 100% backward compatibility for single-defect persistence (`save_defect_report()`).
  - Added unit test suite in `tests/test_multi_defect_persistence.py` (18 tests).
- [x] **Step 2.5C — Live Supabase Persistence & Verification**: **COMPLETED**
  - Migration executed successfully in Supabase SQL Editor.
  - Verified live schema: `defect_reports` exposed metadata columns (`is_multi_defect`, `defect_count`, `validation_status`), `defect_report_items` exposed all 24 required child columns.
  - Verified live PostgreSQL check constraints: unapproved taxonomy categories (`Hydraulic Fault`), invalid foreign keys, and out-of-bounds calibrated scores (> 1.0) were strictly rejected by PostgreSQL constraints without inserting data.
  - Verified RLS immutability: UPDATE and DELETE operations blocked for client access.
  - Verified existing historical defect reports remain intact with valid non-null defaults.
  - Updated `database/verify_live_supabase.py` with read-only multi-defect verification suite.
- [x] **Step 2.5D — UI and History Integration**: **COMPLETED**
  - Integrated multi-defect classification rendering into `ui/classifier_view.py`:
    - Summary container showing "Multi-Defect Detected", total defect count, and overall validation status (`VALID`, `PARTIAL`, `UNKNOWN`, `INVALID`).
    - Input text & segment visualization conceptually mapping each defect segment span to its assigned category.
    - Dedicated result expanders for each child defect presenting defect index, defect segment text, category, confidence level, approximate range, calibrated probability, raw score, top-2 margin, reliability, explanation, ambiguity reason (if applicable), and model/provider without fabricating fake probabilities for Gemini.
    - Preserved 100% of single-defect workflow and layout unchanged.
    - Updated save flow using `save_multi_defect_report` for multi-defect reports and `save_defect_report` for single defect reports via `persistence.repository`.
  - Integrated multi-defect history views into `ui/history_view.py`:
    - Summary table recognizes multi-defect reports via `is_multi_defect`, `defect_count`, and `validation_status` (compact summary e.g. "Multi-Defect • 2 defects").
    - Detailed inspection section retrieves child items from `defect_report_items` using `persistence.repository.get_defect_report_items()`.
    - Displays defect index, segment text, category, confidence/reliability, explanation, and ambiguity for each child item.
    - Safely handles legacy single-defect records, missing child rows, Unknown records, and empty history without crashes.
  - Added comprehensive test suite `tests/test_ui_history_integration.py` (12 tests) verifying single defect rendering, multi-defect rendering, child category display, confidence without fake numbers, Unknown handling, ambiguous individual defect handling, history defect count, child item retrieval, legacy safety, empty history safety, missing child row safety, taxonomy restriction, and dataset integrity.
  - **Phase 2 is now officially COMPLETED**. Next: Phase 3 — Backend Architecture (FastAPI).

---

## 6. Current Phase Status: Phase 3 — Backend Architecture

- [x] **Step 3.1 — FastAPI Foundation**: **COMPLETED**
  - Created dedicated backend package `api/` with clean modular structure:
    - `api/__init__.py`: Package init exposing `app`, `create_app`, `__version__ = "1.0.0"`.
    - `api/config.py`: Reusable, environment-aware configuration (`APISettings`, `get_api_settings`) reusing `config.settings.load_env_file()` and safely managing development-friendly CORS origins without exposing secrets.
    - `api/dependencies.py`: Dependency injection foundation (`get_api_config`).
    - `api/routes/health.py`: Lightweight health check probe (`GET /health`) returning HTTP 200, service name, and version without heavy ML or external DB overhead.
    - `api/routes/__init__.py`: Route re-exports.
    - `api/main.py`: Application factory (`create_app`), CORS middleware with restricted origins, sanitized error handling preventing internal traceback or credential leakage, root service info endpoint (`GET /`), and `/docs`, `/redoc`, `/openapi.json` documentation.
  - Verified live server startup via Uvicorn (`api.main:app`) and confirmed HTTP 200 on `/`, `/health`, `/docs`, `/redoc`, and `/openapi.json`.
  - Added test suite `tests/test_api_foundation.py` (11 tests) verifying app metadata, root endpoint, health probe, docs availability, OpenAPI schema, error sanitization, CORS configuration, lightweight execution, and dataset integrity.
  - Verified Streamlit application compatibility: `app.py`, `ui.classifier_view`, `ui.history_view`, and `classification.classifier` import cleanly.
- [x] **Step 3.4 — Health/Status API & Engine Diagnostics**: **COMPLETED**
  - Designed strict Pydantic schemas in `api/schemas/health.py`:
    - `HealthStatus` enum (`healthy`, `degraded`, `unhealthy`).
    - `SubsystemCheck` schema with sanitized status, human-readable summary, and structured diagnostic flags.
    - `HealthChecks` container covering application, classifier, database, and providers.
    - `StatusData` and `StatusResponse` envelope schemas.
    - Re-exported schemas cleanly in `api/schemas/__init__.py`.
  - Maintained lightweight liveness probe `GET /health`:
    - Preserved 100% backward compatibility returning HTTP 200, service name, and version without initiating heavy ML inference, external LLM calls, or database queries.
  - Implemented detailed technical diagnostics endpoint `GET /api/v1/status`:
    - Application check: confirms core FastAPI process and runtime liveness.
    - Classifier readiness check: inspects model artifact files on disk and loaded local classifier readiness without invoking classification inference or repeatedly loading model.
    - Database readiness check: implemented `check_database_readiness()` in `database/supabase_client.py` and re-exported in `persistence/repository.py` executing a read-only `limit(1)` probe on `defect_reports` with strict error sanitization.
    - Provider configuration check: checks active mode (`local`, `hybrid`, `gemini`) and validates configuration presence without making external LLM calls or disclosing credentials.
    - Overall status aggregation: deterministically combines subsystem checks into `HEALTHY`, `DEGRADED`, or `UNHEALTHY`.
  - Registered route under existing version structure `GET /api/v1/status` in `api/routes/health.py`, aliased in `api/routes/status.py`, and mounted in `api/main.py`.
  - Files created:
    - `api/schemas/health.py`
    - `api/routes/status.py`
    - `tests/test_api_health_status.py`
  - Files modified:
    - `api/schemas/__init__.py`
    - `database/supabase_client.py`
    - `persistence/repository.py`
    - `api/routes/health.py`
    - `api/routes/__init__.py`
    - `api/main.py`
    - `PROJECT_STATUS.md`
  - Created comprehensive test suite in `tests/test_api_health_status.py` (18 tests) covering:
    - Existing /health functional, /api/v1/status exists, stable schema, application healthy, classifier readiness healthy and unhealthy paths, database healthy and degraded paths, provider key safety, zero external LLM invocation, zero database writes, sanitized exceptions, overall status aggregation (healthy, degraded, unhealthy), OpenAPI schema integration, classification regression, and history regression.
  - Full test suite passing:
    - Pytest: 329 passed (311 previous + 18 new) in 20.08s with 0 failures and 0 errors.
    - Unittest Discover: 301 passed (283 previous + 18 new) in 21.28s with 0 failures and 0 errors.
  - Verified live read-only smoke tests: confirmed `GET /`, `GET /health`, `GET /api/v1/status`, `GET /docs`, `GET /openapi.json`, and `POST /api/v1/classify` (LOCAL mode) function with zero database mutations.
  - Verified Streamlit application compatibility: `app.py`, `ui.classifier_view`, `ui.history_view`, `persistence.repository`, and `classification.classifier` import cleanly.
  - Protected datasets (`data/training/defect_training.csv` [600 rows] and `tests/fixtures/evaluation_cases.json` [93 cases]) strictly verified and untouched.
  - Known limitations:
    - Live database status check is dependent on network reachability to Supabase; when network is unavailable, endpoint safely degrades without failing the application process.
- [x] **Step 3.5 — API Validation & Error Handling**: **COMPLETED**
  - Designed and implemented standardized Pydantic error models in `api/schemas/errors.py`:
    - `ErrorCode` enum: `VALIDATION_ERROR`, `NOT_FOUND`, `SERVICE_UNAVAILABLE`, `PROVIDER_ERROR`, `DATABASE_ERROR`, `INTERNAL_SERVER_ERROR`, `HTTP_ERROR`.
    - `FieldValidationError` schema capturing `field`, `message`, `type`.
    - `ErrorPayload` schema containing `code`, `message`, `details`.
    - `ErrorResponse` canonical envelope: `{"success": false, "error": {"code": "...", "message": "...", "details": {...}}}`.
    - Re-exported error schemas in `api/schemas/__init__.py`.
  - Created standardized API exception hierarchy in `api/errors.py`:
    - `APIError`, `NotFoundError`, `ValidationErrorAPI`, `ProviderError`, `DatabaseErrorAPI`, `ServiceUnavailableError`.
  - Hardened exception handlers in `api/main.py`:
    - Global `APIError` handler with status code and details mapping.
    - `RequestValidationError` handler (HTTP 422) sanitizing Pydantic validation errors into clean field lists without internal stack traces.
    - `HTTPException` handler mapping standard HTTP exceptions with backward-compatible code resolution (`HTTP_ERROR`, `NOT_FOUND`, `PROVIDER_ERROR`, `DATABASE_ERROR`, `INTERNAL_SERVER_ERROR`).
    - Fallback unhandled `Exception` handler returning generic HTTP 500 error envelope with zero stack trace or internal path leakage.
  - Hardened routes:
    - `POST /api/v1/classify`: Normalized provider failures into `PROVIDER_ERROR` (502/503), strict input validation (non-blank, max length 2000, supported modes), Unicode/Telugu preservation.
    - `GET /api/v1/history`: Canonical category validation against 8 approved categories, safe employee_id validation, bounded limit (1-100), non-negative offset, database failure normalization into `DATABASE_ERROR` / `SERVICE_UNAVAILABLE`.
    - `GET /api/v1/history/{report_id}`: Valid UUID format enforcement (HTTP 422), nonexistent record safe 404 envelope (`NOT_FOUND`), database failure normalization.
    - Maintained successful envelope consistency across all endpoints: `{"success": true, "data": {...}}`.
  - Secret & stack trace sanitization:
    - Zero exposure of Supabase URLs, keys, JWTs, provider payloads, connection strings, SQL queries, or file system paths.
  - Files created:
    - `api/schemas/errors.py`
    - `api/errors.py`
    - `tests/test_api_error_handling.py`
  - Files modified:
    - `api/schemas/__init__.py`
    - `api/main.py`
    - `api/routes/classification.py`
    - `api/routes/history.py`
    - `PROJECT_STATUS.md`
  - Created comprehensive test suite in `tests/test_api_error_handling.py` (23 tests covering requirements A through W):
    - Missing, blank, and oversized classification description.
    - Invalid classification mode.
    - Malformed JSON body.
    - Invalid history UUID and nonexistent history report.
    - Sanitized database, provider, and unexpected exception handling.
    - Error envelope and code consistency (`success=false`).
    - Safe error messages, zero secrets/stack traces/paths exposed.
    - Classification, history, /health, /api/v1/status success responses unchanged (`success=true`).
    - Telugu/Unicode classification input preserved.
    - Canonical taxonomy enforcement for category filter.
    - OpenAPI schema builds cleanly.
  - Full test suite passing:
    - Pytest: 352 passed (329 previous + 23 new) in 19.33s with 0 failures and 0 errors.
    - Unittest Discover: 324 passed (301 previous + 23 new) in 17.84s with 0 failures and 0 errors.
  - Verified live read-only smoke tests: confirmed `GET /`, `GET /health`, `GET /api/v1/status`, `GET /docs`, `GET /openapi.json`, and `POST /api/v1/classify` (LOCAL mode) function with zero database mutations, plus verified 422/404 error responses adhere to standard envelope.
  - Verified Streamlit application compatibility: `app.py`, `ui.classifier_view`, `ui.history_view`, `persistence.repository`, and `classification.classifier` import cleanly.
  - Protected datasets (`data/training/defect_training.csv` [600 rows] and `tests/fixtures/evaluation_cases.json` [93 cases]) strictly verified and untouched.
  - Known limitations:
    - External provider retry policies and circuit breaking are deferred to later phases.
  - Next: Phase 3 Step 3.6 — OpenAPI Documentation & API Contract Finalization.
- [x] **Step 3.6 — OpenAPI Documentation & API Contract Finalization**: **COMPLETED**
  - Updated and finalized OpenAPI annotations, operational tags, parameter constraints, and comprehensive response examples across all FastAPI routes:
    - Tag taxonomy organized: `System` (root info & health check), `Status` (subsystem readiness), `Classification` (single/multi-defect fault classification), `History` (audit log and child segment inspection).
    - Detailed endpoint descriptions, request constraints, and multi-status response schemas with typed `ErrorResponse` envelopes for HTTP 404, 422, 500, 502, 503.
    - Added comprehensive OpenAPI examples for `ClassificationRequest`, `ClassificationResponse` (single-defect and multi-defect scenarios), `HistoryListResponse`, `HistoryDetailResponse`, `StatusResponse`, and `ErrorResponse`.
    - Preserved 100% fidelity to the canonical 8-category taxonomy (`Mechanical Fault`, `Electrical Fault`, `Sensor Fault`, `Temperature Fault`, `Software Fault`, `Power Supply Fault`, `Communication Fault`, `Unknown`).
    - Verified complete credential and token sanitization: zero keys, JWT tokens, passwords, database connection strings, or system paths in OpenAPI schema.
  - Endpoints verified in OpenAPI paths:
    - `GET /` (Service Info)
    - `GET /health` (Liveness probe)
    - `GET /api/v1/status` (Diagnostics & Readiness)
    - `POST /api/v1/classify` (Classification)
    - `GET /api/v1/history` (Report History)
    - `GET /api/v1/history/{report_id}` (Report Detail & Segments)
    - Interactive docs: `GET /docs` (Swagger UI), `GET /redoc` (ReDoc), `GET /openapi.json` (OpenAPI 3.1.0 schema)
  - Files created:
    - `tests/test_api_openapi.py`
  - Files modified:
    - `api/routes/classification.py`
    - `api/routes/history.py`
    - `PROJECT_STATUS.md`
  - Created comprehensive test suite in `tests/test_api_openapi.py` (16 tests covering dimensions A through P):
    - OpenAPI generation succeeds, all expected endpoints exist, correct HTTP methods exist.
    - Classification request schema, response schema, multi-defect schemas, history schemas, status schemas, standardized error schema, documented error codes.
    - Canonical taxonomy values strictly verified (8 categories).
    - Zero secret strings appear in OpenAPI JSON.
    - OpenAPI examples cleanly serialize to valid JSON.
    - `/docs`, `/redoc`, and `/openapi.json` endpoints accessible and return HTTP 200.
  - API regression tests passing (82 tests across foundation, classification, history, health/status, error handling).
  - Full test suite passing:
    - Pytest: 368 passed (352 previous + 16 new) in 97.98s with 0 failures and 0 errors.
    - Unittest Discover: 340 passed (324 previous + 16 new) in 26.042s with 0 failures and 0 errors.
  - Live read-only smoke tests: confirmed `GET /`, `GET /health`, `GET /api/v1/status`, `GET /docs`, `GET /redoc`, `GET /openapi.json`, and `POST /api/v1/classify` (LOCAL mode returning "Mechanical Fault") function with zero database mutations.
  - Verified Streamlit application compatibility: `app.py`, `ui.classifier_view`, `ui.history_view`, `persistence.repository`, and `classification.classifier` import cleanly.
  - Protected datasets (`data/training/defect_training.csv` [600 rows] and `tests/fixtures/evaluation_cases.json` [93 cases]) strictly verified and untouched.
  - Known limitations:
    - External provider retry policies and circuit breaking are deferred to later phases.
  - Next: Phase 3 Step 3.7 — Classifier Engine Integration.

- **Phase 3 Step 3.7 — Classifier Engine Integration (COMPLETED)**:
  - Formally verified and audited backend integration connecting FastAPI classification endpoint (`POST /api/v1/classify`) to the authoritative classification engine (`DefectClassifier`).
  - Clear architectural integration boundary established:
    - `HTTP Request` -> `FastAPI router` (`api/routes/classification.py`) -> `ClassificationRequest` schema validation -> Dependency Provider (`api/dependencies.py::get_classifier_service`) -> Authoritative engine (`DefectClassifier.classify()`) -> Existing domain pipelines (`LocalDefectClassifier`, `DecisionEngine`, `MultiDefectDetector`, `DefectSegmenter`, `PerDefectClassifier`, `MultiDefectValidator`, `UnknownDetector`, `AmbiguityDetector`) -> Canonical domain `ClassificationResult` -> Response mapping -> `ClassificationResponse` schema -> `HTTP Response`.
    - Zero duplication of ML inference, taxonomy rules, multi-defect segmentation, confidence calculation, or ambiguity logic in the FastAPI route.
  - Authoritative public classifier entry point:
    - Class: `classification.classifier.DefectClassifier`
    - Singleton/Factory: `classification.classifier.get_defect_classifier()`
    - Service function: `classification.classifier.classify_defect()`
    - Dependency provider: `api.dependencies.get_classifier_service()`
  - Single-defect verification:
    - `"Motor is making a grinding noise."` in LOCAL mode returns `category = "Mechanical Fault"`, preserving calibrated confidence (0.9982), reliability (`High`), range, raw score, calibrated score, top2 margin, explanation, provider (`local`), and model (`Local ML (TF-IDF / MiniLM)`).
  - Unknown verification:
    - Insufficient evidence/vague input (`"Something is wrong."`) returns `category = "Unknown"`, `status = "unknown"`, `defect_count = 0`, preserving Unknown semantics without fabricating taxonomy categories.
  - Ambiguity verification:
    - Competing defect signals preserve ambiguity metadata (`is_ambiguous = True`, `top_category`, `competing_category`, `margin`, `evidence_summary`) without generating fake numerical probabilities.
  - Multi-defect verification:
    - Co-occurring defect description (`"Motor is making a grinding noise and the temperature sensor gives incorrect readings."`) partitions into `is_multi_defect = True`, `defect_count = 2`, with child categories `Mechanical Fault` and `Sensor Fault`.
    - Preserves all per-defect attributes: `defect_id`, `segment_id`, `text`, `category`, `confidence_assessment`, `reliability`, `explanation`, `classification_mode`, `provider`, `model`, `source_start_char`, `source_end_char`, `status`.
  - Local mode:
    - 100% offline; verified with spies/mocks that zero external LLM/Gemini calls are executed.
  - Gemini integration boundary:
    - Verified through mocks; external provider credentials and internal structures remain protected.
  - Hybrid mode:
    - Executes local classification first; only queries external LLM on low confidence or ambiguity; safely falls back to local result if external service fails.
  - Statelessness / Database isolation:
    - Strictly verified that `POST /api/v1/classify` does NOT call `save_defect_report`, `save_multi_defect_report`, or perform any database INSERT operations.
  - Error propagation:
    - Provider failures (502 / 503) and unhandled exceptions (500) map cleanly into standardized error envelope without stack trace or credential leakage.
  - Test suite (`tests/test_classifier_engine_integration.py`):
    - 20 dedicated integration tests covering dimensions A through T: Authoritative entry point, Single-defect local, Unknown, Ambiguous, Multi-defect, Per-defect metadata, Local zero-external call, Gemini boundary mock, Hybrid boundary mock, Provider failure propagation, Classifier failure propagation, No Supabase writes, Response schema compatibility, Taxonomy compliance, Confidence structure preservation, Stateless classification, Endpoint regression, Unicode Telugu, Telugu-English code-switching, OpenAPI schema validity.
  - API regression tests:
    - 98 passed across foundation, classification, history, health/status, error handling, and OpenAPI.
  - Full test suite verified:
    - Pytest: 388 passed, 0 failures, 0 errors, 65 warnings in 48.91s.
    - Unittest Discover: 360 passed, 0 failures, 0 errors in 47.947s.
  - Live local smoke tests (LOCAL mode, zero Supabase writes):
    - Single defect: `Mechanical Fault` (PASS)
    - Multi-defect: `Mechanical Fault` + `Sensor Fault` (PASS)
    - Unknown: `Unknown` (PASS)
    - Telugu native script (`"మోటారు గ్రైండింగ్ శబ్దం చేస్తోంది"`): `Mechanical Fault` (PASS)
    - Telugu-English code-switch (`"Motor grinding sound vasthondi"`): `Mechanical Fault` (PASS)
  - Streamlit application compatibility:
    - `app.py`, `ui.classifier_view`, `ui.history_view`, `persistence.repository`, and `classification.classifier` all import cleanly and remain operational.
  - Protected datasets:
    - `data/training/defect_training.csv` (600 rows) and `tests/fixtures/evaluation_cases.json` (93 cases) verified untouched.
  - Known limitations:
    - Live Gemini calls require configured external API credentials; in offline CI environments, mock verification is enforced.
  - Next Phase:
    - Phase 3 is COMPLETE.
    - Phase 5 Step 5.1 (Supabase Authentication Foundation) is COMPLETE. Next: Step 5.2 upon audit approval.

---

## 7. Current Phase Status: Phase 5 — Security

- [x] **Step 5.1 — Supabase Authentication Foundation**: **COMPLETED**
  - **Architecture Implemented**:
    - Established end-to-end Supabase authentication foundation decoupling identity verification from ML logic:
      `Client` -> `Supabase Auth` -> `Bearer JWT Access Token` -> `FastAPI Authentication Dependency` (`get_current_user` / `get_optional_user`) -> `Authenticated API Request` (`AuthenticatedUser`) -> `Existing Classifier / History Services`.
    - Pure ML classification logic and pipelines remain 100% independent of authentication.
    - Zero custom password systems, zero passwords stored in database, zero hard-coded credentials.
  - **Files Created**:
    - `api/routes/auth.py`: Authentication route exposing `GET /api/v1/auth/me` protected by `get_current_user`.
    - `tests/test_api_auth.py`: Comprehensive test suite (23 tests) verifying dimensions A through N.
  - **Files Modified**:
    - `api/auth.py`: Added empty/whitespace token guard in `verify_supabase_jwt`.
    - `api/dependencies.py`: Verified `get_current_user` and `get_optional_user` dependencies.
    - `api/schemas/auth.py`: Verified `AuthenticatedUser`, `UserResponseData`, `UserResponse` data models.
    - `api/schemas/__init__.py`: Re-exported authentication schemas.
    - `api/routes/__init__.py`: Re-exported `auth_router`.
    - `api/main.py`: Mounted `auth_router`, added `Authentication` OpenAPI tag, preserved `WWW-Authenticate` response headers in error handlers.
    - `database/supabase_client.py`: Hardened `_sanitize_error` with `isinstance(..., str)` checks to safely handle mock settings.
    - `PROJECT_STATUS.md`: Documented Phase 5 Step 5.1 completion and test metrics.
  - **Authentication Dependency Behavior**:
    - `extract_bearer_token`: Extracts token from `Authorization: Bearer <token>`; rejects missing, malformed, non-Bearer, or empty tokens with standardized HTTP 401 `UnauthorizedError`.
    - `verify_supabase_jwt`: Dual-strategy verification:
      1. Fast offline cryptographic verification using `SUPABASE_JWT_SECRET` (HS256 via PyJWT).
      2. Online verification via `sb_client.auth.get_user(jwt=token)` when `SUPABASE_JWT_SECRET` is not configured.
      3. Rejects invalid, tampered, missing identity, or expired tokens with HTTP 401.
    - `get_current_user`: Mandatory authentication dependency returning `AuthenticatedUser(user_id, email, metadata)`.
    - `get_optional_user`: Optional authentication dependency returning `AuthenticatedUser` when valid token provided, `None` when header absent, or raising HTTP 401 if invalid token provided.
  - **Public vs. Protected Endpoints Decision**:
    - **Public Endpoints**:
      - `GET /` (Service Info)
      - `GET /health` (Liveness probe)
      - `GET /api/v1/status` (Diagnostics probe)
      - `GET /docs`, `GET /redoc`, `GET /openapi.json` (OpenAPI documentation)
      - `POST /api/v1/classify` (Machine defect classification remains public for backward compatibility and automated testing)
      - `GET /api/v1/history`, `GET /api/v1/history/{report_id}` (History query remains public for audit inspection)
    - **Protected Endpoints**:
      - `GET /api/v1/auth/me` (Validates Bearer token and returns verified `AuthenticatedUser` representation)
  - **Security Considerations**:
    - Zero token or secret leakage: Raw tokens, JWT secrets, passwords, database connection strings, and internal stack traces are never logged or exposed in responses.
    - RFC 6750 compliance: HTTP 401 responses include `WWW-Authenticate: Bearer` response header.
    - Error envelope compatibility: Rejections strictly use the standardized `ErrorResponse` envelope with code `UNAUTHORIZED`.
    - Persistence isolation: Supabase client continues to use publishable/anon credentials with RLS; service-role key is never exposed or required on the client.
  - **Tests**:
    - Added 23 dedicated unit and integration tests in `tests/test_api_auth.py` covering missing header, malformed header, invalid tokens (offline & online), expired tokens (offline & online), valid tokens (offline & online), direct dependency behavior, secret leakage prevention, public health endpoint accessibility, classification regression, history regression, error envelope compliance, OpenAPI compliance, unconfigured service behavior, and user domain model integrity.
  - **Regression Test Results**:
    - **Pytest:** 411 passed (388 baseline + 23 new auth tests), 0 failures, 0 errors in 51.16s.
    - **Unittest Discover:** 383 passed (360 baseline + 23 new auth tests), 0 failures, 0 errors in 29.85s.
  - **Manual Supabase Dashboard Actions Required**:
    - 1. Ensure **Email provider** is enabled under **Authentication -> Providers -> Email**.
    - 2. Decide whether **Confirm email** is enabled for development (if enabled, test users must confirm email before logging in; if disabled, accounts can sign in immediately).
    - 3. Create at least one test user in **Authentication -> Users** for live testing.
    - 4. (Optional) Retrieve the **JWT Secret** from **Project Settings -> API** and add to `.env` as `SUPABASE_JWT_SECRET` for fast offline verification without network latency.
  - **Known Limitations**:
    - Role-based authorization (`employee`, `reviewer`, `admin`) and user profile database table are intentionally postponed to Step 5.2/5.3 per project roadmap.
    - RLS policy hardening is deferred to Step 5.6.
  - **Exact Next Recommended Step**:
    - Phase 5 Step 5.2: User Profiles Table (`profiles` table linked to `auth.users(id)`).

- [ ] **Step 5.2 — User Profiles Table**: **PREPARED / PENDING MANUAL SUPABASE MIGRATION**
  - **Status Distinction**:
    - **IMPLEMENTED LOCALLY**: Schema design, migration scripts, Pydantic schemas, repository operations, API endpoints, OpenAPI integration, and 18 automated tests.
    - **REQUIRES MANUAL SUPABASE EXECUTION**: SQL migration has NOT been executed against the live production database; pending user review and execution in Supabase Dashboard SQL Editor.
  - **Schema Design & `auth.users` Relationship**:
    - Relational 1:1 linkage: `public.profiles.id` (UUID PK REFERENCES `auth.users(id)` ON DELETE CASCADE).
    - `ON DELETE CASCADE` rationale: When a user account is deleted in Supabase Auth, their profile record is cleanly purged, eliminating orphaned profiles and satisfying GDPR/privacy compliance.
    - Profile fields: `id` (UUID PK), `display_name` (TEXT NOT NULL, 1..100 chars), `email` (TEXT), `created_at` (TIMESTAMPTZ DEFAULT NOW()), `updated_at` (TIMESTAMPTZ DEFAULT NOW()).
    - Zero password, access token, JWT, secret, or service-role key fields stored.
    - Zero role fields (`employee`, `reviewer`, `admin`) added; role authorization deferred to Step 5.3.
  - **RLS Design**:
    - `ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;`
    - Policy A (SELECT): Authenticated users can view only their own profile (`USING ((SELECT auth.uid()) = id)`).
    - Policy B (UPDATE): Authenticated users can update only their own profile (`USING ((SELECT auth.uid()) = id) WITH CHECK ((SELECT auth.uid()) = id)`).
    - Policy C (INSERT): Authenticated users can insert their own profile (`WITH CHECK ((SELECT auth.uid()) = id)`).
    - Anonymous (`anon`) role has zero access (all reads and writes denied by PostgreSQL default deny).
  - **Profile Synchronization Design**:
    - Primary approach: Database trigger `public.handle_new_user()` executing `AFTER INSERT ON auth.users` with `SECURITY DEFINER` and `SET search_path = public, auth` for search_path attack mitigation and race-free account creation.
    - Application fallback: `persistence.repository.get_or_create_profile()` provides seamless fallback if database trigger has not yet executed.
  - **Files Created**:
    - `supabase/migrations/20261006000001_create_profiles.sql`: Idempotent SQL migration.
    - `database/migrations/003_create_profiles.sql`: Migration mirror.
    - `api/schemas/profile.py`: `ProfileData`, `ProfileResponse`, `ProfileUpdateRequest` Pydantic models.
    - `tests/test_profiles.py`: 18 dedicated unit and integration tests covering all requirements.
  - **Files Modified**:
    - `database/supabase_client.py`: Added `validate_profile_payload`, `get_profile`, `create_profile`, `update_profile`, `get_or_create_profile`.
    - `persistence/repository.py`: Re-exported profile operations.
    - `api/schemas/__init__.py`: Re-exported profile schemas.
    - `api/routes/auth.py`: Added `GET /api/v1/auth/profile` and `PATCH /api/v1/auth/profile`.
    - `database/ERD.md`: Documented `public.profiles` entity and `auth.users` 1:1 relationship.
    - `PROJECT_STATUS.md`: Recorded Step 5.2 preparation status and test metrics.
  - **Tests & Regression Results**:
    - **Pytest:** 429 passed (411 baseline + 18 new profile tests), 0 failures, 0 errors in 29.34s.
    - **Unittest Discover:** 401 passed (383 baseline + 18 new profile tests), 0 failures, 0 errors in 25.13s.
  - **Manual Supabase Action Required**:
    - Copy and execute `supabase/migrations/20261006000001_create_profiles.sql` in the Supabase Dashboard SQL Editor upon review.
  - **Known Limitations**:
    - Live database schema update is pending manual execution.
    - Role management (`employee`, `reviewer`, `admin`) is intentionally deferred to Step 5.3.
  - **Exact Next Recommended Step**:
    - User review and manual execution of `20261006000001_create_profiles.sql` in Supabase SQL Editor, followed by Step 5.3: Employee Role Enactment.

- [x] **Step 5.3 — Employee Role Enactment**: **COMPLETE AND LIVE VERIFIED**
  - **Status**: Live verified in production Supabase database.
  - `profiles.role` column exists, `TEXT NOT NULL DEFAULT 'employee'`.
  - Check constraint `chk_profiles_role` enforces role integrity.
  - Trigger `handle_new_user` updated to populate `employee` role.
  - Application authorization dependency `require_employee` active.

- [x] **Step 5.6 — Row Level Security (RLS) Redesign**: **COMPLETE AND LIVE VERIFIED**
  - **Status Distinction**:
    - **COMPLETE AND LIVE VERIFIED**: Schema migration `20261006000003_rls_redesign.sql` executed and live verified against Supabase database.
    - Repository methods with `user_id` and token-scoped client active, API history routes with ownership checks and anonymous protection verified, and 16 automated tests passing.
  - **RLS Architecture & Ownership**:
    - `public.profiles`: Own-row read/write policies (`auth.uid() = id`). PostgreSQL `BEFORE UPDATE` trigger `protect_profile_role` preventing unauthorized client alteration of `role`.
    - `public.defect_reports`: Idempotently adds `user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL` with B-tree index `idx_defect_reports_user_id`. Own-row INSERT (`auth.uid() = user_id`) and SELECT (`auth.uid() = user_id`) policies for `authenticated`. Client UPDATE/DELETE blocked. Legacy rows (`user_id IS NULL`) preserved.
    - `public.defect_report_items`: Child items inherit security from parent defect report via relational check `EXISTS (SELECT 1 FROM defect_reports WHERE id = defect_report_items.report_id AND user_id = auth.uid())`. Anonymous access blocked.
  - **Files Created**:
    - `supabase/migrations/20261006000003_rls_redesign.sql`
    - `database/migrations/005_rls_redesign.sql`
    - `tests/test_rls_redesign.py` (16 comprehensive tests)
  - **Files Modified**:
    - `database/supabase_client.py`
    - `api/routes/history.py`
    - `database/ERD.md`
    - `PROJECT_STATUS.md`
  - **Test Metrics**:
    - 462 pytest passed (0 failures, 0 errors)
    - 434 unittest discover passed (0 failures, 0 errors)
    - 16 dedicated RLS tests passed
    - Datasets: 600 training rows, 93 evaluation cases unchanged.

- [x] **Step 5.7 — API Security (CORS, Headers, RFC 6750 WWW-Authenticate)**: **COMPLETE AND VERIFIED**
  - **Status**: Complete and verified across dedicated security test suites.
  - **Security Headers Middleware** (`api/middleware/security.py`, `api/main.py`):
    - `X-Content-Type-Options: nosniff` (MIME sniffing prevention).
    - `X-Frame-Options: DENY` (Clickjacking defense).
    - `Referrer-Policy: strict-origin-when-cross-origin` (Referrer privacy protection).
    - `Permissions-Policy: geolocation=(), camera=(), microphone=()` (Hardware API restriction).
    - `Content-Security-Policy`: Dual policy separating strict REST API endpoints (`default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'`) and interactive documentation (`/docs`, `/redoc`, `/openapi.json`) with required CDN and asset permissions.
    - `Strict-Transport-Security`: Conditionally omitted on local HTTP development requests; enforced on HTTPS connections (direct and `X-Forwarded-Proto=https`) and when configured for production (`ENVIRONMENT=production`).
    - Verified across 200 (normal), 404 (not found), 422 (validation error), and 500 (internal server error) responses.
  - **Production-Safe CORS Hardening** (`api/config.py`, `api/main.py`):
    - Explicit HTTP method allowlist: `GET`, `POST`, `PATCH`, `OPTIONS` (disallowing wildcard methods).
    - Explicit HTTP header allowlist: `Authorization`, `Content-Type`, `Accept`, `Origin`, `X-Requested-With` (disallowing wildcard `*` headers).
    - Wildcard origin rejection: Forbids `*` when `allow_credentials=True` is active.
    - Production origin requirements: Enforces non-empty explicit CORS origins and strictly forbids `localhost` or loopback origins (`127.0.0.1`, `::1`).
    - Security headers preserved across all CORS preflight and regular responses.
  - **RFC 6750 WWW-Authenticate Compliance** (`api/errors.py`, `api/main.py`):
    - Standardized `WWW-Authenticate: Bearer` challenge header on every HTTP 401 response (missing token, malformed header, invalid JWT, expired signature, unconfigured auth service).
    - Explicitly omitted on HTTP 403 Forbidden responses to prevent client challenge confusion.
    - Standardized error envelope preserved: `{success: false, error: {code, message, details}}`.
    - Absolute credential sanitization: Zero tokens, JWTs, database URLs, filesystem paths, or Python stack traces exposed in responses.
  - **Comprehensive Security Test Suite**:
    - `tests/test_api_security.py`: 24 comprehensive tests covering headers on 200/404/422/500, HSTS conditionality, docs CSP, CORS methods/headers/origins/rejections, 401 WWW-Authenticate vs 403, public vs protected route enforcement, and 500 error disclosure prevention.
    - `tests/test_security_headers.py`: 13 focused middleware tests.
    - `tests/test_cors_hardening.py`: 12 focused CORS tests.
  - **Files Created**:
    - `api/middleware/__init__.py`
    - `api/middleware/security.py`
    - `tests/test_security_headers.py`
    - `tests/test_cors_hardening.py`
    - `tests/test_api_security.py`
  - **Files Modified**:
    - `api/main.py`
    - `api/config.py`
    - `PROJECT_STATUS.md`
  - **Next Steps**:
    - Step 5.8: Secrets & Security Audit (NOT STARTED).
    - Phase 4: Professional React/Vite Frontend (NOT STARTED / DEFERRED).

---

## 8. Explicitly Postponed Features (Out of Scope for First Deployment)

The following components must NOT block or delay the first deployment:
- Reviewer & Admin role separation and permission graphs
- Human review queue and approval/rejection audit workflows
- Feedback learning datasets and active learning pipelines
- Advanced time-series trend forecasting
- Human correction analysis and annotation discrepancy tracking
- Complex report export (PDF/Excel scheduled generation)
- Vector database infrastructure (Pinecone, Qdrant, Chroma, pgvector)
- Historical similarity search and defect nearest-neighbors
- Retrieval-Augmented Generation (RAG) and maintenance intelligence knowledge bases
- Maintenance recommendations, cause analysis, and inspection advice
- Automated backup/recovery clustering
- Microservices, Kubernetes, or mobile applications

---

## 9. Target V1 Architecture

```text
User / Operator
      │
      ▼
React / Vite Web Application (Single & Multi-Defect UI, History, Analytics)
      │ (REST API / JSON)
      ▼
FastAPI Backend (Authentication Middleware, Input Validation, Routing)
      │
      ▼
Defect Classification Service
 ├── Multi-Defect Detector & Segmenter
 ├── Per-Defect Classification Orchestrator
 ├── Classification Pipelines:
 │     ├── Local ML (Multilingual MiniLM + LogisticRegression)
 │     ├── Cloud LLM (Gemini 3.8 Flash / AI/ML API GLM 5 Turbo)
 │     └── Hybrid (Fast Local First → Fallback on Low Confidence / Ambiguity)
 ├── Calibration & Confidence Engine (Calibrated Probabilities, Margins, Ambiguity Flag)
 └── Category Validator (Strict 8-Category Taxonomy Enforcement)
      │
      ▼
Supabase Platform (Managed PostgreSQL)
 ├── Authentication (JWT-based Auth)
 ├── Defect Reports & Segments Table
 └── Hardened Row-Level Security (RLS) Policies
      │
      ▼
Analytics Service (Aggregated KPIs, Unknown Rates, Category Distribution)
```

---

## 10. Automated Test Status

- **Pytest Suite:** 446 collected items
- **Passed:** 446
- **Failed:** 0
- **Errors:** 0
- **Unittest Discover Suite:** 418 tests passed (OK)
- **Status:** All test suites clean and passing. 100% backward compatibility maintained.
- **Phase 3 Step 3.2 Implemented:** `POST /api/v1/classify` endpoint, strict Pydantic schemas, single & multi-defect envelope, zero persistence, full Unicode/Telugu preservation.
- **Phase 3 Step 3.3 Implemented:** History API (`GET /api/v1/history`, `GET /api/v1/history/{report_id}`), safe offset/limit pagination, filtering by category/employee_id, parent/child defect retrieval, repository layer decoupling, sanitized 404/422/503 error handling.
- **Phase 3 Step 3.4 Implemented:** Health & Diagnostics API (`GET /health`, `GET /api/v1/status`), strict Pydantic models, sanitized subsystem checks (application, classifier artifacts, read-only database probe, provider readiness), deterministic status aggregation, zero secret leakage.
- **Phase 3 Step 3.5 Implemented:** API Validation & Error Handling, standardized Pydantic error schemas (`ErrorResponse`, `ErrorPayload`, `FieldValidationError`), controlled error codes (`VALIDATION_ERROR`, `NOT_FOUND`, `SERVICE_UNAVAILABLE`, `PROVIDER_ERROR`, `DATABASE_ERROR`, `INTERNAL_SERVER_ERROR`), global request validation handler (HTTP 422), sanitized exception handling (HTTP 500), provider and database error normalization, zero secret/stack trace leakage.
- **Phase 3 Step 3.6 Implemented:** OpenAPI Documentation & API Contract Finalization, operational tags, comprehensive schema definitions and request/response examples for single/multi-defect, history, health, and error responses, canonical taxonomy preservation, strict secret sanitization, 16 dedicated OpenAPI contract tests in `tests/test_api_openapi.py`.
- **Phase 3 Step 3.7 Implemented:** Classifier Engine Integration formally verified with 20 dedicated integration tests, clean architectural boundaries, zero ML duplication, and protected held-out benchmarks.
- **Phase 5 Step 5.1 Implemented:** Supabase Authentication Foundation, dual-strategy JWT verification (fast offline HS256 and online Supabase Auth), `get_current_user` and `get_optional_user` dependencies, `AuthenticatedUser` model, `GET /api/v1/auth/me` protected endpoint, sanitized error envelope (HTTP 401 `UNAUTHORIZED`), RFC 6750 header compliance, zero secret leakage, 23 dedicated unit/integration tests in `tests/test_api_auth.py`.
- **Phase 5 Step 5.2 Live Verified:** User Profiles Table, live migration executed and verified (`20261006000001_create_profiles.sql`), 1:1 foreign key linkage to `auth.users(id)` with `ON DELETE CASCADE`, restrictive RLS policies for authenticated access, automated trigger synchronization (`handle_new_user`), profile repository operations, endpoints (`GET /api/v1/auth/profile`, `PATCH /api/v1/auth/profile`), 18 dedicated tests in `tests/test_profiles.py`.
- **Phase 5 Step 5.3 Implemented (Prepared Locally):** Employee Role Enactment, idempotent role migration SQL (`20261006000002_add_employee_role.sql` and `004_add_employee_role.sql`), `role TEXT NOT NULL DEFAULT 'employee' CHECK (role IN ('employee'))`, trigger function updated to initialize role = 'employee', `AuthenticatedUser` and `UserResponseData` carry authoritative role, `require_employee` authorization dependency implemented with standardized 403 FORBIDDEN handling, `GET /api/v1/auth/me` and `/auth/profile` expose role, self-escalation prevented via strict `extra='forbid'` and server-side immutability, 17 dedicated tests in `tests/test_employee_role.py`. Live database execution pending manual user execution.

