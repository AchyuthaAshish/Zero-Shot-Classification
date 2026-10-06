# PRD --- Zero-Shot Classification of New Defect Descriptions

**Project ID:** 24CC3006-P068\
**Project type:** Applied LLM\
**Implementation:** Python + Streamlit, developed through Antigravity
with agentic AI coding.

## 0. Authority

This document is the product source of truth. It is aligned to the
original PS:

-   Map free-text defect descriptions to a taxonomy without training.
-   Support taxonomy rollout across plants.

The PS identifies two bottlenecks:

1.  Zero-shot output may invent categories outside the taxonomy.
2.  Local-language descriptions may be handled poorly.

The MVP therefore remains **zero-shot and no-training**. It is not a
conventional supervised ML project and must not be converted into one
without explicit human approval.

## 1. Product Overview

The product accepts an unstructured industrial defect description and
maps it to exactly one approved defect category.

Core flow:

``` text
Free-text description
        ↓
Input/language analysis
        ↓
Semantic understanding
        ↓
Taxonomy-guided classification
        ↓
Zero-shot LLM
        ↓
Structured output
        ↓
Strict taxonomy validation
        ↓
Optional verification
        ↓
Final category + explanation
```

The application is a controlled classification system, not a chatbot.

## 2. Problem

Industrial workers may describe the same defect with different
terminology, sentence structures, local languages, or mixed-language
technical language.

Examples:

-   `Motor producing abnormal vibration`
-   `The motor is making a strange noise`
-   `Motor lo unusual sound vastundi`

The system must generalize to unseen wording without training a new
classifier.

The two primary engineering problems are:

### Taxonomy violation

The LLM may output labels such as `Motor Failure`, `Bearing Problem`, or
`Hardware Issue` even when they are not approved.

### Local-language handling

Descriptions may be written in English, a local language, or
code-switched language such as Telugu-English.

## 3. Approved Taxonomy

The final category MUST be exactly one of:

1.  `Mechanical Fault`
2.  `Electrical Fault`
3.  `Sensor Fault`
4.  `Temperature Fault`
5.  `Software Fault`
6.  `Power Supply Fault`
7.  `Communication Fault`
8.  `Unknown`

No other final category is permitted.

## 4. Taxonomy Knowledge Base

Each category should contain a definition, positive examples, relevant
signals, boundaries, confusing categories, and exclusion guidance.

### Mechanical Fault

Physical/mechanical abnormality: vibration, bearing noise/failure, gear
damage, shaft/misalignment issues, abnormal motor/mechanical noise.

### Electrical Fault

Electrical circuit, wiring, current, or electrical-component problems:
short circuit, damaged wiring, connection failure, abnormal electrical
behavior.

### Sensor Fault

The sensor or its measurement is incorrect: wrong readings, disconnected
sensor, calibration problem, failure to detect, stuck/invalid output.

### Temperature Fault

The physical temperature is abnormal: overheating, temperature outside
expected/safe range, thermal abnormality.

### Software Fault

Application, firmware, software, or program behavior: crashes, firmware
errors, bugs, program not responding.

### Power Supply Fault

The supply of electrical power is the primary issue: power not reaching
equipment, battery/source failure, supply or adapter problem.

### Communication Fault

Communication between devices/systems/networks is the primary issue:
Wi-Fi, Bluetooth, CAN, MQTT, or device communication failure.

### Unknown

Insufficient or ambiguous evidence. Examples include
`Something is wrong with the machine.` and
`Machine is behaving strangely.`

## 5. Goals

-   Zero-shot classification without training a new classifier.
-   Prevent invalid/hallucinated taxonomy labels.
-   Improve local-language and code-switched handling.
-   Generalize to unseen wording and paraphrases.
-   Use `Unknown` instead of forcing unsupported classifications.
-   Provide a concise evidence-based explanation.
-   Support consistent taxonomy use across plants.
-   Deliver a feasible hackathon MVP.

## 6. Non-goals

The MVP must not:

-   train a supervised classifier;
-   fine-tune an LLM;
-   require a labeled training dataset;
-   claim guaranteed zero semantic mistakes;
-   create new taxonomy categories;
-   perform physical root-cause diagnosis;
-   control industrial equipment;
-   automatically create maintenance work orders;
-   add voice, images, enterprise authentication, or IIoT integrations
    unless explicitly promoted to scope.

## 7. Product Principles

1.  **Zero-shot first:** no training/fine-tuning for the core
    classifier.
2.  **Taxonomy is authoritative:** the LLM cannot modify it.
3.  **Local language is first-class:** non-English/code-switched text
    must enter the same semantic pipeline.
4.  **Evidence over forced classification:** insufficient evidence →
    `Unknown`.
5.  **Application validation is authoritative:** the LLM proposes; the
    application validates.
6.  **Semantic understanding over keyword matching.**
7.  **No hidden scope expansion.**
8.  **Explain results without exposing private model chain-of-thought.**

## 8. User Journeys

### J1 --- Normal English

`Motor is producing abnormal vibration.` → `Mechanical Fault`

### J2 --- Unseen phrasing

`The motor is shaking badly during operation.` should be understood
semantically rather than by exact keyword matching.

### J3 --- Local language

`మోటార్ ఎక్కువగా వైబ్రేట్ అవుతోంది` enters the same classification pipeline.

### J4 --- Code switching

`Motor lo unusual sound vastundi.` should map to the standardized
taxonomy.

### J5 --- Taxonomy challenge

`The bearing has failed and the motor is making grinding noise.` should
produce an approved category such as `Mechanical Fault`, not
`Bearing Failure`.

### J6 --- Unknown

`Something is wrong with the machine.` should be `Unknown` when evidence
is insufficient.

## 9. Features

### MUST HAVE

**F1 --- Free-text input:** accept natural-language descriptions.

**F2 --- Language-aware processing:** preserve original text, handle
local/mixed language, avoid destructive preprocessing and mandatory
blind translation.

**F3 --- Taxonomy knowledge base:** centralized
definitions/examples/boundaries.

**F5 --- Zero-shot LLM classifier:** classify against the taxonomy
without training.

**F6 --- Structured output:** parse model responses into a controlled
schema.

**F7 --- Strict taxonomy validation:** invalid labels cannot become
final output.

**F9 --- Unknown:** supported as a normal safe outcome.

**F10 --- Reliability:** High/Medium/Low or another documented
system-level reliability indicator.

**F11 --- Explanation:** concise, evidence-based explanation.

### SHOULD HAVE

**F4 --- Semantic candidate retrieval:** optional pre-trained
embeddings/retrieval to support local language and unseen phrasing. This
is semantic support, not training.

**F8 --- Verification:** optional second-pass evidence check constrained
to the same taxonomy.

**F12 --- Classification history.**

**F13 --- Category statistics.**

### FUTURE

Voice reporting, maintenance-ticket generation, Industrial IoT
integration, enterprise plant deployment, multimodal evidence, and
taxonomy feedback tools.

## 10. Business Rules / Invariants

-   BR-001: Every completed classification has exactly one final
    category.
-   BR-002: Final category is always one of the eight approved values.
-   BR-003: LLM cannot create categories.
-   BR-004: `Unknown` is valid.
-   BR-005: Ambiguous evidence must not be forced into a specific fault.
-   BR-006: Original user text is preserved.
-   BR-007: Taxonomy has one authoritative source.
-   BR-008: Local-language input cannot be rejected solely because it is
    non-English.
-   BR-009: Application validation is authoritative over model output.
-   BR-010: Model/infrastructure failure is not disguised as `Unknown`.
-   BR-011: No supervised training/fine-tuning in MVP.
-   BR-012: Agents cannot silently change taxonomy or scope.

## 11. Success Metrics

Evaluation must directly test the official bottlenecks.

-   **Taxonomy compliance:** 100% of successful final outputs use
    approved labels.
-   **Local-language handling:** evaluate English, local-language and
    code-switched examples.
-   **Unseen phrasing:** evaluate descriptions whose exact wording is
    absent from taxonomy examples.
-   **Unknown handling:** ambiguous/vague inputs should not be
    force-classified.
-   **Reliability:** record disagreement/failure cases.

Do not invent accuracy percentages before testing.

## 12. Constraints

-   12-hour hackathon target.
-   No supervised training or fine-tuning.
-   Python + Streamlit preferred.
-   LLM API required unless a local model is explicitly approved.
-   External API limits may apply.
-   Beginner-friendly implementation.
-   Antigravity agents must follow this PRD and the SRS.

## 13. Risks and Mitigations

**Category invention:** structured output + application-level
enumeration validation.

**Local-language failure:** language-aware processing + semantic
representation + multilingual evaluation.

**Ambiguity:** Unknown + optional verification.

**Hallucinated explanation:** evidence-constrained explanation.

**API failure:** controlled error state and optional deterministic
development fixtures.

**Agent scope creep:** binding requirements and human approval gates.

## 14. MVP Definition of Done

-   Free-text input works.
-   Eight-category taxonomy is centralized.
-   Taxonomy definitions/examples are available.
-   Zero-shot classification works.
-   Local/code-switched cases are tested.
-   Structured output is validated.
-   Invalid labels cannot become final categories.
-   Unknown works.
-   Reliability and explanation are shown.
-   Model/API errors are controlled.
-   Core logic has tests independent of live LLM calls.
-   Taxonomy-invention, local-language, unseen-phrasing and Unknown
    tests are evaluated.
-   No supervised training/fine-tuning is introduced.

## 15. Agentic Implementation Contract

Before coding, an agent MUST read this PRD and the SRS, identify
requirement IDs, inspect the repository, inspect existing
architecture/code, and identify unresolved decisions.

During coding it MUST implement the smallest requirement-aligned change,
preserve zero-shot classification and the eight-category taxonomy,
validate model output, test changes, reuse existing code where
appropriate, and keep secrets out of source control.

Agents MUST NOT train a classifier, fine-tune the LLM, invent
categories, silently change requirements, fabricate metrics/test
results, expose hidden chain-of-thought, or add unrelated features.

## 16. Human Approval Decisions

The following must not be silently chosen by an agent:

-   LLM provider/model;
-   embedding model/provider;
-   semantic retrieval enabled/disabled;
-   verification enabled/disabled;
-   exact local-language scope;
-   reliability threshold;
-   persistence requirement;
-   deployment platform.

## 17. Final Product Definition

**A taxonomy-constrained, multilingual, zero-shot industrial defect
classification system that maps free-text defect descriptions to an
approved taxonomy without training a new classification model.**

The two primary engineering objectives are:

1.  Prevent zero-shot category invention.
2.  Improve local-language and mixed-language handling.
