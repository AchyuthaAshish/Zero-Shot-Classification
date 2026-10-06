"""Reusable Honest Evaluation Runner for Zero-Shot Industrial Defect Classification.

Conforms to PRD Section 11 (Success Metrics), Section 14 (Definition of Done),
SRS Section 11 (Testing Strategy), and SRS Section 19.

Strict Requirements:
1. Accepts a classifier instance.
2. Runs approved cases in tests/fixtures/evaluation_cases.json.
3. Compares final category with expected_category.
4. Calculates category correctness, taxonomy compliance, Unknown behavior,
   and breakdown by language form and case type.
5. Writes NO fabricated results — computes strictly from actual run execution.
6. Clearly marks results as fake-client results or live-Gemini results.
7. Explicitly refuses to claim benchmark accuracy until:
   - the project owner has reviewed the dataset labels; and
   - the evaluation has run against the real Gemini API.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from classification.classifier import DefectClassifier, get_defect_classifier
from taxonomy.repository import get_taxonomy_repository, TaxonomyRepository
from llm.client import FakeLLMClient, GeminiLLMClient


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE_PATH = BASE_DIR / "tests" / "fixtures" / "evaluation_cases.json"


@dataclass
class EvaluationReport:
    """Structured evaluation report containing strictly computed, non-fabricated metrics."""
    execution_timestamp: str
    client_type: str
    is_fake_client: bool
    is_live_gemini: bool
    model_name: str
    total_cases: int
    correct_count: int
    accuracy: float
    taxonomy_compliance_count: int
    taxonomy_compliance_rate: float
    unknown_metrics: Dict[str, Any]
    language_form_results: Dict[str, Dict[str, Any]]
    case_type_results: Dict[str, Dict[str, Any]]
    category_results: Dict[str, Dict[str, Any]]
    dataset_review_status: str
    benchmark_status: str
    benchmark_accuracy_claimable: bool
    disclaimer: str
    case_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Exports the report as a clean serializable dictionary."""
        return asdict(self)

    def format_summary(self) -> str:
        """Formats an honest human-readable evaluation summary with prominent disclaimers."""
        mode_header = (
            "OFFLINE FAKE-CLIENT RUN (RUNNER MECHANICS VERIFICATION ONLY)"
            if self.is_fake_client else "LIVE GEMINI API EVALUATION RUN"
        )
        
        lines = [
            "=" * 78,
            f"EVALUATION BENCHMARK REPORT: {mode_header}",
            "=" * 78,
            f"Execution Timestamp : {self.execution_timestamp}",
            f"Client Type         : {self.client_type}",
            f"Model               : {self.model_name}",
            f"Dataset Review State: {self.dataset_review_status}",
            f"Benchmark Status    : {self.benchmark_status}",
            f"Benchmark Claimable : {'YES' if self.benchmark_accuracy_claimable else 'NO (Strict PRD/SRS Constraint)'}",
            "",
            ">>> IMPORTANT NOTICE / DISCLAIMER <<<",
            self.disclaimer,
            "",
            "-" * 78,
            "OVERALL METRICS:",
            "-" * 78,
            f"Total Evaluated Cases   : {self.total_cases}",
            f"Exact Category Matches  : {self.correct_count} / {self.total_cases}",
            f"Category Correctness    : {self.accuracy * 100:.2f}%",
            f"Taxonomy Compliance     : {self.taxonomy_compliance_rate * 100:.2f}% ({self.taxonomy_compliance_count}/{self.total_cases})",
            "",
            "-" * 78,
            "UNKNOWN BEHAVIOR METRICS:",
            "-" * 78,
            f"Expected Unknown Cases  : {self.unknown_metrics.get('expected_unknown_total', 0)}",
            f"Expected Unknown Correct: {self.unknown_metrics.get('expected_unknown_correct', 0)} ({self.unknown_metrics.get('expected_unknown_accuracy', 0.0) * 100:.2f}%)",
            f"Non-Unknown Fallbacks   : {self.unknown_metrics.get('non_unknown_predicted_as_unknown', 0)} / {self.unknown_metrics.get('non_unknown_total', 0)} ({self.unknown_metrics.get('false_unknown_rate', 0.0) * 100:.2f}%)",
            f"Total Predicted Unknown : {self.unknown_metrics.get('total_predicted_unknown', 0)}",
            "",
            "-" * 78,
            "RESULTS BY LANGUAGE FORM:",
            "-" * 78,
        ]

        for lang, res in sorted(self.language_form_results.items()):
            lines.append(
                f"  - {lang:<28}: {res['correct']:>2}/{res['total']:<2} "
                f"({res['accuracy'] * 100:>6.2f}%)"
            )

        lines.extend([
            "",
            "-" * 78,
            "RESULTS BY CASE TYPE:",
            "-" * 78,
        ])
        for ctype, res in sorted(self.case_type_results.items()):
            lines.append(
                f"  - {ctype:<28}: {res['correct']:>2}/{res['total']:<2} "
                f"({res['accuracy'] * 100:>6.2f}%)"
            )

        lines.extend([
            "",
            "-" * 78,
            "RESULTS BY CATEGORY (EXPECTED vs PREDICTED):",
            "-" * 78,
        ])
        for cat, res in sorted(self.category_results.items()):
            lines.append(
                f"  - {cat:<24}: Expected={res['expected_count']:>2}, "
                f"Predicted={res['predicted_count']:>2}, Correct={res['correct_count']:>2}"
            )

        lines.append("=" * 78)
        return "\n".join(lines)


class EvaluationRunner:
    """Reusable runner executing and scoring benchmark evaluation datasets."""

    def __init__(
        self,
        classifier: Optional[DefectClassifier] = None,
        fixture_path: Optional[Path] = None,
        taxonomy_repo: Optional[TaxonomyRepository] = None
    ):
        self.classifier = classifier or get_defect_classifier()
        self.fixture_path = fixture_path or DEFAULT_FIXTURE_PATH
        self.taxonomy_repo = taxonomy_repo or get_taxonomy_repository()
        self.approved_categories = set(self.taxonomy_repo.get_categories())

    def _detect_client_type(self) -> tuple[str, bool, bool]:
        """Detects whether the underlying classifier client is Fake or Live Gemini."""
        client = getattr(self.classifier, "llm_client", None)
        if isinstance(client, FakeLLMClient):
            return "fake_client", True, False
        if isinstance(client, GeminiLLMClient):
            return "live_gemini", False, True
        
        # Check by class name fallback
        cname = client.__class__.__name__ if client else ""
        if "Fake" in cname:
            return "fake_client", True, False
        if "Gemini" in cname:
            return "live_gemini", False, True
        return "custom_client", False, False

    def load_cases(self) -> List[Dict[str, Any]]:
        """Loads and returns benchmark cases from the designated fixture file."""
        if not self.fixture_path.exists():
            raise FileNotFoundError(f"Evaluation fixture not found at {self.fixture_path}")
        with open(self.fixture_path, "r", encoding="utf-8") as f:
            cases = json.load(f)
        if not isinstance(cases, list) or not cases:
            raise ValueError("Evaluation fixture must contain a non-empty list of test cases.")
        return cases

    def run(
        self,
        cases: Optional[List[Dict[str, Any]]] = None,
        limit: Optional[int] = None
    ) -> EvaluationReport:
        """
        Executes evaluation across benchmark cases and computes genuine, non-fabricated metrics.
        """
        eval_cases = cases if cases is not None else self.load_cases()
        if limit is not None and limit > 0:
            eval_cases = eval_cases[:limit]

        total_cases = len(eval_cases)
        if total_cases == 0:
            raise ValueError("No test cases provided for evaluation.")

        client_type, is_fake_client, is_live_gemini = self._detect_client_type()
        model_name = getattr(self.classifier.llm_client, "model_name", "fake-offline-rules")

        # Datasets are authored with review_status; check if any are pending
        all_pending = any(c.get("review_status") == "pending_human_review" for c in eval_cases)
        dataset_review_status = "pending_human_review" if all_pending else "approved"

        # Benchmarks can only be claimed if:
        # 1. Dataset labels have been reviewed and approved by project owner.
        # 2. Evaluation was run against real Gemini API (not Fake client).
        benchmark_accuracy_claimable = (not is_fake_client) and (dataset_review_status == "approved")

        if is_fake_client:
            benchmark_status = "mechanics_verification_only"
            disclaimer = (
                "MECHANICS VERIFICATION ONLY (OFFLINE FAKE CLIENT): This evaluation was executed "
                "using a deterministic FakeLLMClient. It validates evaluation runner mechanics, "
                "taxonomy enforcement, schema parsing, and data flow. It DOES NOT measure actual "
                "Gemini model accuracy or zero-shot classification quality. No benchmark accuracy "
                "claims can be made from this run."
            )
        elif dataset_review_status == "pending_human_review":
            benchmark_status = "provisional_unreviewed_dataset"
            disclaimer = (
                "LIVE GEMINI RUN (PROVISIONAL): This evaluation was executed against the Google Gemini API. "
                "However, benchmark accuracy claims remain PROVISIONAL until the project owner has "
                "formally reviewed and signed off on the evaluation dataset labels."
            )
        else:
            benchmark_status = "verified_live_gemini_benchmark"
            disclaimer = (
                "VERIFIED BENCHMARK: Run executed against the live Google Gemini API using "
                "human-reviewed and approved evaluation ground truth labels."
            )

        # Metrics Accumulators
        correct_count = 0
        taxonomy_compliance_count = 0
        case_results: List[Dict[str, Any]] = []

        # Unknown Behavior Tracking
        expected_unknown_total = 0
        expected_unknown_correct = 0
        non_unknown_total = 0
        non_unknown_predicted_as_unknown = 0
        total_predicted_unknown = 0

        # Language Form Tracking
        language_stats: Dict[str, Dict[str, int]] = {}
        # Case Type Tracking
        case_type_stats: Dict[str, Dict[str, int]] = {}
        # Category Tracking
        category_stats: Dict[str, Dict[str, int]] = {
            cat: {"expected_count": 0, "predicted_count": 0, "correct_count": 0}
            for cat in self.approved_categories
        }

        # Execute classification for every case
        for case in eval_cases:
            case_id = case.get("id", "unknown_id")
            description = case["description"]
            expected_category = case["expected_category"]
            language_form = case.get("language_form", "English")
            case_type = case.get("case_type", "direct")

            # Run classifier pipeline
            result = self.classifier.classify(description)
            predicted_category = result.category

            # Check taxonomy compliance
            is_compliant = predicted_category in self.approved_categories
            if is_compliant:
                taxonomy_compliance_count += 1

            # Check exact category correctness
            is_correct = (predicted_category == expected_category)
            if is_correct:
                correct_count += 1

            # Update unknown behavior stats
            if expected_category == "Unknown":
                expected_unknown_total += 1
                if is_correct:
                    expected_unknown_correct += 1
            else:
                non_unknown_total += 1
                if predicted_category == "Unknown":
                    non_unknown_predicted_as_unknown += 1

            if predicted_category == "Unknown":
                total_predicted_unknown += 1

            # Update language form stats
            if language_form not in language_stats:
                language_stats[language_form] = {"total": 0, "correct": 0}
            language_stats[language_form]["total"] += 1
            if is_correct:
                language_stats[language_form]["correct"] += 1

            # Update case type stats
            if case_type not in case_type_stats:
                case_type_stats[case_type] = {"total": 0, "correct": 0}
            case_type_stats[case_type]["total"] += 1
            if is_correct:
                case_type_stats[case_type]["correct"] += 1

            # Update category stats
            if expected_category in category_stats:
                category_stats[expected_category]["expected_count"] += 1
            if predicted_category in category_stats:
                category_stats[predicted_category]["predicted_count"] += 1
            if is_correct and expected_category in category_stats:
                category_stats[expected_category]["correct_count"] += 1

            case_results.append({
                "id": case_id,
                "description": description,
                "expected_category": expected_category,
                "predicted_category": predicted_category,
                "is_correct": is_correct,
                "is_compliant": is_compliant,
                "status": result.status,
                "reliability": result.reliability,
                "language_form": language_form,
                "case_type": case_type
            })

        # Calculate final rates strictly from actual execution
        accuracy = round(correct_count / total_cases, 4)
        taxonomy_compliance_rate = round(taxonomy_compliance_count / total_cases, 4)

        unknown_metrics = {
            "expected_unknown_total": expected_unknown_total,
            "expected_unknown_correct": expected_unknown_correct,
            "expected_unknown_accuracy": (
                round(expected_unknown_correct / expected_unknown_total, 4)
                if expected_unknown_total > 0 else 1.0
            ),
            "non_unknown_total": non_unknown_total,
            "non_unknown_predicted_as_unknown": non_unknown_predicted_as_unknown,
            "false_unknown_rate": (
                round(non_unknown_predicted_as_unknown / non_unknown_total, 4)
                if non_unknown_total > 0 else 0.0
            ),
            "total_predicted_unknown": total_predicted_unknown
        }

        language_form_results = {
            lang: {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": round(stats["correct"] / stats["total"], 4) if stats["total"] > 0 else 0.0
            }
            for lang, stats in language_stats.items()
        }

        case_type_results = {
            ctype: {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": round(stats["correct"] / stats["total"], 4) if stats["total"] > 0 else 0.0
            }
            for ctype, stats in case_type_stats.items()
        }

        return EvaluationReport(
            execution_timestamp=datetime.now(timezone.utc).isoformat(),
            client_type=client_type,
            is_fake_client=is_fake_client,
            is_live_gemini=is_live_gemini,
            model_name=str(model_name),
            total_cases=total_cases,
            correct_count=correct_count,
            accuracy=accuracy,
            taxonomy_compliance_count=taxonomy_compliance_count,
            taxonomy_compliance_rate=taxonomy_compliance_rate,
            unknown_metrics=unknown_metrics,
            language_form_results=language_form_results,
            case_type_results=case_type_results,
            category_results=category_stats,
            dataset_review_status=dataset_review_status,
            benchmark_status=benchmark_status,
            benchmark_accuracy_claimable=benchmark_accuracy_claimable,
            disclaimer=disclaimer,
            case_results=case_results
        )
