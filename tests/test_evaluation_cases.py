"""Automated tests validating evaluation benchmark dataset and offline evaluation runner.

Verifies PRD Section 11 (Success Metrics), Section 14 (Definition of Done),
and SRS Section 11 (Testing Strategy T1-T6).
Validates tests/fixtures/evaluation_cases.json schema, distribution, and pipeline integration.
"""

import json
from pathlib import Path
import unittest
from taxonomy.repository import get_taxonomy_repository
from classification.classifier import DefectClassifier
from llm.client import FakeLLMClient


class TestEvaluationDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fixture_path = Path(__file__).resolve().parent / "fixtures" / "evaluation_cases.json"
        with open(cls.fixture_path, "r", encoding="utf-8") as f:
            cls.cases = json.load(f)
        cls.approved_categories = set(get_taxonomy_repository().get_categories())

    def test_fixture_file_exists_and_contains_sufficient_cases(self):
        """Must contain approximately 80 or more comprehensive evaluation cases."""
        self.assertTrue(self.fixture_path.exists())
        self.assertGreaterEqual(len(self.cases), 80, "Evaluation cases must be approximately 80 or more.")

    def test_case_schema_integrity(self):
        """Every case must strictly conform to the required benchmark schema."""
        required_keys = {
            "id", "description", "language_form", "case_type",
            "expected_category", "rationale", "review_status"
        }
        seen_ids = set()

        for case in self.cases:
            self.assertEqual(set(case.keys()), required_keys, f"Schema mismatch in case {case.get('id')}")
            self.assertNotIn(case["id"], seen_ids, f"Duplicate case ID found: {case['id']}")
            seen_ids.add(case["id"])

            self.assertIn(case["expected_category"], self.approved_categories)
            self.assertIn(case["language_form"], {"English", "Telugu", "Telugu-English"})
            self.assertIn(case["case_type"], {"direct", "paraphrase", "boundary", "unknown", "prompt_injection"})
            self.assertEqual(case["review_status"], "pending_human_review")
            self.assertTrue(len(case["description"].strip()) > 5)
            self.assertTrue(len(case["rationale"].strip()) > 5)

    def test_required_coverage_dimensions(self):
        """Verifies evaluation dataset covers all mandated evaluation scenarios."""
        case_types = {c["case_type"] for c in self.cases}
        languages = {c["language_form"] for c in self.cases}
        categories_covered = {c["expected_category"] for c in self.cases}

        self.assertIn("direct", case_types)
        self.assertIn("paraphrase", case_types)
        self.assertIn("boundary", case_types)
        self.assertIn("unknown", case_types)
        self.assertIn("prompt_injection", case_types)

        self.assertIn("English", languages)
        self.assertIn("Telugu", languages)
        self.assertIn("Telugu-English", languages)

        # All 8 approved categories must be represented in the evaluation dataset
        self.assertEqual(categories_covered, self.approved_categories)

    def test_offline_evaluation_execution_with_fake_client(self):
        """Runs the entire benchmark dataset through DefectClassifier using FakeLLMClient."""
        classifier = DefectClassifier(llm_client=FakeLLMClient())
        evaluated_count = 0

        for case in self.cases:
            result = classifier.classify(case["description"])
            self.assertIn(result.category, self.approved_categories)
            self.assertIn(result.status, {"success", "unknown", "validation_error"})
            self.assertEqual(result.original_description, case["description"])
            evaluated_count += 1

        self.assertEqual(evaluated_count, len(self.cases))

    def test_reusable_evaluation_runner_mechanics_and_honesty(self):
        """
        Verifies EvaluationRunner computes authentic metrics, marks results as fake-client,
        and refuses to claim benchmark accuracy.
        """
        from tests.evaluation_runner import EvaluationRunner

        classifier = DefectClassifier(llm_client=FakeLLMClient())
        runner = EvaluationRunner(classifier=classifier, fixture_path=self.fixture_path)
        report = runner.run()

        # Runner identification and honesty assertions
        self.assertEqual(report.client_type, "fake_client")
        self.assertTrue(report.is_fake_client)
        self.assertFalse(report.is_live_gemini)
        self.assertEqual(report.benchmark_status, "mechanics_verification_only")
        self.assertFalse(report.benchmark_accuracy_claimable, "Fake client runs must NOT claim benchmark accuracy.")
        self.assertIn("MECHANICS VERIFICATION ONLY", report.disclaimer)

        # Core metric assertions
        self.assertEqual(report.total_cases, len(self.cases))
        self.assertEqual(report.taxonomy_compliance_count, len(self.cases))
        self.assertEqual(report.taxonomy_compliance_rate, 1.0, "Taxonomy compliance must strictly be 100%.")
        self.assertGreater(report.correct_count, 0)
        self.assertGreater(report.accuracy, 0.0)

        # Breakdown by language form assertions
        self.assertIn("English", report.language_form_results)
        self.assertIn("Telugu", report.language_form_results)
        self.assertIn("Telugu-English", report.language_form_results)
        for lang, stats in report.language_form_results.items():
            self.assertGreater(stats["total"], 0)
            self.assertIn("correct", stats)
            self.assertIn("accuracy", stats)

        # Breakdown by case type assertions
        for ctype in ["direct", "paraphrase", "boundary", "unknown", "prompt_injection"]:
            self.assertIn(ctype, report.case_type_results)
            self.assertGreater(report.case_type_results[ctype]["total"], 0)

        # Unknown behavior metrics assertions
        self.assertIn("expected_unknown_total", report.unknown_metrics)
        self.assertIn("expected_unknown_correct", report.unknown_metrics)
        self.assertIn("false_unknown_rate", report.unknown_metrics)
        self.assertGreater(report.unknown_metrics["expected_unknown_total"], 0)

        # Human-readable summary output format verification
        summary = report.format_summary()
        self.assertIn("EVALUATION BENCHMARK REPORT", summary)
        self.assertIn("OFFLINE FAKE-CLIENT RUN", summary)
        self.assertIn("Taxonomy Compliance     : 100.00%", summary)
        self.assertIn(">>> IMPORTANT NOTICE / DISCLAIMER <<<", summary)

    def test_evaluation_runner_with_subset_limit(self):
        """EvaluationRunner can evaluate a constrained subset of cases for quick verification."""
        from tests.evaluation_runner import EvaluationRunner

        classifier = DefectClassifier(llm_client=FakeLLMClient())
        runner = EvaluationRunner(classifier=classifier, fixture_path=self.fixture_path)
        report = runner.run(limit=5)

        self.assertEqual(report.total_cases, 5)
        self.assertEqual(len(report.case_results), 5)


if __name__ == "__main__":
    unittest.main()
