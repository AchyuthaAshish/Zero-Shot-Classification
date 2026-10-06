"""Unit tests for category validation logic.

Verifies PRD Section 9 (F7), SRS Section 4 (FR-007), AC-002, and BR-002, BR-003.
"""

import unittest
from validation.category_validator import CategoryValidator, get_category_validator
from core.exceptions import ValidationError


class TestCategoryValidator(unittest.TestCase):

    def setUp(self):
        self.validator = get_category_validator()
        self.approved_categories = self.validator._repo.get_categories()

    def test_all_approved_categories_pass(self):
        """All categories derived from the authoritative taxonomy must pass validation."""
        self.assertEqual(len(self.approved_categories), 8)
        for cat in self.approved_categories:
            is_valid, canonical, error_msg = self.validator.validate_category(cat)
            self.assertTrue(is_valid, f"Approved category '{cat}' failed validation.")
            self.assertEqual(canonical, cat)
            self.assertIsNone(error_msg)

    def test_case_insensitive_reconciliation(self):
        """Lower-case or mixed-case versions of approved categories should reconcile to canonical form."""
        is_valid, canonical, error_msg = self.validator.validate_category("mechanical fault")
        self.assertTrue(is_valid)
        self.assertEqual(canonical, "Mechanical Fault")

        is_valid, canonical, error_msg = self.validator.validate_category("ELECTRICAL FAULT")
        self.assertTrue(is_valid)
        self.assertEqual(canonical, "Electrical Fault")

        is_valid, canonical, error_msg = self.validator.validate_category("unknown")
        self.assertTrue(is_valid)
        self.assertEqual(canonical, "Unknown")

    def test_rejection_of_invented_categories(self):
        """AC-002: Model-invented categories MUST be rejected."""
        hallucinated_labels = [
            "Motor Failure",
            "Bearing Failure",
            "Bearing Problem",
            "Hardware Issue",
            "Vibration Fault",
            "Overheating Issue",
            "Network Glitch",
            "General Breakdown"
        ]
        for label in hallucinated_labels:
            is_valid, canonical, error_msg = self.validator.validate_category(label)
            self.assertFalse(is_valid, f"Invented category '{label}' was incorrectly accepted.")
            self.assertIsNone(canonical)
            self.assertIsNotNone(error_msg)
            self.assertIn("Invalid category", error_msg)

    def test_enforce_category_raises_exception_on_invalid(self):
        """Enforce method must raise ValidationError on illegal categories."""
        with self.assertRaises(ValidationError):
            self.validator.enforce_category("Motor Failure")

        # Valid category returns string
        result = self.validator.enforce_category("Mechanical Fault")
        self.assertEqual(result, "Mechanical Fault")

    def test_empty_or_none_category_rejected(self):
        """None or empty category is rejected."""
        is_valid, _, error_msg = self.validator.validate_category(None)
        self.assertFalse(is_valid)
        self.assertIsNotNone(error_msg)

        is_valid, _, error_msg = self.validator.validate_category("   ")
        self.assertFalse(is_valid)
        self.assertIsNotNone(error_msg)


if __name__ == "__main__":
    unittest.main()
