"""Unit tests for centralized taxonomy knowledge base and repository.

Verifies PRD Section 3 & 4, SRS Section 1.2, FR-003, and BR-002, BR-007.
"""

import os
import json
import tempfile
import unittest
from taxonomy.repository import TaxonomyRepository, get_taxonomy_repository
from core.exceptions import TaxonomyError


class TestTaxonomyRepository(unittest.TestCase):

    def setUp(self):
        self.repo = get_taxonomy_repository()

    def test_approved_categories_count_and_derivation(self):
        """Must derive exactly the 8 approved categories from defect_taxonomy.json."""
        categories = self.repo.get_categories()
        self.assertEqual(len(categories), 8)
        # Categories must strictly match JSON keys in defect_taxonomy.json
        raw_taxonomy = self.repo.get_taxonomy()
        self.assertEqual(categories, list(raw_taxonomy.keys()))
        self.assertIn("Unknown", categories)

    def test_category_schema_completeness(self):
        """Every category derived from JSON must contain definition, examples, signals, and boundaries."""
        taxonomy = self.repo.get_taxonomy()
        categories = self.repo.get_categories()
        for cat in categories:
            self.assertIn(cat, taxonomy)
            entry = taxonomy[cat]
            self.assertTrue(len(entry["definition"]) > 10, f"Definition for {cat} is too short.")
            self.assertTrue(len(entry["examples"]) >= 1, f"Examples for {cat} must not be empty.")
            self.assertIn("signals", entry)
            self.assertIn("boundaries", entry)

    def test_is_valid_category(self):
        """Only categories in defect_taxonomy.json return True."""
        for cat in self.repo.get_categories():
            self.assertTrue(self.repo.is_valid_category(cat))

        # Rejection of unapproved / invented labels
        self.assertFalse(self.repo.is_valid_category("Motor Failure"))
        self.assertFalse(self.repo.is_valid_category("Bearing Problem"))
        self.assertFalse(self.repo.is_valid_category("Hardware Issue"))
        self.assertFalse(self.repo.is_valid_category(""))
        self.assertFalse(self.repo.is_valid_category(None))

    def test_format_taxonomy_for_prompt(self):
        """Formatted prompt string must include all 8 categories derived from JSON."""
        prompt_text = self.repo.format_taxonomy_for_prompt()
        self.assertIsInstance(prompt_text, str)
        for cat in self.repo.get_categories():
            self.assertIn(cat, prompt_text)

    def _create_temp_taxonomy_file(self, data: dict) -> str:
        """Helper to create a temporary taxonomy JSON file."""
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return path

    def test_rejection_of_missing_approved_category(self):
        """Must reject taxonomy data if any approved category is missing."""
        tax_copy = dict(self.repo.get_taxonomy())
        del tax_copy["Sensor Fault"]  # Missing 1 approved category (7 total)
        temp_path = self._create_temp_taxonomy_file(tax_copy)
        try:
            with self.assertRaises(TaxonomyError) as ctx:
                TaxonomyRepository(taxonomy_file_path=temp_path)
            self.assertIn("missing approved categories", str(ctx.exception))
            self.assertIn("Sensor Fault", str(ctx.exception))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_rejection_of_unexpected_replacement_category(self):
        """Must reject taxonomy data if an approved category is replaced by an unapproved label."""
        tax_copy = dict(self.repo.get_taxonomy())
        # Replace Mechanical Fault with an unapproved label
        mech_data = tax_copy.pop("Mechanical Fault")
        tax_copy["Motor Failure"] = mech_data  # 8 categories, but one unexpected replacement
        temp_path = self._create_temp_taxonomy_file(tax_copy)
        try:
            with self.assertRaises(TaxonomyError) as ctx:
                TaxonomyRepository(taxonomy_file_path=temp_path)
            self.assertIn("Taxonomy integrity check failed", str(ctx.exception))
            self.assertIn("unexpected categories present", str(ctx.exception))
            self.assertIn("Motor Failure", str(ctx.exception))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_rejection_of_taxonomy_without_unknown(self):
        """Must reject taxonomy data if the required 'Unknown' safe outcome category is missing."""
        tax_copy = dict(self.repo.get_taxonomy())
        unknown_data = tax_copy.pop("Unknown")
        tax_copy["Other Fault"] = unknown_data  # 8 categories, but missing Unknown
        temp_path = self._create_temp_taxonomy_file(tax_copy)
        try:
            with self.assertRaises(TaxonomyError) as ctx:
                TaxonomyRepository(taxonomy_file_path=temp_path)
            self.assertIn("missing approved categories", str(ctx.exception))
            self.assertIn("Unknown", str(ctx.exception))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
