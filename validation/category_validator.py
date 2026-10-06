"""Category Validator for Zero-Shot Industrial Defect Classification.

Conforms to PRD Section 9 (F7), SRS Section 3 (C7), SRS Section 4 (FR-007),
AC-002, and Business Rules BR-002, BR-003, and BR-009.

Guarantees that no model-invented category (such as 'Motor Failure',
'Bearing Failure', or 'Hardware Issue') can ever be returned or persisted.
"""

from typing import Tuple, Optional
from taxonomy.repository import get_taxonomy_repository, TaxonomyRepository
from core.exceptions import ValidationError


class CategoryValidator:
    """Independent validator checking proposed categories against authoritative taxonomy."""

    def __init__(self, repository: Optional[TaxonomyRepository] = None):
        self._repo = repository or get_taxonomy_repository()
        self._approved_categories = self._repo.get_categories()
        # Case-insensitive lookup map for clean casing reconciliation
        self._case_insensitive_map = {cat.lower(): cat for cat in self._approved_categories}

    def validate_category(self, proposed_category: Optional[str]) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validates the proposed category string.

        Returns:
            Tuple of:
            - is_valid (bool): True if valid, False otherwise
            - canonical_category (Optional[str]): Clean canonical category name if valid, None if invalid
            - error_message (Optional[str]): Rejection reason if invalid, None if valid
        """
        if not proposed_category or not isinstance(proposed_category, str):
            return False, None, "Proposed category is empty or not a string."

        cleaned = proposed_category.strip()

        # Exact match
        if cleaned in self._approved_categories:
            return True, cleaned, None

        # Case normalization check (e.g., 'mechanical fault' -> 'Mechanical Fault')
        lower_cleaned = cleaned.lower()
        if lower_cleaned in self._case_insensitive_map:
            return True, self._case_insensitive_map[lower_cleaned], None

        # Rejection of invented categories
        return (
            False,
            None,
            f"Invalid category '{cleaned}'. Category does not exist in authoritative 8-category taxonomy."
        )

    def is_valid_category(self, proposed_category: Optional[str]) -> bool:
        """Returns True if proposed_category is a valid approved category."""
        is_valid, _, _ = self.validate_category(proposed_category)
        return is_valid


    def enforce_category(self, proposed_category: Optional[str]) -> str:
        """
        Strictly enforces that the category belongs to the approved taxonomy.
        Raises ValidationError if invalid.
        """
        is_valid, canonical_category, error_msg = self.validate_category(proposed_category)
        if not is_valid or canonical_category is None:
            raise ValidationError(error_msg or f"Invalid category '{proposed_category}'.")
        return canonical_category


# Global singleton instance
_default_validator: Optional[CategoryValidator] = None

def get_category_validator() -> CategoryValidator:
    """Returns singleton instance of CategoryValidator."""
    global _default_validator
    if _default_validator is None:
        _default_validator = CategoryValidator()
    return _default_validator
