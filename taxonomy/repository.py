"""Authoritative Taxonomy Repository for Zero-Shot Industrial Defect Classification.

Conforms to PRD Section 3 & 4, SRS Section 1.2, FR-003, and Component C3.

IMPORTANT ARCHITECTURAL RULE:
`taxonomy/defect_taxonomy.json` is the SOLE authoritative source for the approved
defect taxonomy labels, their exact definitions, and their canonical ordering.
Python modules must NOT duplicate the eight-category list.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from core.exceptions import TaxonomyError

# IMMUTABLE SPECIFICATION CONTRACT (PRD Section 3 / SRS Section 1.2):
# This set serves ONLY as an immutable specification contract to detect unauthorized
# taxonomy changes, missing categories, or unapproved additions in defect_taxonomy.json.
# It is NOT an independent taxonomy knowledge base.
# `defect_taxonomy.json` remains the SOLE authoritative source for definitions,
# examples, boundaries, exclusions, signals, and canonical display order.
REQUIRED_APPROVED_CATEGORIES: frozenset[str] = frozenset([
    "Mechanical Fault",
    "Electrical Fault",
    "Sensor Fault",
    "Temperature Fault",
    "Software Fault",
    "Power Supply Fault",
    "Communication Fault",
    "Unknown"
])



class TaxonomyRepository:
    """Centralized repository loading and serving the authoritative defect taxonomy from JSON."""

    def __init__(self, taxonomy_file_path: Optional[str] = None):
        if taxonomy_file_path:
            self._path = Path(taxonomy_file_path)
        else:
            # Default to defect_taxonomy.json in the same folder as this module
            self._path = Path(__file__).resolve().parent / "defect_taxonomy.json"
        self._taxonomy: Optional[Dict[str, Any]] = None
        self._categories: Optional[List[str]] = None
        self._load_taxonomy()

    def _load_taxonomy(self) -> None:
        """
        Loads and validates the taxonomy JSON file.
        Derives the approved categories and their canonical ordering directly from JSON keys.
        """
        if not self._path.exists():
            raise TaxonomyError(f"Authoritative taxonomy file not found at {self._path}")

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise TaxonomyError(f"Failed to parse taxonomy JSON file: {e}") from e

        if not isinstance(data, dict):
            raise TaxonomyError("Taxonomy JSON root must be a dictionary.")

        loaded_keys = set(data.keys())
        categories = list(data.keys())

        # Exact taxonomy integrity check: rejected if any required category is missing or unexpected category present
        if loaded_keys != REQUIRED_APPROVED_CATEGORIES:
            missing = sorted(list(REQUIRED_APPROVED_CATEGORIES - loaded_keys))
            unexpected = sorted(list(loaded_keys - REQUIRED_APPROVED_CATEGORIES))
            err_parts = []
            if missing:
                err_parts.append(f"missing approved categories: {missing}")
            if unexpected:
                err_parts.append(f"unexpected categories present: {unexpected}")
            raise TaxonomyError(
                f"Taxonomy integrity check failed for {self._path}: {', '.join(err_parts)}."
            )


        # Validate structural completeness of each category entry
        for cat, entry in data.items():
            if not isinstance(entry, dict) or "definition" not in entry or "examples" not in entry:
                raise TaxonomyError(f"Category '{cat}' is missing required fields ('definition', 'examples').")

        self._taxonomy = data
        self._categories = categories

    def get_taxonomy(self) -> Dict[str, Any]:
        """Returns the full taxonomy data dictionary derived from JSON."""
        if self._taxonomy is None:
            self._load_taxonomy()
        return self._taxonomy

    def get_categories(self) -> List[str]:
        """Returns the canonical list of approved categories derived directly from defect_taxonomy.json."""
        if self._categories is None:
            self._load_taxonomy()
        return list(self._categories)

    def is_valid_category(self, category: Optional[str]) -> bool:
        """Checks if a category string matches one of the approved categories in the taxonomy."""
        if not category or not isinstance(category, str):
            return False
        return category.strip() in self.get_categories()

    def get_category_info(self, category: str) -> Dict[str, Any]:
        """Returns the dictionary description for a given category."""
        tax = self.get_taxonomy()
        if category not in tax:
            raise TaxonomyError(f"Unknown category '{category}' requested.")
        return tax[category]

    def format_taxonomy_for_prompt(self) -> str:
        """Formats the taxonomy definitions and signals for inclusion in LLM prompts."""
        tax = self.get_taxonomy()
        lines = []
        for cat in self.get_categories():
            info = tax[cat]
            defn = info.get("definition", "")
            signals = ", ".join(info.get("signals", []))
            lines.append(f"- **{cat}**: {defn}")
            if signals:
                lines.append(f"  *Key signals*: {signals}")
            exclusion = info.get("exclusion_guidance")
            if exclusion:
                lines.append(f"  *Exclusion*: {exclusion}")
        return "\n".join(lines)


# Global singleton instance for clean access across services
_default_repository: Optional[TaxonomyRepository] = None

def get_taxonomy_repository() -> TaxonomyRepository:
    """Returns the singleton instance of TaxonomyRepository."""
    global _default_repository
    if _default_repository is None:
        _default_repository = TaxonomyRepository()
    return _default_repository
