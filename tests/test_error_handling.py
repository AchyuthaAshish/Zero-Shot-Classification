"""Unit tests for domain exceptions and error handling contracts.

Verifies SRS Section 4 (FR-014) and SRS Section 20.
"""

import unittest
from core.exceptions import (
    DefectClassificationException,
    InputError,
    TaxonomyError,
    ModelError,
    ValidationError,
    VerificationError,
    PersistenceError,
    ConfigurationError
)


class TestErrorHandling(unittest.TestCase):

    def test_domain_exception_hierarchy(self):
        """All domain exceptions must inherit from DefectClassificationException."""
        exceptions = [
            InputError("Input invalid"),
            TaxonomyError("Taxonomy corrupt"),
            ModelError("Model timed out"),
            ValidationError("Category invalid"),
            VerificationError("Verification failed"),
            PersistenceError("DB failed"),
            ConfigurationError("API key missing")
        ]

        for exc in exceptions:
            self.assertIsInstance(exc, DefectClassificationException)
            self.assertIsInstance(exc, Exception)

    def test_error_codes(self):
        """Each exception class must have its distinct error code conforming to SRS Sec 20."""
        self.assertEqual(InputError().code, "INPUT_ERROR")
        self.assertEqual(TaxonomyError().code, "TAXONOMY_ERROR")
        self.assertEqual(ModelError().code, "MODEL_ERROR")
        self.assertEqual(ValidationError().code, "VALIDATION_ERROR")
        self.assertEqual(VerificationError().code, "VERIFICATION_ERROR")
        self.assertEqual(PersistenceError().code, "PERSISTENCE_ERROR")
        self.assertEqual(ConfigurationError().code, "CONFIGURATION_ERROR")

    def test_string_representation_is_clean(self):
        """Exception string formatting should include code and message for controlled logging."""
        exc = ModelError("Connection refused by endpoint")
        self.assertEqual(str(exc), "[MODEL_ERROR] Connection refused by endpoint")


if __name__ == "__main__":
    unittest.main()
