"""Unit tests for input preprocessing and language-aware processing.

Verifies PRD Section 9 (F1, F2), SRS Section 4 (FR-001, FR-002),
AC-003, and BR-006, BR-008.
"""

import unittest
from preprocessing.text_processor import TextProcessor, get_text_processor
from config.settings import get_settings, DEFAULT_MAX_INPUT_LENGTH
from core.exceptions import InputError


class TestTextProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = get_text_processor()

    def test_empty_input_rejected(self):
        """FR-001: Empty or whitespace-only input must raise InputError."""
        with self.assertRaises(InputError):
            self.processor.preprocess("")

        with self.assertRaises(InputError):
            self.processor.preprocess("   \n\t  ")

        with self.assertRaises(InputError):
            self.processor.preprocess(None)

    def test_preserves_original_user_input(self):
        """BR-006: Original user text must be preserved unmodified."""
        raw_text = "  Motor   is making  a strange noise!  "
        result = self.processor.preprocess(raw_text)
        self.assertEqual(result.raw_text, raw_text)
        self.assertEqual(result.normalized_text, "Motor is making a strange noise!")

    def test_english_language_detection(self):
        """Standard English defect description."""
        text = "Motor producing abnormal vibration and bearing noise."
        result = self.processor.preprocess(text)
        self.assertEqual(result.detected_language, "English")
        self.assertFalse(result.is_code_switched)

    def test_telugu_native_script_detection(self):
        """PRD Section 8 (J3): Telugu script description enters same pipeline."""
        text = "మోటార్ ఎక్కువగా వైబ్రేట్ అవుతోంది"
        result = self.processor.preprocess(text)
        self.assertEqual(result.detected_language, "Telugu")
        self.assertFalse(result.is_code_switched)
        self.assertEqual(result.raw_text, text)

    def test_telugu_english_code_switched_detection(self):
        """PRD Section 8 (J4) & SRS FR-002: Code-switched Telugu-English."""
        text = "Motor lo unusual sound vastundi"
        result = self.processor.preprocess(text)
        self.assertEqual(result.detected_language, "Telugu-English (Code-Switched)")
        self.assertTrue(result.is_code_switched)

    def test_mixed_script_detection(self):
        """Defect description mixing Roman and Telugu scripts."""
        text = "Motor మోటార్ abnormal sound vastundi"
        result = self.processor.preprocess(text)
        self.assertEqual(result.detected_language, "Telugu-English Mixed Script")
        self.assertTrue(result.is_code_switched)

    def test_default_max_length_uses_configuration_layer(self):
        """Verifies TextProcessor inherits max_input_length from config/settings.py."""
        settings = get_settings()
        self.assertEqual(settings.max_input_length, DEFAULT_MAX_INPUT_LENGTH)
        self.assertEqual(self.processor._max_length, 2000)

        # 2000 chars should pass
        valid_input = "A" * 2000
        res = self.processor.preprocess(valid_input)
        self.assertEqual(res.char_count, 2000)

        # 2001 chars should be rejected
        with self.assertRaises(InputError) as ctx:
            self.processor.preprocess("A" * 2001)
        self.assertIn("exceeds maximum length of 2000 characters", str(ctx.exception))

    def test_custom_configured_max_length_enforced(self):
        """Verifies explicitly configured limit is enforced."""
        short_processor = TextProcessor(max_length=100)
        self.assertEqual(short_processor._max_length, 100)

        with self.assertRaises(InputError) as ctx:
            short_processor.preprocess("A" * 101)
        self.assertIn("exceeds maximum length of 100 characters", str(ctx.exception))

        passed = short_processor.preprocess("A" * 100)
        self.assertEqual(passed.char_count, 100)

    def test_settings_max_input_length_validation(self):
        """Verifies zero, negative, and malformed MAX_INPUT_LENGTH fall back to 2,000."""
        import os
        from config.settings import load_settings

        # Test zero value
        os.environ["MAX_INPUT_LENGTH"] = "0"
        s = load_settings()
        self.assertEqual(s.max_input_length, 2000)

        # Test negative value
        os.environ["MAX_INPUT_LENGTH"] = "-50"
        s = load_settings()
        self.assertEqual(s.max_input_length, 2000)

        # Test malformed non-integer value
        os.environ["MAX_INPUT_LENGTH"] = "invalid_number"
        s = load_settings()
        self.assertEqual(s.max_input_length, 2000)

        # Test valid positive override
        os.environ["MAX_INPUT_LENGTH"] = "1500"
        s = load_settings()
        self.assertEqual(s.max_input_length, 1500)

        # Cleanup
        del os.environ["MAX_INPUT_LENGTH"]


if __name__ == "__main__":
    unittest.main()

