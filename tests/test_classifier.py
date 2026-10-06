"""Unit and integration tests for DefectClassifier using FakeLLMClient.

Verifies SRS Section 3 (C6), FR-004, FR-005, FR-006, FR-014, and NFR-004.
Executes without live API calls or API keys.
"""

import unittest
from classification.classifier import DefectClassifier
from llm.client import FakeLLMClient
from core.exceptions import ModelError, ConfigurationError


class TestDefectClassifier(unittest.TestCase):

    def setUp(self):
        self.fake_client = FakeLLMClient()
        self.classifier = DefectClassifier(llm_client=self.fake_client)

    def test_successful_classification_flow(self):
        """Verifies end-to-end classification with approved category proposal."""
        result = self.classifier.classify("Motor producing abnormal vibration and bearing chatter.")
        self.assertEqual(result.category, "Mechanical Fault")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.reliability, "High")
        self.assertEqual(result.language, "English")
        self.assertIn("vibration", result.original_description)

    def test_rejection_of_hallucinated_model_category(self):
        """AC-002: Model-invented labels become validation_error and are not accepted."""
        fake_client = FakeLLMClient(
            canned_responses={
                "bearing failed": {
                    "category": "Bearing Failure",
                    "reason": "The bearing has failed."
                }
            }
        )
        classifier = DefectClassifier(llm_client=fake_client)
        result = classifier.classify("Machine bearing failed completely.")

        self.assertEqual(result.status, "validation_error")
        self.assertEqual(result.category, "Unknown")
        self.assertEqual(result.reliability, "Low")
        self.assertIn("Bearing Failure", result.reason)

    def test_unknown_result_for_vague_description(self):
        """AC-004: Insufficient evidence description resolves to Unknown category and unknown status."""
        result = self.classifier.classify("Something is wrong with the machine.")
        self.assertEqual(result.category, "Unknown")
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.reliability, "Medium")

    def test_empty_input_handled_as_input_error(self):
        """Empty input produces controlled input_error result without crashing."""
        result = self.classifier.classify("   ")
        self.assertEqual(result.status, "input_error")
        self.assertEqual(result.category, "Unknown")
        self.assertIsNotNone(result.error_message)

    def test_missing_api_key_configuration_error(self):
        """Missing API credentials produces controlled configuration_error result."""
        error_client = FakeLLMClient(
            canned_responses={
                "motor": ConfigurationError("Missing Gemini API key.")
            }
        )
        classifier = DefectClassifier(llm_client=error_client)
        result = classifier.classify("Motor is vibrating.")
        self.assertEqual(result.status, "configuration_error")
        self.assertEqual(result.category, "Unknown")
        self.assertIn("Missing Gemini API key", result.reason)

    def test_model_timeout_or_rate_limit_handled(self):
        """BR-010: Model timeout or rate limit produces controlled model_error."""
        error_client = FakeLLMClient(
            canned_responses={
                "motor": ModelError("Gemini API request timed out after 10s.")
            }
        )
        classifier = DefectClassifier(llm_client=error_client)
        result = classifier.classify("Motor is vibrating.")
        self.assertEqual(result.status, "model_error")
        self.assertEqual(result.category, "Unknown")
        self.assertIn("timed out", result.reason)

    def test_code_switched_classification_flow(self):
        """Code-switched input successfully detected and passed through pipeline."""
        result = self.classifier.classify("Motor lo unusual sound vastundi and vibration undi.")
        self.assertEqual(result.category, "Mechanical Fault")
        self.assertEqual(result.status, "success")
    def test_public_classify_defect_service_function(self):
        """Verifies the public service function classify_defect returns canonical ClassificationResult."""
        from classification.classifier import classify_defect
        result = classify_defect("Short circuit in 480V distribution panel with sparking.", classifier=self.classifier)
        self.assertEqual(result.category, "Electrical Fault")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.reliability, "High")
        self.assertEqual(result.original_description, "Short circuit in 480V distribution panel with sparking.")

    def test_unsupported_llm_provider_raises_configuration_error(self):
        """Verifies that an unsupported LLM_PROVIDER raises a controlled ConfigurationError."""
        import os
        from config.settings import load_settings, reset_settings

        orig_provider = os.environ.get("LLM_PROVIDER")
        try:
            os.environ["LLM_PROVIDER"] = "unsupported_llm_vendor"
            reset_settings()
            with self.assertRaises(ConfigurationError) as ctx:
                load_settings(auto_load_env=False)
            self.assertIn("Unsupported LLM provider 'unsupported_llm_vendor'", str(ctx.exception))
        finally:
            if orig_provider is not None:
                os.environ["LLM_PROVIDER"] = orig_provider
            else:
                os.environ.pop("LLM_PROVIDER", None)
            reset_settings()

    def test_no_silent_fallback_to_fake_client_outside_explicit_test_mode(self):
        """
        Verifies that when LLM_PROVIDER is 'gemini', get_llm_client() instantiates GeminiLLMClient
        and does NOT silently fall back to FakeLLMClient.
        """
        import os
        from config.settings import reset_settings
        from llm.client import get_llm_client, GeminiLLMClient

        orig_provider = os.environ.get("LLM_PROVIDER")
        orig_key = os.environ.get("LLM_API_KEY")
        try:
            os.environ["LLM_PROVIDER"] = "gemini"
            os.environ["LLM_API_KEY"] = ""
            reset_settings()
            client = get_llm_client(force_fake=False)
            self.assertIsInstance(client, GeminiLLMClient, "Must instantiate Gemini client, not Fake client.")
            # If API key is missing, attempting to invoke it raises controlled ConfigurationError, not silent fake fallback
            with self.assertRaises(ConfigurationError) as ctx:
                client.classify("test", "test")
            self.assertIn("Gemini API key is missing", str(ctx.exception))
        finally:
            if orig_provider is not None:
                os.environ["LLM_PROVIDER"] = orig_provider
            else:
                os.environ.pop("LLM_PROVIDER", None)
            if orig_key is not None:
                os.environ["LLM_API_KEY"] = orig_key
            else:
                os.environ.pop("LLM_API_KEY", None)
            reset_settings()


if __name__ == "__main__":
    unittest.main()
