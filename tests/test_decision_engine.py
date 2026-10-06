"""Unit tests for the Decision Engine.

Verifies PRD Section 9 (F8, F9), SRS Section 3 (C9), SRS Section 4 (FR-008, FR-010),
AC-002, AC-004, and BR-002, BR-004, BR-010.
"""

import unittest
from core.decision_engine import DecisionEngine
from core.schemas import PreprocessedInput, LLMRawResponse
from validation.category_validator import get_category_validator


class TestDecisionEngine(unittest.TestCase):

    def setUp(self):
        self.engine = DecisionEngine(validator=get_category_validator())
        self.sample_input = PreprocessedInput(
            raw_text="Motor producing abnormal vibration",
            normalized_text="Motor producing abnormal vibration",
            detected_language="English",
            is_code_switched=False,
            char_count=35,
            word_count=4
        )

    def test_decide_valid_proposal(self):
        """Valid proposal from approved taxonomy returns success status."""
        llm_resp = LLMRawResponse(
            proposed_category="Mechanical Fault",
            reason="Abnormal vibration in motor indicates mechanical fault."
        )
        result = self.engine.decide_from_llm(self.sample_input, llm_resp)
        self.assertEqual(result.category, "Mechanical Fault")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.reliability, "High")
        self.assertEqual(result.original_description, self.sample_input.raw_text)

    def test_decide_invented_proposal_rejected(self):
        """AC-002: Model-invented label triggers validation_error status, not persisted as final label."""
        llm_resp = LLMRawResponse(
            proposed_category="Motor Failure",
            reason="The motor has failed."
        )
        result = self.engine.decide_from_llm(self.sample_input, llm_resp)
        self.assertEqual(result.status, "validation_error")
        self.assertNotEqual(result.category, "Motor Failure")
        self.assertEqual(result.category, "Unknown")
        self.assertEqual(result.reliability, "Low")
        self.assertIsNotNone(result.error_message)

    def test_decide_unknown_proposal(self):
        """AC-004 & DR-006: Insufficient evidence proposals yield Unknown category with unknown status."""
        llm_resp = LLMRawResponse(
            proposed_category="Unknown",
            reason="The description is too vague to determine a specific failure mode."
        )
        result = self.engine.decide_from_llm(self.sample_input, llm_resp)
        self.assertEqual(result.category, "Unknown")
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.reliability, "Medium")

    def test_verification_feedback_adjustment(self):
        """When verification suggests an approved alternative, decision engine reconciles it to success."""
        llm_resp = LLMRawResponse(
            proposed_category="Mechanical Fault",
            reason="Motor stopped."
        )
        feedback = {
            "verified": False,
            "alternative_category": "Power Supply Fault"
        }
        result = self.engine.decide_from_llm(self.sample_input, llm_resp, verification_feedback=feedback)
        self.assertEqual(result.category, "Power Supply Fault")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.reliability, "Medium")
        self.assertIn("Power Supply Fault", result.reason)

    def test_verification_rejection_without_alternative_yields_unknown_status(self):
        """When verification rejects proposal and provides no alternative, status must be 'unknown', not 'success'."""
        llm_resp = LLMRawResponse(
            proposed_category="Mechanical Fault",
            reason="Motor stopped suddenly."
        )
        feedback = {
            "verified": False,
            "alternative_category": None
        }
        result = self.engine.decide_from_llm(self.sample_input, llm_resp, verification_feedback=feedback)
        self.assertEqual(result.category, "Unknown")
        self.assertEqual(result.status, "unknown")
        self.assertNotEqual(result.status, "success")
        self.assertEqual(result.reliability, "Low")
        self.assertIn("Defaulted to Unknown", result.reason)

    def test_model_error_not_disguised_as_unknown(self):
        """BR-010: Model/infrastructure failures must be marked with error status, not disguised."""
        error_result = self.engine.create_error_result(
            raw_text=self.sample_input.raw_text,
            error_message="API connection timed out after 10 seconds.",
            error_type="model_error"
        )
        self.assertEqual(error_result.status, "model_error")
        self.assertEqual(error_result.reliability, "Low")
        self.assertIn("API connection timed out", error_result.reason)


if __name__ == "__main__":
    unittest.main()
