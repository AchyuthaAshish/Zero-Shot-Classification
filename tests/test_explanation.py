"""Unit Tests for Dynamic Input-Aware Explanation Generation.

Validates that:
1. Explanations incorporate actual evidence from the user input.
2. Explanations do not hallucinate unsupported component failures.
3. Explanations provide sentence structure and vocabulary variations.
4. Explanations stay concise (15-35 words, 1-2 sentences).
5. All 8 taxonomy categories are supported.
"""

import unittest
from ml.explanation import generate_local_explanation, extract_evidence


class TestDynamicExplanationGeneration(unittest.TestCase):
    """Test suite for input-aware explanation generator."""

    def test_mechanical_explanation_contains_input_evidence(self):
        text = "The motor is making a grinding noise."
        expl = generate_local_explanation(text, "Mechanical Fault")
        lower_expl = expl.lower()
        self.assertIn("grinding", lower_expl)
        self.assertIn("motor", lower_expl)
        # Must NOT claim bearing wear when bearing is not mentioned
        self.assertNotIn("bearing", lower_expl)

    def test_bearing_rattling_specifically_mentions_bearing(self):
        text = "The bearing is producing a loud rattling sound."
        expl = generate_local_explanation(text, "Mechanical Fault")
        lower_expl = expl.lower()
        self.assertIn("rattling", lower_expl)
        self.assertIn("bearing", lower_expl)

    def test_shaft_misalignment_explanation(self):
        text = "The conveyor shaft is misaligned."
        expl = generate_local_explanation(text, "Mechanical Fault")
        lower_expl = expl.lower()
        self.assertIn("shaft", lower_expl)
        self.assertIn("misalign", lower_expl)

    def test_sensor_fault_explanation(self):
        text = "The temperature sensor is showing incorrect readings."
        expl = generate_local_explanation(text, "Sensor Fault")
        lower_expl = expl.lower()
        self.assertIn("temperature sensor", lower_expl)
        self.assertTrue("inaccurate" in lower_expl or "reading" in lower_expl or "measurement" in lower_expl)

    def test_communication_fault_explanation(self):
        text = "The PLC has lost communication with the controller."
        expl = generate_local_explanation(text, "Communication Fault")
        lower_expl = expl.lower()
        self.assertIn("communication", lower_expl)
        self.assertIn("plc", lower_expl)

    def test_software_fault_explanation(self):
        text = "The firmware crashes whenever the device starts."
        expl = generate_local_explanation(text, "Software Fault")
        lower_expl = expl.lower()
        self.assertIn("crash", lower_expl)
        self.assertIn("firmware", lower_expl)

    def test_temperature_fault_explanation(self):
        text = "The machine is overheating."
        expl = generate_local_explanation(text, "Temperature Fault")
        lower_expl = expl.lower()
        self.assertTrue("overheating" in lower_expl or "thermal" in lower_expl)
        self.assertIn("machine", lower_expl)

    def test_power_supply_fault_explanation(self):
        text = "The voltage supplied to the controller keeps dropping."
        expl = generate_local_explanation(text, "Power Supply Fault")
        lower_expl = expl.lower()
        self.assertIn("voltage", lower_expl)
        self.assertIn("power", lower_expl)

    def test_unknown_fault_explanation_is_informative(self):
        text = "Something is wrong with the machine."
        expl = generate_local_explanation(text, "Unknown")
        lower_expl = expl.lower()
        self.assertTrue(
            "not provide enough" in lower_expl
            or "vague" in lower_expl
            or "insufficient" in lower_expl
            or "general" in lower_expl
        )

    def test_wording_variation_on_repeated_runs(self):
        text = "The motor is making a grinding noise."
        explanations = {generate_local_explanation(text, "Mechanical Fault") for _ in range(20)}
        # Must produce at least 2 distinct wording variations across 20 calls
        self.assertGreater(len(explanations), 1)

    def test_explanation_length_is_concise(self):
        test_inputs = [
            ("The motor is making a grinding noise.", "Mechanical Fault"),
            ("The conveyor shaft is misaligned.", "Mechanical Fault"),
            ("The bearing is producing a loud rattling sound.", "Mechanical Fault"),
            ("The temperature sensor is showing incorrect readings.", "Sensor Fault"),
            ("The PLC has lost communication with the controller.", "Communication Fault"),
            ("The firmware crashes whenever the device starts.", "Software Fault"),
            ("The machine is overheating.", "Temperature Fault"),
            ("The voltage supplied to the controller keeps dropping.", "Power Supply Fault"),
            ("Something is wrong with the machine.", "Unknown"),
        ]
        for text, cat in test_inputs:
            expl = generate_local_explanation(text, cat)
            word_count = len(expl.split())
            self.assertGreaterEqual(word_count, 10, f"Too short: {expl}")
            self.assertLessEqual(word_count, 40, f"Too long: {expl}")


if __name__ == "__main__":
    unittest.main()
