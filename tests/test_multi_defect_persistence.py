"""Unit Tests for Multi-Defect Persistence Layer (Phase 2 Step 2.5B).

Tests:
1. save single-defect report still works (backward compatibility)
2. multi-defect parent payload is correct
3. two child defects are generated correctly
4. three child defects are generated correctly
5. same-category child defects are allowed
6. different-category child defects are allowed
7. child defect ordering is preserved (defect_index >= 1, monotonically increasing)
8. source spans are preserved
9. confidence fields are mapped correctly (raw_score, calibrated_prob, top2_margin)
10. ambiguity fields are preserved (is_ambiguous, ambiguity_reason)
11. Unknown input does not fabricate child defects
12. INVALID validation result is rejected with clear error
13. invalid taxonomy category is rejected by child validator
14. invalid confidence bounds (< 0.0 or > 1.0) rejected by child validator
15. invalid source span (end <= start or negative) rejected by child validator
16. parent/child report_id relationship is strictly bound
17. database insert error is handled and sanitized safely
18. query helper get_defect_report_items works correctly
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.exceptions import PersistenceError
from core.schemas import (
    ClassificationResult,
    ConfidenceAssessment,
    AmbiguityAssessment,
    PerDefectClassification,
    MultiDefectClassificationResult,
    MultiDefectValidationResult,
    ValidationIssue
)
from database.supabase_client import (
    validate_defect_report_payload,
    validate_defect_item_payload,
    save_defect_report,
    save_multi_defect_report,
    get_defect_report_items,
    DatabaseError,
    MultiDefectPersistenceError,
    reset_supabase_client
)


class TestMultiDefectPersistence(unittest.TestCase):
    """Test suite for Phase 2 Step 2.5B multi-defect persistence layer."""

    def setUp(self):
        reset_supabase_client()
        self.mock_client = MagicMock()
        self.reports_table = MagicMock()
        self.items_table = MagicMock()

        def table_router(table_name):
            if table_name == "defect_reports":
                return self.reports_table
            elif table_name == "defect_report_items":
                return self.items_table
            return MagicMock()

        self.mock_client.table.side_effect = table_router

    def tearDown(self):
        reset_supabase_client()

    # 1. Single-defect backward compatibility
    def test_save_single_defect_report_still_works(self):
        """Verifies existing save_defect_report works identically as before."""
        mock_execute = MagicMock()
        mock_execute.return_value = MagicMock(data=[{
            "id": "single-uuid-001",
            "reporter_name": "Tech Alpha",
            "employee_id": "EMP-001",
            "defect_description": "Conveyor belt is slipping",
            "category": "Mechanical Fault",
            "is_multi_defect": False,
            "defect_count": 1,
            "created_at": "2026-09-30T10:00:00Z"
        }])
        self.reports_table.insert.return_value.execute = mock_execute

        res = save_defect_report(
            reporter_name="Tech Alpha",
            employee_id="EMP-001",
            defect_description="Conveyor belt is slipping",
            category="Mechanical Fault",
            confidence=0.92,
            reliability="High",
            explanation="Belt slippage detected",
            client=self.mock_client
        )

        self.assertEqual(res["id"], "single-uuid-001")
        self.assertEqual(res["category"], "Mechanical Fault")
        self.assertEqual(res["employee_id"], "EMP-001")
        # Ensure items table was not touched in legacy save
        self.items_table.insert.assert_not_called()

    # 2. Multi-defect parent payload is correct
    def test_multi_defect_parent_payload_is_correct(self):
        """Verifies multi-defect parent payload contains is_multi_defect, defect_count, and validation_status."""
        parent_id = "parent-uuid-100"
        self.reports_table.insert.return_value.execute.return_value = MagicMock(data=[{
            "id": parent_id,
            "reporter_name": "Supervisor One",
            "employee_id": "EMP-900",
            "category": "Mechanical Fault",
            "is_multi_defect": True,
            "defect_count": 2,
            "validation_status": "VALID"
        }])
        self.items_table.insert.return_value.execute.return_value = MagicMock(data=[
            {"id": "item-1", "report_id": parent_id},
            {"id": "item-2", "report_id": parent_id}
        ])

        ca1 = ConfidenceAssessment(level="High", approximate_range="Calibrated probability", raw_score=0.91, calibrated_prob=0.89, top2_margin=0.45)
        ca2 = ConfidenceAssessment(level="High", approximate_range="Calibrated probability", raw_score=0.88, calibrated_prob=0.85, top2_margin=0.40)

        d1 = PerDefectClassification(
            defect_id=1, segment_id=1, text="motor grinding noise",
            category="Mechanical Fault", confidence_assessment=ca1, reliability="High",
            explanation="Friction in motor", classification_mode="local", source_start_char=0, source_end_char=20
        )
        d2 = PerDefectClassification(
            defect_id=2, segment_id=2, text="sensor reading 0V",
            category="Sensor Fault", confidence_assessment=ca2, reliability="High",
            explanation="Zero voltage output", classification_mode="local", source_start_char=25, source_end_char=43
        )

        val_res = MultiDefectValidationResult(is_valid=True, status="VALID", validated_defect_count=2, expected_segment_count=2)
        md_res = MultiDefectClassificationResult(
            original_text="motor grinding noise and sensor reading 0V",
            is_multi_defect=True,
            defect_count=2,
            defects=[d1, d2],
            overall_status="success",
            validation_result=val_res
        )

        clf_res = ClassificationResult(
            category="Mechanical Fault",
            reason="Two independent faults detected",
            language="en",
            reliability="High",
            status="success",
            original_description="motor grinding noise and sensor reading 0V",
            normalized_description="motor grinding noise and sensor reading 0V",
            multi_defect_classification=md_res,
            validation_result=val_res
        )

        res = save_multi_defect_report(
            reporter_name="Supervisor One",
            employee_id="EMP-900",
            classification_result=clf_res,
            client=self.mock_client
        )

        self.assertEqual(res["id"], parent_id)
        parent_call_payload = self.reports_table.insert.call_args[0][0]
        self.assertTrue(parent_call_payload["is_multi_defect"])
        self.assertEqual(parent_call_payload["defect_count"], 2)
        self.assertEqual(parent_call_payload["validation_status"], "VALID")
        self.assertEqual(parent_call_payload["category"], "Mechanical Fault")

    # 3. Two child defects generated correctly
    def test_two_child_defects_generated_correctly(self):
        """Verifies 2 child defects are inserted into defect_report_items."""
        parent_id = "parent-two-defects"
        self.reports_table.insert.return_value.execute.return_value = MagicMock(data=[{"id": parent_id}])
        self.items_table.insert.return_value.execute.return_value = MagicMock(data=[
            {"id": "c1", "report_id": parent_id},
            {"id": "c2", "report_id": parent_id}
        ])

        ca = ConfidenceAssessment(level="High", approximate_range="Calibrated probability", raw_score=0.9, calibrated_prob=0.88)
        d1 = PerDefectClassification(defect_id=1, segment_id=1, text="valve leaking", category="Mechanical Fault", confidence_assessment=ca, reliability="High", explanation="Liquid escaping", classification_mode="local")
        d2 = PerDefectClassification(defect_id=2, segment_id=2, text="pressure transducer offline", category="Sensor Fault", confidence_assessment=ca, reliability="High", explanation="No response", classification_mode="local")

        md_res = MultiDefectClassificationResult(
            original_text="valve leaking and pressure transducer offline",
            is_multi_defect=True,
            defect_count=2,
            defects=[d1, d2],
            validation_result=MultiDefectValidationResult(is_valid=True, status="VALID")
        )
        clf = ClassificationResult(
            category="Mechanical Fault", reason="Multi defect", language="en", reliability="High",
            status="success", original_description="valve leaking and pressure transducer offline",
            normalized_description="valve leaking and pressure transducer offline",
            multi_defect_classification=md_res
        )

        res = save_multi_defect_report("Operator", "EMP-042", clf, client=self.mock_client)

        self.assertEqual(len(res["items"]), 2)
        items_payload = self.items_table.insert.call_args[0][0]
        self.assertEqual(len(items_payload), 2)
        self.assertEqual(items_payload[0]["defect_text"], "valve leaking")
        self.assertEqual(items_payload[1]["defect_text"], "pressure transducer offline")

    # 4. Three child defects generated correctly
    def test_three_child_defects_generated_correctly(self):
        """Verifies 3 child defects are inserted into defect_report_items."""
        parent_id = "parent-three-defects"
        self.reports_table.insert.return_value.execute.return_value = MagicMock(data=[{"id": parent_id}])
        self.items_table.insert.return_value.execute.return_value = MagicMock(data=[{"id": "c1"}, {"id": "c2"}, {"id": "c3"}])

        ca = ConfidenceAssessment(level="High", approximate_range="Calibrated", raw_score=0.85, calibrated_prob=0.82)
        d1 = PerDefectClassification(defect_id=1, segment_id=1, text="gearbox overheating", category="Temperature Fault", confidence_assessment=ca, reliability="High", explanation="Heat buildup", classification_mode="local")
        d2 = PerDefectClassification(defect_id=2, segment_id=2, text="spindle vibrating", category="Mechanical Fault", confidence_assessment=ca, reliability="High", explanation="Mechanical rattle", classification_mode="local")
        d3 = PerDefectClassification(defect_id=3, segment_id=3, text="PLC UI freeze", category="Software Fault", confidence_assessment=ca, reliability="High", explanation="Screen hang", classification_mode="local")

        md_res = MultiDefectClassificationResult(
            original_text="gearbox overheating, spindle vibrating, PLC UI freeze",
            is_multi_defect=True, defect_count=3, defects=[d1, d2, d3],
            validation_result=MultiDefectValidationResult(is_valid=True, status="VALID")
        )
        clf = ClassificationResult(
            category="Temperature Fault", reason="Three faults", language="en", reliability="High",
            status="success", original_description="gearbox overheating, spindle vibrating, PLC UI freeze",
            normalized_description="gearbox overheating, spindle vibrating, PLC UI freeze",
            multi_defect_classification=md_res
        )

        res = save_multi_defect_report("Operator", "EMP-042", clf, client=self.mock_client)
        self.assertEqual(len(res["items"]), 3)
        items_payload = self.items_table.insert.call_args[0][0]
        self.assertEqual(len(items_payload), 3)
        self.assertEqual(items_payload[0]["category"], "Temperature Fault")
        self.assertEqual(items_payload[1]["category"], "Mechanical Fault")
        self.assertEqual(items_payload[2]["category"], "Software Fault")

    # 5. Same-category child defects allowed
    def test_same_category_child_defects_allowed(self):
        """Verifies multiple defects with the exact same category (e.g. Mechanical Fault) are valid and persisted."""
        parent_id = "parent-same-cat"
        self.reports_table.insert.return_value.execute.return_value = MagicMock(data=[{"id": parent_id}])
        self.items_table.insert.return_value.execute.return_value = MagicMock(data=[{"id": "c1"}, {"id": "c2"}])

        ca = ConfidenceAssessment(level="High", approximate_range="Calibrated", raw_score=0.9, calibrated_prob=0.88)
        d1 = PerDefectClassification(defect_id=1, segment_id=1, text="motor grinding", category="Mechanical Fault", confidence_assessment=ca, reliability="High", explanation="Motor noise", classification_mode="local")
        d2 = PerDefectClassification(defect_id=2, segment_id=2, text="fan loose bolt", category="Mechanical Fault", confidence_assessment=ca, reliability="High", explanation="Loose fastener", classification_mode="local")

        md_res = MultiDefectClassificationResult(
            original_text="motor grinding and fan loose bolt",
            is_multi_defect=True, defect_count=2, defects=[d1, d2],
            validation_result=MultiDefectValidationResult(is_valid=True, status="VALID")
        )
        clf = ClassificationResult(
            category="Mechanical Fault", reason="Dual mechanical", language="en", reliability="High",
            status="success", original_description="motor grinding and fan loose bolt",
            normalized_description="motor grinding and fan loose bolt",
            multi_defect_classification=md_res
        )

        res = save_multi_defect_report("Operator", "EMP-042", clf, client=self.mock_client)
        items_payload = self.items_table.insert.call_args[0][0]
        self.assertEqual(items_payload[0]["category"], "Mechanical Fault")
        self.assertEqual(items_payload[1]["category"], "Mechanical Fault")
        self.assertEqual(items_payload[0]["defect_index"], 1)
        self.assertEqual(items_payload[1]["defect_index"], 2)

    # 6. Different-category child defects allowed
    def test_different_category_child_defects_allowed(self):
        """Verifies multiple defects with different categories are accepted."""
        parent_id = "parent-diff-cat"
        self.reports_table.insert.return_value.execute.return_value = MagicMock(data=[{"id": parent_id}])
        self.items_table.insert.return_value.execute.return_value = MagicMock(data=[{"id": "c1"}, {"id": "c2"}])

        ca = ConfidenceAssessment(level="High", approximate_range="Calibrated", raw_score=0.92, calibrated_prob=0.89)
        d1 = PerDefectClassification(defect_id=1, segment_id=1, text="fuse burnt", category="Electrical Fault", confidence_assessment=ca, reliability="High", explanation="Short circuit", classification_mode="local")
        d2 = PerDefectClassification(defect_id=2, segment_id=2, text="packet loss on CAN bus", category="Communication Fault", confidence_assessment=ca, reliability="High", explanation="Network drops", classification_mode="local")

        md_res = MultiDefectClassificationResult(
            original_text="fuse burnt and packet loss on CAN bus",
            is_multi_defect=True, defect_count=2, defects=[d1, d2],
            validation_result=MultiDefectValidationResult(is_valid=True, status="VALID")
        )
        clf = ClassificationResult(
            category="Electrical Fault", reason="Electrical + Comm", language="en", reliability="High",
            status="success", original_description="fuse burnt and packet loss on CAN bus",
            normalized_description="fuse burnt and packet loss on CAN bus",
            multi_defect_classification=md_res
        )

        save_multi_defect_report("Tech", "EMP-007", clf, client=self.mock_client)
        items_payload = self.items_table.insert.call_args[0][0]
        self.assertEqual(items_payload[0]["category"], "Electrical Fault")
        self.assertEqual(items_payload[1]["category"], "Communication Fault")

    # 7. Child defect ordering is preserved
    def test_child_defect_ordering_preserved(self):
        """Verifies defect_index reflects 1-based sequential ordering."""
        parent_id = "parent-order"
        self.reports_table.insert.return_value.execute.return_value = MagicMock(data=[{"id": parent_id}])
        self.items_table.insert.return_value.execute.return_value = MagicMock(data=[{"id": "c1"}, {"id": "c2"}, {"id": "c3"}])

        ca = ConfidenceAssessment(level="Medium", approximate_range="Calibrated", raw_score=0.8, calibrated_prob=0.75)
        defects = [
            PerDefectClassification(defect_id=1, segment_id=1, text="first fault", category="Mechanical Fault", confidence_assessment=ca, reliability="Medium", explanation="E1", classification_mode="local"),
            PerDefectClassification(defect_id=2, segment_id=2, text="second fault", category="Sensor Fault", confidence_assessment=ca, reliability="Medium", explanation="E2", classification_mode="local"),
            PerDefectClassification(defect_id=3, segment_id=3, text="third fault", category="Software Fault", confidence_assessment=ca, reliability="Medium", explanation="E3", classification_mode="local"),
        ]

        md_res = MultiDefectClassificationResult(
            original_text="first fault, second fault, third fault",
            is_multi_defect=True, defect_count=3, defects=defects,
            validation_result=MultiDefectValidationResult(is_valid=True, status="VALID")
        )
        clf = ClassificationResult(category="Mechanical Fault", reason="Three", language="en", reliability="Medium", status="success", original_description="...", normalized_description="...", multi_defect_classification=md_res)

        save_multi_defect_report("Tech", "EMP-007", clf, client=self.mock_client)
        items = self.items_table.insert.call_args[0][0]
        self.assertEqual([i["defect_index"] for i in items], [1, 2, 3])

    # 8. Source spans are preserved
    def test_source_spans_preserved(self):
        """Verifies start_char and end_char offsets are preserved."""
        payload = validate_defect_item_payload(
            report_id="uuid-1",
            defect_index=1,
            defect_text="pump bearing overheating",
            category="Temperature Fault",
            start_char=5,
            end_char=29
        )
        self.assertEqual(payload["start_char"], 5)
        self.assertEqual(payload["end_char"], 29)

    # 9. Confidence fields are mapped correctly
    def test_confidence_fields_mapped_correctly(self):
        """Verifies raw_score, calibrated_score, top2_margin, level, range are accurately captured."""
        payload = validate_defect_item_payload(
            report_id="uuid-1",
            defect_index=1,
            defect_text="voltage drop on rail",
            category="Power Supply Fault",
            confidence=0.91,
            confidence_level="High",
            confidence_range="Calibrated probability",
            raw_score=0.91,
            calibrated_score=0.88,
            top2_margin=0.42
        )
        self.assertEqual(payload["confidence"], 0.91)
        self.assertEqual(payload["confidence_level"], "High")
        self.assertEqual(payload["confidence_range"], "Calibrated probability")
        self.assertEqual(payload["raw_score"], 0.91)
        self.assertEqual(payload["calibrated_score"], 0.88)
        self.assertEqual(payload["top2_margin"], 0.42)

    # 10. Ambiguity fields are preserved
    def test_ambiguity_fields_preserved(self):
        """Verifies is_ambiguous and ambiguity_reason are preserved."""
        payload = validate_defect_item_payload(
            report_id="uuid-1",
            defect_index=1,
            defect_text="motor grinding or sensor loose",
            category="Mechanical Fault",
            is_ambiguous=True,
            ambiguity_reason="Ambiguous between Mechanical Fault and Sensor Fault"
        )
        self.assertTrue(payload["is_ambiguous"])
        self.assertEqual(payload["ambiguity_reason"], "Ambiguous between Mechanical Fault and Sensor Fault")

    # 11. Unknown input does not fabricate child defects
    def test_unknown_input_does_not_fabricate_child_defects(self):
        """Verifies non-defect or vague input resulting in Unknown does NOT create child defect records."""
        parent_id = "parent-unknown"
        self.reports_table.insert.return_value.execute.return_value = MagicMock(data=[{
            "id": parent_id,
            "category": "Unknown",
            "is_multi_defect": False,
            "defect_count": 0
        }])

        ca = ConfidenceAssessment(level="Uncertain", approximate_range="Insufficient Evidence")
        clf = ClassificationResult(
            category="Unknown",
            reason="Vague greeting with no defect symptoms",
            language="en",
            reliability="Low",
            status="unknown",
            original_description="Good morning team",
            normalized_description="good morning team",
            confidence_assessment=ca,
            multi_defect_classification=MultiDefectClassificationResult(
                original_text="Good morning team",
                is_multi_defect=False,
                defect_count=0,
                defects=[],
                overall_status="unknown"
            )
        )

        res = save_multi_defect_report("Supervisor", "EMP-001", clf, client=self.mock_client)
        self.assertEqual(res["category"], "Unknown")
        self.assertEqual(len(res["items"]), 0)
        self.items_table.insert.assert_not_called()

    # 12. INVALID validation result is rejected
    def test_invalid_validation_result_rejected(self):
        """Verifies structurally INVALID validation result is rejected without writing to database."""
        val_res = MultiDefectValidationResult(
            is_valid=False,
            status="INVALID",
            errors=[ValidationIssue(code="ERR_SEG_COUNT", severity="error", message="Segment count exceeds defects")],
            validated_defect_count=1,
            expected_segment_count=2
        )
        md_res = MultiDefectClassificationResult(
            original_text="broken shaft",
            is_multi_defect=True,
            defect_count=1,
            defects=[],
            validation_result=val_res
        )
        clf = ClassificationResult(
            category="Mechanical Fault", reason="Err", language="en", reliability="Low",
            status="validation_error", original_description="broken shaft", normalized_description="broken shaft",
            multi_defect_classification=md_res, validation_result=val_res
        )

        with self.assertRaises(MultiDefectPersistenceError) as ctx:
            save_multi_defect_report("Supervisor", "EMP-001", clf, client=self.mock_client)

        self.assertIn("INVALID", str(ctx.exception))
        # Ensure database writes were completely aborted
        self.reports_table.insert.assert_not_called()
        self.items_table.insert.assert_not_called()

    # 13. Invalid taxonomy category is rejected by child validator
    def test_invalid_taxonomy_rejected(self):
        """Verifies child item validator rejects unapproved categories."""
        with self.assertRaises(ValueError) as ctx:
            validate_defect_item_payload(
                report_id="uuid-1",
                defect_index=1,
                defect_text="broken rotor",
                category="Rotor Breakage"  # Unapproved!
            )
        self.assertIn("Invalid category", str(ctx.exception))

    # 14. Invalid confidence bounds rejected by child validator
    def test_invalid_confidence_rejected(self):
        """Verifies confidence and calibrated_score out of [0.0, 1.0] are rejected."""
        with self.assertRaises(ValueError) as ctx:
            validate_defect_item_payload(
                report_id="uuid-1",
                defect_index=1,
                defect_text="pump fault",
                category="Mechanical Fault",
                confidence=1.5  # Out of bounds!
            )
        self.assertIn("confidence must be between 0.0 and 1.0", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            validate_defect_item_payload(
                report_id="uuid-1",
                defect_index=1,
                defect_text="pump fault",
                category="Mechanical Fault",
                calibrated_score=-0.2  # Out of bounds!
            )
        self.assertIn("calibrated_score must be between 0.0 and 1.0", str(ctx2.exception))

    # 15. Invalid source span rejected by child validator
    def test_invalid_source_span_rejected(self):
        """Verifies end_char <= start_char or negative offsets are rejected."""
        with self.assertRaises(ValueError) as ctx:
            validate_defect_item_payload(
                report_id="uuid-1",
                defect_index=1,
                defect_text="pump fault",
                category="Mechanical Fault",
                start_char=20,
                end_char=10  # end <= start!
            )
        self.assertIn("end_char must be strictly greater than start_char", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            validate_defect_item_payload(
                report_id="uuid-1",
                defect_index=1,
                defect_text="pump fault",
                category="Mechanical Fault",
                start_char=-5,  # negative!
                end_char=10
            )
        self.assertIn("start_char must be >= 0", str(ctx2.exception))

    # 16. Parent/child report_id relationship is strictly bound
    def test_parent_child_report_id_relationship_correct(self):
        """Verifies report_id on each child record precisely matches the parent record ID."""
        expected_parent_id = "parent-uuid-xyz-789"
        self.reports_table.insert.return_value.execute.return_value = MagicMock(data=[{"id": expected_parent_id}])
        self.items_table.insert.return_value.execute.return_value = MagicMock(data=[
            {"id": "item-1", "report_id": expected_parent_id}
        ])

        ca = ConfidenceAssessment(level="High", approximate_range="Calibrated", raw_score=0.9, calibrated_prob=0.88)
        d = PerDefectClassification(defect_id=1, segment_id=1, text="motor rattle", category="Mechanical Fault", confidence_assessment=ca, reliability="High", explanation="Rattle", classification_mode="local")

        md_res = MultiDefectClassificationResult(
            original_text="motor rattle", is_multi_defect=False, defect_count=1, defects=[d],
            validation_result=MultiDefectValidationResult(is_valid=True, status="VALID")
        )
        clf = ClassificationResult(category="Mechanical Fault", reason="Rattle", language="en", reliability="High", status="success", original_description="motor rattle", normalized_description="motor rattle", multi_defect_classification=md_res)

        save_multi_defect_report("Supervisor", "EMP-001", clf, client=self.mock_client)
        called_items = self.items_table.insert.call_args[0][0]
        self.assertEqual(len(called_items), 1)
        self.assertEqual(called_items[0]["report_id"], expected_parent_id)

    # 17. Database insert error is handled safely
    def test_database_insert_error_handled_safely(self):
        """Verifies sanitized DatabaseError is raised if Supabase call raises an exception."""
        self.reports_table.insert.return_value.execute.return_value = MagicMock(data=[{"id": "p-1"}])
        self.items_table.insert.return_value.execute.side_effect = Exception("Supabase connection timeout to https://secret-url.supabase.co with key secret-key-xyz")

        ca = ConfidenceAssessment(level="High", approximate_range="Calibrated", raw_score=0.9, calibrated_prob=0.88)
        d = PerDefectClassification(defect_id=1, segment_id=1, text="motor rattle", category="Mechanical Fault", confidence_assessment=ca, reliability="High", explanation="Rattle", classification_mode="local")
        md_res = MultiDefectClassificationResult(original_text="motor rattle", is_multi_defect=False, defect_count=1, defects=[d], validation_result=MultiDefectValidationResult(is_valid=True, status="VALID"))
        clf = ClassificationResult(category="Mechanical Fault", reason="Rattle", language="en", reliability="High", status="success", original_description="motor rattle", normalized_description="motor rattle", multi_defect_classification=md_res)

        with self.assertRaises(DatabaseError) as ctx:
            save_multi_defect_report("Supervisor", "EMP-001", clf, client=self.mock_client)

        self.assertIn("Failed to save defect report items", str(ctx.exception))

    # 18. Query helper get_defect_report_items works correctly
    def test_query_defect_report_items(self):
        """Verifies get_defect_report_items queries child table with report_id filter and order by defect_index."""
        mock_query = MagicMock()
        mock_query.select.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_query.execute.return_value = MagicMock(data=[
            {"id": "c1", "defect_index": 1, "category": "Mechanical Fault"},
            {"id": "c2", "defect_index": 2, "category": "Sensor Fault"}
        ])
        self.items_table.select.return_value = mock_query

        items = get_defect_report_items("report-uuid-999", client=self.mock_client)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["defect_index"], 1)
        self.assertEqual(items[1]["defect_index"], 2)
        mock_query.eq.assert_called_with("report_id", "report-uuid-999")
        mock_query.order.assert_called_with("defect_index", desc=False)


if __name__ == "__main__":
    unittest.main()
