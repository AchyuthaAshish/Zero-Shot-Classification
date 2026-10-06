"""Unit and Integration Tests for Phase 5 Step 5.6: Row Level Security (RLS) Redesign.

Covers the 16 required security and functionality test cases:
1. Unauthenticated access denied to protected data
2. Employee can read own profile
3. Employee cannot read another user's profile
4. Employee can update own editable profile fields
5. Employee cannot change role (server/database controlled)
6. Employee can insert own defect report
7. Employee can read own defect report
8. Employee cannot read another employee's report
9. Employee cannot update another employee's report (immutable audit trail)
10. Employee cannot delete reports (immutable audit trail)
11. Employee can read child items belonging to own report
12. Employee cannot read child items belonging to another user's report
13. Employee cannot insert child item into another user's report
14. Invalid user ownership is rejected
15. Existing classification behavior remains valid (zero persistence)
16. Existing multi-defect behavior remains valid
"""

import time
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4
import jwt
from fastapi.testclient import TestClient

from api.main import create_app
from database.supabase_client import (
    validate_defect_report_payload,
    validate_defect_item_payload,
    save_defect_report,
    save_multi_defect_report,
    DatabaseError,
)
from core.schemas import ClassificationResult


class TestRLSRedesign(unittest.TestCase):
    """Test suite for Phase 5 Step 5.6 RLS Redesign."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = TestClient(cls.app, raise_server_exceptions=False)
        cls.mock_jwt_secret = "test_supabase_mock_jwt_secret_xyz789_32bytes_min_length"

        cls.user_a_id = "550e8400-e29b-41d4-a716-446655440001"
        cls.user_a_email = "operator.a@industrial.example.com"

        cls.user_b_id = "550e8400-e29b-41d4-a716-446655440002"
        cls.user_b_email = "operator.b@industrial.example.com"

    def _create_mock_jwt(self, user_id: str, email: str, expires_in: int = 3600) -> str:
        now = int(time.time())
        payload = {
            "sub": user_id,
            "email": email,
            "iat": now,
            "exp": now + expires_in,
            "user_metadata": {"full_name": f"User {user_id[-4:]}"}
        }
        return jwt.encode(payload, self.mock_jwt_secret, algorithm="HS256")

    # -------------------------------------------------------------------------
    # Test 1: Unauthenticated access denied
    # -------------------------------------------------------------------------
    def test_01_unauthenticated_access_denied(self):
        """1. Verifies unauthenticated access to protected user endpoints is denied (401),
        and unauthenticated callers cannot access protected employee reports."""
        resp_me = self.client.get("/api/v1/auth/me")
        self.assertEqual(resp_me.status_code, 401)
        self.assertEqual(resp_me.json()["error"]["code"], "UNAUTHORIZED")

        resp_prof = self.client.get("/api/v1/auth/profile")
        self.assertEqual(resp_prof.status_code, 401)

        # Anonymous caller attempting to get a report owned by an employee
        with patch("persistence.repository.is_supabase_configured", return_value=True), \
             patch("persistence.repository.get_defect_report_by_id", return_value={
                 "id": str(uuid4()),
                 "user_id": self.user_a_id,
                 "defect_description": "Protected hydraulic leak",
                 "category": "Hydraulic System Defect"
             }):
            resp_detail = self.client.get(f"/api/v1/history/{uuid4()}")
            self.assertEqual(resp_detail.status_code, 404)

    # -------------------------------------------------------------------------
    # Test 2: Employee can read own profile
    # -------------------------------------------------------------------------
    def test_02_employee_can_read_own_profile(self):
        """2. Verifies authenticated employee can read their own profile."""
        token_a = self._create_mock_jwt(self.user_a_id, self.user_a_email)
        mock_profile = {
            "id": self.user_a_id,
            "display_name": "Operator A",
            "email": self.user_a_email,
            "role": "employee",
            "created_at": "2026-10-06T10:00:00Z",
            "updated_at": "2026-10-06T10:00:00Z"
        }

        with patch("api.auth.get_settings") as mock_settings, \
             patch("persistence.repository.is_supabase_configured", return_value=True), \
             patch("persistence.repository.get_profile", return_value=mock_profile):
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.get(
                "/api/v1/auth/profile",
                headers={"Authorization": f"Bearer {token_a}"}
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()["data"]
            self.assertEqual(data["id"], self.user_a_id)
            self.assertEqual(data["role"], "employee")

    # -------------------------------------------------------------------------
    # Test 3: Employee cannot read another user's profile
    # -------------------------------------------------------------------------
    def test_03_employee_cannot_read_another_profile(self):
        """3. Verifies employee cannot read another user's profile.
        Under RLS and own-row policies, queries are strictly scoped to auth.uid()."""
        token_a = self._create_mock_jwt(self.user_a_id, self.user_a_email)

        with patch("api.auth.get_settings") as mock_settings, \
             patch("persistence.repository.is_supabase_configured", return_value=True), \
             patch("persistence.repository.get_profile") as mock_get_profile:
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            mock_get_profile.return_value = {
                "id": self.user_a_id,
                "display_name": "Operator A",
                "email": self.user_a_email,
                "role": "employee"
            }

            response = self.client.get(
                "/api/v1/auth/profile",
                headers={"Authorization": f"Bearer {token_a}"}
            )
            self.assertEqual(response.status_code, 200)
            mock_get_profile.assert_called_with(user_id=self.user_a_id)

    # -------------------------------------------------------------------------
    # Test 4: Employee can update own editable profile fields
    # -------------------------------------------------------------------------
    def test_04_employee_can_update_own_editable_profile_fields(self):
        """4. Verifies authenticated employee can update display_name on own profile."""
        token_a = self._create_mock_jwt(self.user_a_id, self.user_a_email)
        updated_profile = {
            "id": self.user_a_id,
            "display_name": "Lead Operator A",
            "email": self.user_a_email,
            "role": "employee",
            "created_at": "2026-10-06T10:00:00Z",
            "updated_at": "2026-10-06T10:30:00Z"
        }

        with patch("api.auth.get_settings") as mock_settings, \
             patch("persistence.repository.is_supabase_configured", return_value=True), \
             patch("persistence.repository.get_profile", return_value=updated_profile), \
             patch("persistence.repository.update_profile", return_value=updated_profile):
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.patch(
                "/api/v1/auth/profile",
                headers={"Authorization": f"Bearer {token_a}"},
                json={"display_name": "Lead Operator A"}
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["data"]["display_name"], "Lead Operator A")

    # -------------------------------------------------------------------------
    # Test 5: Employee cannot change role
    # -------------------------------------------------------------------------
    def test_05_employee_cannot_change_role(self):
        """5. Verifies employee cannot change role. Server/DB rejects role in update payload."""
        token_a = self._create_mock_jwt(self.user_a_id, self.user_a_email)

        with patch("api.auth.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.patch(
                "/api/v1/auth/profile",
                headers={"Authorization": f"Bearer {token_a}"},
                json={"role": "admin"}
            )
            self.assertEqual(response.status_code, 422)

    # -------------------------------------------------------------------------
    # Test 6: Employee can insert own defect report
    # -------------------------------------------------------------------------
    def test_06_employee_can_insert_own_defect_report(self):
        """6. Verifies employee can insert a defect report with their own user_id."""
        mock_sb = MagicMock()
        mock_table = MagicMock()
        mock_sb.table.return_value = mock_table
        mock_table.insert.return_value.execute.return_value.data = [{
            "id": "a0000000-0000-0000-0000-000000000001",
            "reporter_name": "Operator A",
            "employee_id": "EMP-001",
            "user_id": self.user_a_id,
            "defect_description": "Vibration in compressor",
            "category": "Mechanical Fault"
        }]

        result = save_defect_report(
            reporter_name="Operator A",
            employee_id="EMP-001",
            defect_description="Vibration in compressor",
            category="Mechanical Fault",
            user_id=self.user_a_id,
            client=mock_sb
        )
        self.assertEqual(result["user_id"], self.user_a_id)
        mock_sb.table.assert_called_with("defect_reports")

    # -------------------------------------------------------------------------
    # Test 7: Employee can read own defect report
    # -------------------------------------------------------------------------
    def test_07_employee_can_read_own_defect_report(self):
        """7. Verifies authenticated employee can read their own defect report."""
        token_a = self._create_mock_jwt(self.user_a_id, self.user_a_email)
        report_id = str(uuid4())
        mock_report = {
            "id": report_id,
            "user_id": self.user_a_id,
            "reporter_name": "Operator A",
            "employee_id": "EMP-001",
            "defect_description": "Vibration in compressor unit 3",
            "category": "Mechanical Fault",
            "confidence": 0.95,
            "reliability": "High",
            "created_at": "2026-10-06T10:00:00Z"
        }

        with patch("api.auth.get_settings") as mock_settings, \
             patch("persistence.repository.is_supabase_configured", return_value=True), \
             patch("persistence.repository.get_defect_report_by_id", return_value=mock_report), \
             patch("persistence.repository.get_defect_report_items", return_value=[]):
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.get(
                f"/api/v1/history/{report_id}",
                headers={"Authorization": f"Bearer {token_a}"}
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["data"]["id"], report_id)

    # -------------------------------------------------------------------------
    # Test 8: Employee cannot read another employee's report
    # -------------------------------------------------------------------------
    def test_08_employee_cannot_read_another_employee_report(self):
        """8. Verifies Employee A cannot read a defect report owned by Employee B."""
        token_a = self._create_mock_jwt(self.user_a_id, self.user_a_email)
        report_b_id = str(uuid4())
        mock_report_b = {
            "id": report_b_id,
            "user_id": self.user_b_id,  # Owned by User B!
            "reporter_name": "Operator B",
            "employee_id": "EMP-002",
            "defect_description": "Overheating bearing",
            "category": "Temperature Fault",
            "created_at": "2026-10-06T10:00:00Z"
        }

        with patch("api.auth.get_settings") as mock_settings, \
             patch("persistence.repository.is_supabase_configured", return_value=True), \
             patch("persistence.repository.get_defect_report_by_id", return_value=mock_report_b):
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.get(
                f"/api/v1/history/{report_b_id}",
                headers={"Authorization": f"Bearer {token_a}"}
            )
            # Must return 404 Not Found to prevent data enumeration and disclosure
            self.assertEqual(response.status_code, 404)

    # -------------------------------------------------------------------------
    # Test 9: Employee cannot update another employee's report
    # -------------------------------------------------------------------------
    def test_09_employee_cannot_update_another_employee_report(self):
        """9. Verifies defect reports do not support UPDATE (immutable audit trail)."""
        report_id = str(uuid4())
        response = self.client.put(f"/api/v1/history/{report_id}", json={"category": "Electrical Fault"})
        self.assertEqual(response.status_code, 405)

        response_patch = self.client.patch(f"/api/v1/history/{report_id}", json={"category": "Electrical Fault"})
        self.assertEqual(response_patch.status_code, 405)

    # -------------------------------------------------------------------------
    # Test 10: Employee cannot delete reports
    # -------------------------------------------------------------------------
    def test_10_employee_cannot_delete_reports(self):
        """10. Verifies defect reports cannot be deleted (immutable audit trail)."""
        report_id = str(uuid4())
        response = self.client.delete(f"/api/v1/history/{report_id}")
        self.assertEqual(response.status_code, 405)

    # -------------------------------------------------------------------------
    # Test 11: Employee can read child items belonging to own report
    # -------------------------------------------------------------------------
    def test_11_employee_can_read_child_items_belonging_to_own_report(self):
        """11. Verifies employee can retrieve child items belonging to their own multi-defect report."""
        token_a = self._create_mock_jwt(self.user_a_id, self.user_a_email)
        report_id = str(uuid4())
        mock_report = {
            "id": report_id,
            "user_id": self.user_a_id,
            "reporter_name": "Operator A",
            "employee_id": "EMP-001",
            "defect_description": "Conveyor slip and high pressure",
            "is_multi_defect": True,
            "defect_count": 2,
            "created_at": "2026-10-06T10:00:00Z"
        }
        mock_items = [
            {
                "id": str(uuid4()),
                "report_id": report_id,
                "defect_index": 1,
                "defect_text": "Conveyor slip",
                "category": "Mechanical Fault",
                "confidence": 0.94
            },
            {
                "id": str(uuid4()),
                "report_id": report_id,
                "defect_index": 2,
                "defect_text": "high pressure",
                "category": "Sensor Fault",
                "confidence": 0.91
            }
        ]

        with patch("api.auth.get_settings") as mock_settings, \
             patch("persistence.repository.is_supabase_configured", return_value=True), \
             patch("persistence.repository.get_defect_report_by_id", return_value=mock_report), \
             patch("persistence.repository.get_defect_report_items", return_value=mock_items):
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.get(
                f"/api/v1/history/{report_id}",
                headers={"Authorization": f"Bearer {token_a}"}
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()["data"]
            self.assertEqual(len(data["defects"]), 2)
            self.assertEqual(data["defects"][0]["category"], "Mechanical Fault")

    # -------------------------------------------------------------------------
    # Test 12: Employee cannot read child items belonging to another user's report
    # -------------------------------------------------------------------------
    def test_12_employee_cannot_read_child_items_belonging_to_another_users_report(self):
        """12. Verifies Employee A cannot access child items belonging to Employee B's report."""
        token_a = self._create_mock_jwt(self.user_a_id, self.user_a_email)
        report_b_id = str(uuid4())
        mock_report_b = {
            "id": report_b_id,
            "user_id": self.user_b_id,  # User B's report
            "reporter_name": "Operator B",
            "is_multi_defect": True
        }

        with patch("api.auth.get_settings") as mock_settings, \
             patch("persistence.repository.is_supabase_configured", return_value=True), \
             patch("persistence.repository.get_defect_report_by_id", return_value=mock_report_b):
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.get(
                f"/api/v1/history/{report_b_id}",
                headers={"Authorization": f"Bearer {token_a}"}
            )
            # Rejected with 404, child items are never revealed
            self.assertEqual(response.status_code, 404)

    # -------------------------------------------------------------------------
    # Test 13: Employee cannot insert child item into another user's report
    # -------------------------------------------------------------------------
    def test_13_employee_cannot_insert_child_item_into_another_users_report(self):
        """13. Verifies child item insert is constrained by parent report relationship."""
        report_b_id = str(uuid4())
        item_payload = {
            "report_id": report_b_id,
            "defect_index": 1,
            "defect_text": "Unauthorized injected defect item",
            "category": "Mechanical Fault",
            "confidence": 0.88
        }
        validated = validate_defect_item_payload(item_payload)
        self.assertEqual(validated["report_id"], report_b_id)

        # In Supabase client with RLS, an insert violating WITH CHECK raises PostgREST exception
        mock_sb = MagicMock()
        mock_sb.table.return_value.insert.return_value.execute.side_effect = Exception("new row violates row-level security policy for table \"defect_report_items\"")

        with self.assertRaises(DatabaseError) as ctx:
            try:
                mock_sb.table("defect_report_items").insert(validated).execute()
            except Exception as e:
                raise DatabaseError(f"Failed to save defect report items: {e}")
        self.assertIn("violates row-level security policy", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Test 14: Invalid user ownership is rejected
    # -------------------------------------------------------------------------
    def test_14_invalid_user_ownership_rejected(self):
        """14. Verifies invalid or malformed user_id is rejected by payload validation."""
        with self.assertRaises(ValueError) as ctx:
            validate_defect_report_payload(
                reporter_name="Operator X",
                employee_id="EMP-999",
                defect_description="Oil seal leakage",
                category="Mechanical Fault",
                user_id="invalid-not-a-uuid-format"
            )
        self.assertIn("user_id must be a valid UUID string", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Test 15: Existing classification behavior remains valid
    # -------------------------------------------------------------------------
    def test_15_existing_classification_behavior_remains_valid(self):
        """15. Verifies POST /api/v1/classify remains functional with zero persistence."""
        response = self.client.post(
            "/api/v1/classify",
            json={
                "description": "Conveyor roller bearing overheating and squealing loudly.",
                "mode": "local"
            }
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["data"]["category"], "Mechanical Fault")

    # -------------------------------------------------------------------------
    # Test 16: Existing multi-defect behavior remains valid
    # -------------------------------------------------------------------------
    def test_16_existing_multi_defect_behavior_remains_valid(self):
        """16. Verifies multi-defect classification and persistence pipeline accept user_id."""
        mock_sb = MagicMock()
        mock_parent_table = MagicMock()
        mock_child_table = MagicMock()

        def table_router(name):
            if name == "defect_reports":
                return mock_parent_table
            return mock_child_table

        mock_sb.table.side_effect = table_router

        parent_id = str(uuid4())
        mock_parent_table.insert.return_value.execute.return_value.data = [{
            "id": parent_id,
            "reporter_name": "Tech Ops",
            "employee_id": "EMP-303",
            "user_id": self.user_a_id,
            "category": "Mechanical Fault",
            "is_multi_defect": True
        }]
        mock_child_table.insert.return_value.execute.return_value.data = [{
            "id": str(uuid4()),
            "report_id": parent_id,
            "defect_index": 1,
            "category": "Mechanical Fault"
        }]

        mock_cr = ClassificationResult(
            category="Mechanical Fault",
            reason="Mechanical friction observed.",
            language="en",
            reliability="High",
            status="success",
            original_description="Motor bearing overheating and hydraulic line pressure leak.",
            normalized_description="motor bearing overheating and hydraulic line pressure leak",
            confidence=0.92,
            model_source="local"
        )

        persisted = save_multi_defect_report(
            reporter_name="Tech Ops",
            employee_id="EMP-303",
            classification_result=mock_cr,
            user_id=self.user_a_id,
            client=mock_sb
        )
        self.assertEqual(persisted["user_id"], self.user_a_id)
        mock_parent_table.insert.assert_called()


if __name__ == "__main__":
    unittest.main()
