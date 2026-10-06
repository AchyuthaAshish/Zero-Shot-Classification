"""Unit and Integration Tests for Phase 5 Step 5.3: Employee Role Enactment.

Verifies:
1. Default employee role assignment in profile schema and repository
2. Profile role retrieval via GET /api/v1/auth/profile
3. AuthenticatedUser domain model and /auth/me contain authoritative role
4. require_employee authorization dependency succeeds for valid employee
5. Missing or invalid authentication fails with standardized HTTP 401 Unauthorized
6. Missing application profile fails safely with HTTP 403 Forbidden
7. Unsupported or non-employee role is rejected with HTTP 403 Forbidden
8. User cannot self-promote role through PATCH /api/v1/auth/profile (extra field forbidden / server-controlled)
9. Existing auth tests remain passing
10. Existing API tests remain passing
11. Existing classification behavior remains unchanged
12. Existing history behavior remains unchanged
"""

import time
import unittest
from unittest.mock import MagicMock, patch
import jwt
from fastapi import Depends, APIRouter
from fastapi.testclient import TestClient

from api.main import create_app
from api.dependencies import require_employee, get_current_user
from api.schemas.auth import AuthenticatedUser, UserResponse, UserResponseData
from api.schemas.profile import ProfileData, ProfileResponse, ProfileUpdateRequest
from api.errors import ForbiddenError, UnauthorizedError
from database.supabase_client import (
    validate_profile_payload,
    get_profile,
    create_profile,
    update_profile,
    get_or_create_profile,
    DatabaseError,
)


class TestEmployeeRoleEnactment(unittest.TestCase):
    """Test suite for Phase 5 Step 5.3 Employee Role Enactment."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()

        # Add a test-only route protected by require_employee to explicitly test the dependency
        test_router = APIRouter(prefix="/test-auth-role")

        @test_router.get("/employee-only")
        def employee_only_endpoint(user: AuthenticatedUser = Depends(require_employee)):
            return {"success": True, "message": "Employee authorized", "user_id": user.user_id, "role": user.role}

        cls.app.include_router(test_router)
        cls.client = TestClient(cls.app, raise_server_exceptions=False)

        cls.mock_jwt_secret = "test_supabase_mock_jwt_secret_xyz789_32bytes_min_length"
        cls.emp_user_id = "550e8400-e29b-41d4-a716-446655440010"
        cls.emp_email = "employee.worker@industrial.example.com"

    def _create_mock_jwt(self, user_id: str, email: str, expires_in: int = 3600) -> str:
        """Helper generating a mock HS256 signed JWT for an authenticated user."""
        now = int(time.time())
        payload = {
            "sub": user_id,
            "email": email,
            "iat": now,
            "exp": now + expires_in,
            "user_metadata": {"full_name": "Test Employee"}
        }
        return jwt.encode(payload, self.mock_jwt_secret, algorithm="HS256")

    # -------------------------------------------------------------------------
    # Test 1: Default Employee Role
    # -------------------------------------------------------------------------
    def test_default_employee_role_in_payload_validation(self):
        """1. Verifies validate_profile_payload sets role to 'employee' by default."""
        payload = validate_profile_payload(
            user_id=self.emp_user_id,
            display_name="Line Operator",
            email=self.emp_email
        )
        self.assertEqual(payload["role"], "employee")
        self.assertEqual(payload["id"], self.emp_user_id)
        self.assertEqual(payload["display_name"], "Line Operator")

    def test_validate_profile_payload_accepts_explicit_employee_role(self):
        """1b. Verifies validate_profile_payload accepts explicit role='employee'."""
        payload = validate_profile_payload(
            user_id=self.emp_user_id,
            display_name="Line Operator",
            email=self.emp_email,
            role="employee"
        )
        self.assertEqual(payload["role"], "employee")

    # -------------------------------------------------------------------------
    # Test 2: Profile Role Retrieval
    # -------------------------------------------------------------------------
    def test_profile_role_retrieval_endpoint(self):
        """2. Verifies GET /api/v1/auth/profile returns the authoritative role."""
        token = self._create_mock_jwt(self.emp_user_id, self.emp_email)
        mock_db_record = {
            "id": self.emp_user_id,
            "display_name": "Jane Technician",
            "email": self.emp_email,
            "role": "employee",
            "created_at": "2026-10-06T10:00:00Z",
            "updated_at": "2026-10-06T10:00:00Z"
        }

        with patch("api.auth.get_settings") as mock_settings, \
             patch("persistence.repository.is_supabase_configured", return_value=True), \
             patch("persistence.repository.get_profile", return_value=mock_db_record):
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.get(
                "/api/v1/auth/profile",
                headers={"Authorization": f"Bearer {token}"}
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["data"]["role"], "employee")
            self.assertEqual(data["data"]["display_name"], "Jane Technician")

    # -------------------------------------------------------------------------
    # Test 3: Authenticated User Contains Role (/auth/me)
    # -------------------------------------------------------------------------
    def test_authenticated_user_contains_role_in_auth_me(self):
        """3. Verifies GET /api/v1/auth/me returns the authoritative application role."""
        token = self._create_mock_jwt(self.emp_user_id, self.emp_email)
        mock_db_record = {
            "id": self.emp_user_id,
            "display_name": "Jane Technician",
            "email": self.emp_email,
            "role": "employee"
        }

        with patch("api.auth.get_settings") as mock_settings, \
             patch("database.supabase_client.is_supabase_configured", return_value=True), \
             patch("database.supabase_client.get_profile", return_value=mock_db_record):
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["success"])
            self.assertEqual(payload["data"]["role"], "employee")
            self.assertEqual(payload["data"]["user_id"], self.emp_user_id)
            self.assertTrue(payload["data"]["authenticated"])

    # -------------------------------------------------------------------------
    # Test 4: Employee Authorization Succeeds
    # -------------------------------------------------------------------------
    def test_employee_authorization_succeeds(self):
        """4. Verifies require_employee dependency allows access when role is employee."""
        token = self._create_mock_jwt(self.emp_user_id, self.emp_email)
        mock_db_record = {
            "id": self.emp_user_id,
            "display_name": "Jane Technician",
            "email": self.emp_email,
            "role": "employee"
        }

        with patch("api.auth.get_settings") as mock_settings, \
             patch("persistence.repository.is_supabase_configured", return_value=True), \
             patch("persistence.repository.get_profile", return_value=mock_db_record), \
             patch("database.supabase_client.is_supabase_configured", return_value=True), \
             patch("database.supabase_client.get_profile", return_value=mock_db_record):
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.get(
                "/test-auth-role/employee-only",
                headers={"Authorization": f"Bearer {token}"}
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["role"], "employee")
            self.assertEqual(data["user_id"], self.emp_user_id)

    # -------------------------------------------------------------------------
    # Test 5: Missing Authentication Fails
    # -------------------------------------------------------------------------
    def test_missing_authentication_fails(self):
        """5. Verifies unauthenticated request to require_employee yields HTTP 401."""
        response = self.client.get("/test-auth-role/employee-only")
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error"]["code"], "UNAUTHORIZED")
        self.assertIn("WWW-Authenticate", response.headers)

    def test_invalid_token_authentication_fails(self):
        """5b. Verifies invalid token to require_employee yields HTTP 401."""
        response = self.client.get(
            "/test-auth-role/employee-only",
            headers={"Authorization": "Bearer not-a-valid-token"}
        )
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error"]["code"], "UNAUTHORIZED")

    # -------------------------------------------------------------------------
    # Test 6: Missing Profile Fails Safely
    # -------------------------------------------------------------------------
    def test_missing_profile_fails_safely(self):
        """6. Verifies authenticated user without a profile is rejected with HTTP 403."""
        token = self._create_mock_jwt(self.emp_user_id, self.emp_email)

        with patch("api.auth.get_settings") as mock_settings, \
             patch("persistence.repository.is_supabase_configured", return_value=True), \
             patch("persistence.repository.get_profile", return_value=None), \
             patch("database.supabase_client.is_supabase_configured", return_value=True), \
             patch("database.supabase_client.get_profile", return_value=None):
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.get(
                "/test-auth-role/employee-only",
                headers={"Authorization": f"Bearer {token}"}
            )
            self.assertEqual(response.status_code, 403)
            data = response.json()
            self.assertFalse(data["success"])
            self.assertEqual(data["error"]["code"], "FORBIDDEN")
            self.assertIn("profile", data["error"]["message"].lower())

    # -------------------------------------------------------------------------
    # Test 7: Unsupported Role is Rejected
    # -------------------------------------------------------------------------
    def test_unsupported_role_rejected_in_validation(self):
        """7. Verifies validate_profile_payload rejects non-employee roles."""
        with self.assertRaises(ValueError) as ctx:
            validate_profile_payload(
                user_id=self.emp_user_id,
                display_name="Admin User",
                role="admin"
            )
        self.assertIn("Only 'employee' is currently supported", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            validate_profile_payload(
                user_id=self.emp_user_id,
                display_name="Reviewer User",
                role="reviewer"
            )
        self.assertIn("Only 'employee' is currently supported", str(ctx.exception))

    def test_non_employee_role_rejected_by_require_employee(self):
        """7b. Verifies require_employee rejects user if profile role is not employee."""
        token = self._create_mock_jwt(self.emp_user_id, self.emp_email)
        mock_db_record = {
            "id": self.emp_user_id,
            "display_name": "Suspicious User",
            "role": "intruder"
        }

        with patch("api.auth.get_settings") as mock_settings, \
             patch("persistence.repository.is_supabase_configured", return_value=True), \
             patch("persistence.repository.get_profile", return_value=mock_db_record), \
             patch("database.supabase_client.is_supabase_configured", return_value=True), \
             patch("database.supabase_client.get_profile", return_value=mock_db_record):
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.get(
                "/test-auth-role/employee-only",
                headers={"Authorization": f"Bearer {token}"}
            )
            self.assertEqual(response.status_code, 403)
            data = response.json()
            self.assertFalse(data["success"])
            self.assertEqual(data["error"]["code"], "FORBIDDEN")

    # -------------------------------------------------------------------------
    # Test 8: User Cannot Self-Promote Through PATCH /profile
    # -------------------------------------------------------------------------
    def test_user_cannot_self_promote_through_patch_profile(self):
        """8. Verifies submitting role in PATCH /profile is rejected with HTTP 422."""
        token = self._create_mock_jwt(self.emp_user_id, self.emp_email)

        with patch("api.auth.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            # Attempting to escalate role to admin in PATCH body
            response = self.client.patch(
                "/api/v1/auth/profile",
                headers={"Authorization": f"Bearer {token}"},
                json={"display_name": "Hacked Name", "role": "admin"}
            )
            self.assertEqual(response.status_code, 422)
            data = response.json()
            self.assertFalse(data["success"])
            self.assertEqual(data["error"]["code"], "VALIDATION_ERROR")

    def test_legitimate_patch_profile_preserves_role(self):
        """8b. Verifies legitimate PATCH /profile updates display_name while preserving role."""
        token = self._create_mock_jwt(self.emp_user_id, self.emp_email)
        mock_updated = {
            "id": self.emp_user_id,
            "display_name": "Updated Tech",
            "email": self.emp_email,
            "role": "employee",
            "created_at": "2026-10-06T10:00:00Z",
            "updated_at": "2026-10-06T10:05:00Z"
        }

        with patch("api.auth.get_settings") as mock_settings, \
             patch("persistence.repository.is_supabase_configured", return_value=True), \
             patch("persistence.repository.update_profile", return_value=mock_updated):
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.patch(
                "/api/v1/auth/profile",
                headers={"Authorization": f"Bearer {token}"},
                json={"display_name": "Updated Tech"}
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["data"]["display_name"], "Updated Tech")
            self.assertEqual(data["data"]["role"], "employee")

    # -------------------------------------------------------------------------
    # Test 9: Existing Auth Tests / Public Compatibility
    # -------------------------------------------------------------------------
    def test_existing_auth_profile_get_unauthenticated(self):
        """9. Verifies GET /auth/profile rejects unauthenticated requests with HTTP 401."""
        response = self.client.get("/api/v1/auth/profile")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "UNAUTHORIZED")

    # -------------------------------------------------------------------------
    # Test 10: Existing API Tests Remain Passing
    # -------------------------------------------------------------------------
    def test_existing_api_endpoints_remain_accessible(self):
        """10. Verifies /health and /api/v1/status remain accessible without authentication."""
        resp_health = self.client.get("/health")
        self.assertEqual(resp_health.status_code, 200)

        resp_status = self.client.get("/api/v1/status")
        self.assertEqual(resp_status.status_code, 200)

    # -------------------------------------------------------------------------
    # Test 11: Existing Classification Behavior Unchanged
    # -------------------------------------------------------------------------
    def test_existing_classification_behavior_unchanged(self):
        """11. Verifies POST /api/v1/classify continues to function without mandatory auth."""
        response = self.client.post(
            "/api/v1/classify",
            json={
                "description": "Excessive motor vibration and abnormal shaft chatter during test cycle.",
                "mode": "local"
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("category", data["data"])

    # -------------------------------------------------------------------------
    # Test 12: Existing History Behavior Unchanged
    # -------------------------------------------------------------------------
    def test_existing_history_behavior_unchanged(self):
        """12. Verifies GET /api/v1/history remains accessible and unblocked."""
        with patch("persistence.repository.is_supabase_configured", return_value=False):
            response = self.client.get("/api/v1/history")
            self.assertIn(response.status_code, (200, 503))

    # -------------------------------------------------------------------------
    # Test 13: OpenAPI Schema Documentation
    # -------------------------------------------------------------------------
    def test_openapi_schema_documents_employee_roles(self):
        """13. Verifies OpenAPI schema documents role field in ProfileData and UserResponseData."""
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        schema = response.json()
        components = schema.get("components", {}).get("schemas", {})

        self.assertIn("ProfileData", components)
        profile_properties = components["ProfileData"].get("properties", {})
        self.assertIn("role", profile_properties)
        self.assertEqual(profile_properties["role"].get("default"), "employee")

        self.assertIn("UserResponseData", components)
        user_properties = components["UserResponseData"].get("properties", {})
        self.assertIn("role", user_properties)
        self.assertEqual(user_properties["role"].get("default"), "employee")


if __name__ == "__main__":
    unittest.main()
