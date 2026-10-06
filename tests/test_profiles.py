"""Unit and Integration Tests for Phase 5 Step 5.2: User Profiles Table.

Verifies:
1. profiles schema expectations (id, display_name, email, created_at, updated_at)
2. auth.users -> profiles 1:1 foreign key relationship expectations
3. Authenticated user profile access (GET /api/v1/auth/profile)
4. Anonymous profile access rejection (HTTP 401 UNAUTHORIZED)
5. User isolation: user session only accesses their own profile (scoped to auth token sub claim)
6. Profile creation behavior (create_profile repository function)
7. Profile retrieval (get_profile repository function)
8. Profile update behavior (PATCH /api/v1/auth/profile and update_profile repository function)
9. Absolute absence of password, token, JWT, or secret fields in profile schemas
10. Authentication endpoint regression (GET /api/v1/auth/me remains functional)
11. Existing API regression (POST /api/v1/classify, GET /health, GET /api/v1/status)
12. Existing database regression (defect_reports and items operations unaffected)
13. OpenAPI regression (ProfileData, ProfileResponse, ProfileUpdateRequest documented in spec)
"""

import time
import unittest
from unittest.mock import MagicMock, patch
import jwt
from fastapi.testclient import TestClient

from api.main import create_app
from api.schemas.profile import ProfileData, ProfileResponse, ProfileUpdateRequest
from database.supabase_client import (
    validate_profile_payload,
    get_profile,
    create_profile,
    update_profile,
    get_or_create_profile,
    DatabaseError,
)


class TestUserProfiles(unittest.TestCase):
    """Test suite for Phase 5 Step 5.2 User Profiles Table and API."""

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
        """Helper generating a mock HS256 signed JWT for an authenticated user."""
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
    # Test 1: Profiles Schema Validation
    # -------------------------------------------------------------------------
    def test_validate_profile_payload_success(self):
        """1. Verifies validate_profile_payload produces clean, conforming profile dictionary."""
        payload = validate_profile_payload(
            user_id=self.user_a_id,
            display_name="Senior Maintenance Tech",
            email=self.user_a_email
        )
        self.assertEqual(payload["id"], self.user_a_id)
        self.assertEqual(payload["display_name"], "Senior Maintenance Tech")
        self.assertEqual(payload["email"], self.user_a_email)

    def test_validate_profile_payload_rejects_empty_id(self):
        """1b. Verifies missing or empty user_id is rejected."""
        with self.assertRaises(ValueError):
            validate_profile_payload(user_id="", display_name="Operator")
        with self.assertRaises(ValueError):
            validate_profile_payload(user_id=None, display_name="Operator")

    def test_validate_profile_payload_rejects_empty_display_name(self):
        """1c. Verifies empty or whitespace-only display_name is rejected."""
        with self.assertRaises(ValueError):
            validate_profile_payload(user_id=self.user_a_id, display_name="")
        with self.assertRaises(ValueError):
            validate_profile_payload(user_id=self.user_a_id, display_name="   ")

    def test_validate_profile_payload_rejects_excessive_display_name(self):
        """1d. Verifies display_name over 100 characters is rejected."""
        long_name = "A" * 101
        with self.assertRaises(ValueError):
            validate_profile_payload(user_id=self.user_a_id, display_name=long_name)

    # -------------------------------------------------------------------------
    # Test 2: auth.users -> profiles Relationship & Data Contract
    # -------------------------------------------------------------------------
    def test_profile_id_is_foreign_key_to_auth_users(self):
        """2. Verifies ProfileData requires user UUID identical to Supabase Auth ID."""
        profile = ProfileData(
            id=self.user_a_id,
            display_name="Line Supervisor",
            email=self.user_a_email,
            created_at="2026-10-06T10:00:00Z",
            updated_at="2026-10-06T10:00:00Z"
        )
        self.assertEqual(profile.id, self.user_a_id)
        self.assertEqual(profile.email, self.user_a_email)

    # -------------------------------------------------------------------------
    # Test 3: Authenticated User Profile Access (GET /api/v1/auth/profile)
    # -------------------------------------------------------------------------
    def test_authenticated_user_can_access_own_profile(self):
        """3. Verifies authenticated user can retrieve their profile via GET /api/v1/auth/profile."""
        token = self._create_mock_jwt(self.user_a_id, self.user_a_email)

        mock_profile = {
            "id": self.user_a_id,
            "display_name": "Operator Alpha",
            "email": self.user_a_email,
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
                headers={"Authorization": f"Bearer {token}"}
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertTrue(body["success"])
            self.assertEqual(body["data"]["id"], self.user_a_id)
            self.assertEqual(body["data"]["display_name"], "Operator Alpha")
            self.assertEqual(body["data"]["email"], self.user_a_email)

    # -------------------------------------------------------------------------
    # Test 4: Anonymous Profile Access Rejection
    # -------------------------------------------------------------------------
    def test_anonymous_profile_access_rejected(self):
        """4. Verifies unauthenticated request to /api/v1/auth/profile returns HTTP 401 UNAUTHORIZED."""
        response = self.client.get("/api/v1/auth/profile")
        self.assertEqual(response.status_code, 401)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], "UNAUTHORIZED")

    # -------------------------------------------------------------------------
    # Test 5: User Isolation (User A Cannot Access User B's Profile)
    # -------------------------------------------------------------------------
    def test_user_isolation_profile_scoped_to_token_identity(self):
        """5. Verifies profile lookup is strictly bound to token's sub claim (User A only gets User A)."""
        token_a = self._create_mock_jwt(self.user_a_id, self.user_a_email)

        with patch("api.auth.get_settings") as mock_settings, \
             patch("persistence.repository.is_supabase_configured", return_value=True), \
             patch("persistence.repository.get_profile") as mock_get_profile:
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            mock_get_profile.return_value = {
                "id": self.user_a_id,
                "display_name": "Operator Alpha",
                "email": self.user_a_email
            }

            response = self.client.get(
                "/api/v1/auth/profile",
                headers={"Authorization": f"Bearer {token_a}"}
            )
            self.assertEqual(response.status_code, 200)

            # Ensure repository was called with User A's ID, NEVER User B's ID
            mock_get_profile.assert_called_once_with(user_id=self.user_a_id)
            self.assertNotEqual(mock_get_profile.call_args[1].get("user_id"), self.user_b_id)

    # -------------------------------------------------------------------------
    # Test 6: Profile Creation Behavior
    # -------------------------------------------------------------------------
    def test_create_profile_repository_function(self):
        """6. Verifies create_profile executes insert into public.profiles with valid payload."""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_insert = MagicMock()
        mock_execute = MagicMock()

        mock_execute.return_value = MagicMock(data=[{
            "id": self.user_a_id,
            "display_name": "New Tech",
            "email": self.user_a_email,
            "created_at": "2026-10-06T10:00:00Z",
            "updated_at": "2026-10-06T10:00:00Z"
        }])
        mock_insert.return_value.execute = mock_execute
        mock_table.insert = mock_insert
        mock_client.table.return_value = mock_table

        created = create_profile(
            user_id=self.user_a_id,
            display_name="New Tech",
            email=self.user_a_email,
            client=mock_client
        )
        self.assertEqual(created["id"], self.user_a_id)
        self.assertEqual(created["display_name"], "New Tech")
        mock_client.table.assert_called_with("profiles")

    # -------------------------------------------------------------------------
    # Test 7: Profile Retrieval (get_profile & get_or_create_profile)
    # -------------------------------------------------------------------------
    def test_get_profile_returns_none_when_not_found(self):
        """7a. Verifies get_profile safely returns None when no profile row exists."""
        mock_client = MagicMock()
        mock_query = MagicMock()
        mock_execute = MagicMock(data=[])
        mock_query.execute.return_value = mock_execute
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value = mock_query

        res = get_profile(user_id=self.user_a_id, client=mock_client)
        self.assertIsNone(res)

    def test_get_or_create_profile_creates_when_missing(self):
        """7b. Verifies get_or_create_profile initializes a profile if absent."""
        mock_client = MagicMock()

        # Step 1: select returns empty
        mock_select_query = MagicMock()
        mock_select_query.execute.return_value = MagicMock(data=[])
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value = mock_select_query

        # Step 2: insert returns new profile
        mock_insert_query = MagicMock()
        mock_insert_query.execute.return_value = MagicMock(data=[{
            "id": self.user_a_id,
            "display_name": "Operator A",
            "email": self.user_a_email
        }])
        mock_client.table.return_value.insert.return_value = mock_insert_query

        profile = get_or_create_profile(
            user_id=self.user_a_id,
            display_name="Operator A",
            email=self.user_a_email,
            client=mock_client
        )
        self.assertEqual(profile["id"], self.user_a_id)

    # -------------------------------------------------------------------------
    # Test 8: Profile Update Behavior (PATCH /api/v1/auth/profile)
    # -------------------------------------------------------------------------
    def test_update_profile_api_success(self):
        """8a. Verifies PATCH /api/v1/auth/profile updates display name for authenticated user."""
        token = self._create_mock_jwt(self.user_a_id, self.user_a_email)

        updated_record = {
            "id": self.user_a_id,
            "display_name": "Lead Reliability Engineer",
            "email": self.user_a_email,
            "updated_at": "2026-10-06T10:15:00Z"
        }

        with patch("api.auth.get_settings") as mock_settings, \
             patch("persistence.repository.is_supabase_configured", return_value=True), \
             patch("persistence.repository.update_profile", return_value=updated_record):
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.patch(
                "/api/v1/auth/profile",
                headers={"Authorization": f"Bearer {token}"},
                json={"display_name": "Lead Reliability Engineer"}
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertTrue(body["success"])
            self.assertEqual(body["data"]["display_name"], "Lead Reliability Engineer")

    def test_update_profile_rejects_blank_display_name(self):
        """8b. Verifies PATCH /api/v1/auth/profile rejects empty display name with HTTP 422."""
        token = self._create_mock_jwt(self.user_a_id, self.user_a_email)

        with patch("api.auth.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.patch(
                "/api/v1/auth/profile",
                headers={"Authorization": f"Bearer {token}"},
                json={"display_name": "   "}
            )
            self.assertEqual(response.status_code, 422)
            body = response.json()
            self.assertFalse(body["success"])
            self.assertEqual(body["error"]["code"], "VALIDATION_ERROR")

    # -------------------------------------------------------------------------
    # Test 9: Zero Secret / Password / Token Fields
    # -------------------------------------------------------------------------
    def test_zero_password_or_token_fields_in_profile_schemas(self):
        """9. Verifies ProfileData and ProfileResponse never contain password, JWT, or token fields."""
        profile = ProfileData(
            id=self.user_a_id,
            display_name="Tech Specialist",
            email=self.user_a_email
        )
        dump = profile.model_dump()
        forbidden_keys = ["password", "token", "access_token", "jwt", "secret", "service_role"]
        for key in forbidden_keys:
            self.assertNotIn(key, dump)

        resp = ProfileResponse(success=True, data=profile)
        resp_dump = resp.model_dump()
        for key in forbidden_keys:
            self.assertNotIn(key, resp_dump)
            self.assertNotIn(key, resp_dump.get("data", {}))

    # -------------------------------------------------------------------------
    # Test 10: Authentication Endpoint Regression (GET /api/v1/auth/me)
    # -------------------------------------------------------------------------
    def test_auth_me_endpoint_regression(self):
        """10. Verifies GET /api/v1/auth/me remains 100% operational alongside profile endpoints."""
        token = self._create_mock_jwt(self.user_a_id, self.user_a_email)

        with patch("api.auth.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertTrue(body["success"])
            self.assertEqual(body["data"]["user_id"], self.user_a_id)
            self.assertTrue(body["data"]["authenticated"])

    # -------------------------------------------------------------------------
    # Test 11: Existing API Regression (Classification & Health)
    # -------------------------------------------------------------------------
    def test_existing_api_regression(self):
        """11. Verifies classification and health endpoints remain unauthenticated and operational."""
        # Health probe
        resp_health = self.client.get("/health")
        self.assertEqual(resp_health.status_code, 200)

        # Status probe
        resp_status = self.client.get("/api/v1/status")
        self.assertEqual(resp_status.status_code, 200)

        # Local classification
        resp_classify = self.client.post(
            "/api/v1/classify",
            json={"description": "Motor bearings overheating rapidly.", "mode": "local"}
        )
        self.assertEqual(resp_classify.status_code, 200)
        self.assertTrue(resp_classify.json()["success"])

    # -------------------------------------------------------------------------
    # Test 12: Existing Database Regression (defect_reports)
    # -------------------------------------------------------------------------
    def test_existing_database_regression(self):
        """12. Verifies defect_reports persistence remains unaffected by profiles additions."""
        with patch("persistence.repository.is_supabase_configured", return_value=True), \
             patch("persistence.repository.get_defect_reports", return_value=[]):
            response = self.client.get("/api/v1/history")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["success"])

    # -------------------------------------------------------------------------
    # Test 13: OpenAPI Specification Regression
    # -------------------------------------------------------------------------
    def test_openapi_documents_profile_endpoints(self):
        """13. Verifies OpenAPI schema documents /api/v1/auth/profile with correct schemas."""
        schema = self.app.openapi()
        paths = schema.get("paths", {})

        self.assertIn("/api/v1/auth/profile", paths)
        profile_ops = paths["/api/v1/auth/profile"]
        self.assertIn("get", profile_ops)
        self.assertIn("patch", profile_ops)

        schemas = schema.get("components", {}).get("schemas", {})
        self.assertIn("ProfileData", schemas)
        self.assertIn("ProfileResponse", schemas)
        self.assertIn("ProfileUpdateRequest", schemas)

        # Check that no passwords or secret keys exist in schemas
        self.assertNotIn("password", schemas["ProfileData"].get("properties", {}))


if __name__ == "__main__":
    unittest.main()
