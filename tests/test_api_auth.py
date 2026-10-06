"""Unit and Integration Tests for Phase 5 Step 5.1: Supabase Authentication Foundation.

Verifies:
A. Missing Authorization header (HTTP 401)
B. Malformed Authorization header (HTTP 401)
C. Invalid token (offline HS256 & online Supabase Auth mock, HTTP 401)
D. Expired token (offline HS256 & online Supabase Auth mock, HTTP 401)
E. Valid mocked token/user (offline HS256 & online Supabase Auth mock, HTTP 200)
F. Authenticated dependency behavior (extract_bearer_token, get_current_user, get_optional_user)
G. No secret/token leakage in error responses or logs
H. Existing public health endpoints behavior (GET /, GET /health, GET /api/v1/status)
I. Existing classification API regression (POST /api/v1/classify works unauthenticated)
J. Existing history API regression (GET /api/v1/history works unauthenticated)
K. Existing error envelope compatibility (standardized ErrorResponse schema)
L. Existing OpenAPI compatibility (/api/v1/auth/me registered, clean schema)
M. Unconfigured auth service rejection (HTTP 401)
N. AuthenticatedUser domain schema integrity
"""

import time
import unittest
from unittest.mock import MagicMock, patch
import jwt
from fastapi import Request
from fastapi.testclient import TestClient

from api.main import create_app
from api.auth import extract_bearer_token, verify_supabase_jwt
from api.dependencies import get_current_user, get_optional_user
from api.errors import UnauthorizedError
from api.schemas.auth import AuthenticatedUser, UserResponse
from api.schemas.errors import ErrorCode


class TestSupabaseAuthFoundation(unittest.TestCase):
    """Test suite for Phase 5 Step 5.1 Supabase Authentication Foundation."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = TestClient(cls.app, raise_server_exceptions=False)
        cls.mock_jwt_secret = "test_supabase_mock_jwt_secret_xyz789_32bytes_min_length"
        cls.test_user_id = "550e8400-e29b-41d4-a716-446655440000"
        cls.test_email = "operator@factory.example.com"

    def _create_mock_jwt(
        self,
        sub: str = "550e8400-e29b-41d4-a716-446655440000",
        email: str = "operator@factory.example.com",
        expires_in: int = 3600,
        secret: str = None,
        extra_claims: dict = None
    ) -> str:
        """Helper to generate a mock HS256 signed JWT for testing without real credentials."""
        now = int(time.time())
        payload = {
            "sub": sub,
            "email": email,
            "iat": now,
            "exp": now + expires_in,
            "user_metadata": {"role": "operator", "department": "maintenance"}
        }
        if extra_claims:
            payload.update(extra_claims)
        key = secret or self.mock_jwt_secret
        return jwt.encode(payload, key, algorithm="HS256")

    # -------------------------------------------------------------------------
    # Test A: Missing Authorization Header
    # -------------------------------------------------------------------------
    def test_missing_authorization_header_returns_401(self):
        """A. Verifies missing Authorization header returns HTTP 401 with standard envelope."""
        response = self.client.get("/api/v1/auth/me")
        self.assertEqual(response.status_code, 401)

        body = response.json()
        self.assertFalse(body["success"])
        self.assertIn("error", body)
        self.assertEqual(body["error"]["code"], ErrorCode.UNAUTHORIZED.value)
        self.assertIn("missing authorization header", body["error"]["message"].lower())
        self.assertIn("www-authenticate", response.headers)
        self.assertEqual(response.headers["www-authenticate"], "Bearer")

    # -------------------------------------------------------------------------
    # Test B: Malformed Authorization Header
    # -------------------------------------------------------------------------
    def test_malformed_auth_header_not_bearer(self):
        """B1. Verifies non-Bearer scheme returns HTTP 401 with expected format notice."""
        response = self.client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Basic dXNlcjpwYXNz"}
        )
        self.assertEqual(response.status_code, 401)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], ErrorCode.UNAUTHORIZED.value)
        self.assertIn("bearer <token>", body["error"]["message"].lower())

    def test_malformed_auth_header_missing_token(self):
        """B2. Verifies header with only 'Bearer' (no token) returns HTTP 401."""
        response = self.client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer"}
        )
        self.assertEqual(response.status_code, 401)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], ErrorCode.UNAUTHORIZED.value)

    def test_malformed_auth_header_empty_token(self):
        """B3. Verifies header with 'Bearer   ' (empty/spaces) returns HTTP 401."""
        response = self.client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer    "}
        )
        self.assertEqual(response.status_code, 401)
        body = response.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], ErrorCode.UNAUTHORIZED.value)
        msg = body["error"]["message"].lower()
        self.assertTrue("missing or empty" in msg or "invalid authorization header" in msg)

    # -------------------------------------------------------------------------
    # Test C: Invalid Token
    # -------------------------------------------------------------------------
    def test_invalid_token_offline_secret(self):
        """C1. Verifies malformed/tampered JWT returns HTTP 401 under offline secret mode."""
        with patch("api.auth.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.get(
                "/api/v1/auth/me",
                headers={"Authorization": "Bearer not.a.valid.jwt"}
            )
            self.assertEqual(response.status_code, 401)
            body = response.json()
            self.assertFalse(body["success"])
            self.assertEqual(body["error"]["code"], ErrorCode.UNAUTHORIZED.value)
            self.assertEqual(body["error"]["message"], "Invalid authentication token.")

    def test_invalid_token_wrong_secret(self):
        """C2. Verifies JWT signed with wrong secret is rejected with HTTP 401."""
        token = self._create_mock_jwt(secret="wrong_secret_key_12345678901234567890_32bytes")
        with patch("api.auth.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            self.assertEqual(response.status_code, 401)
            body = response.json()
            self.assertFalse(body["success"])
            self.assertEqual(body["error"]["code"], ErrorCode.UNAUTHORIZED.value)
            self.assertEqual(body["error"]["message"], "Invalid authentication token.")

    def test_invalid_token_missing_sub_claim(self):
        """C3. Verifies JWT missing 'sub' (user identity) claim is rejected with HTTP 401."""
        now = int(time.time())
        token = jwt.encode(
            {"email": "nouser@example.com", "exp": now + 3600},
            self.mock_jwt_secret,
            algorithm="HS256"
        )
        with patch("api.auth.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            self.assertEqual(response.status_code, 401)
            body = response.json()
            self.assertFalse(body["success"])
            self.assertEqual(body["error"]["code"], ErrorCode.UNAUTHORIZED.value)
            self.assertIn("missing subject identity", body["error"]["message"].lower())

    def test_invalid_token_online_supabase_auth_mock(self):
        """C4. Verifies invalid token rejected via online Supabase Auth mock when secret not configured."""
        mock_sb_client = MagicMock()
        mock_sb_client.auth.get_user.side_effect = Exception("Invalid JWT: token has invalid signature")

        with patch("api.auth.get_settings") as mock_settings, \
             patch("api.auth.is_supabase_configured", return_value=True), \
             patch("api.auth.get_supabase_client", return_value=mock_sb_client):
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = None
            mock_settings.return_value = mock_s

            response = self.client.get(
                "/api/v1/auth/me",
                headers={"Authorization": "Bearer mock_invalid_token_abc"}
            )
            self.assertEqual(response.status_code, 401)
            body = response.json()
            self.assertFalse(body["success"])
            self.assertEqual(body["error"]["code"], ErrorCode.UNAUTHORIZED.value)
            self.assertEqual(body["error"]["message"], "Invalid authentication token.")

    # -------------------------------------------------------------------------
    # Test D: Expired Token
    # -------------------------------------------------------------------------
    def test_expired_token_offline_secret(self):
        """D1. Verifies expired JWT returns HTTP 401 with expired message under offline secret."""
        expired_token = self._create_mock_jwt(expires_in=-3600)  # Expired 1 hour ago
        with patch("api.auth.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {expired_token}"}
            )
            self.assertEqual(response.status_code, 401)
            body = response.json()
            self.assertFalse(body["success"])
            self.assertEqual(body["error"]["code"], ErrorCode.UNAUTHORIZED.value)
            self.assertEqual(body["error"]["message"], "Authentication token has expired.")

    def test_expired_token_online_supabase_auth_mock(self):
        """D2. Verifies expired token rejected via online Supabase Auth mock."""
        mock_sb_client = MagicMock()
        mock_sb_client.auth.get_user.side_effect = Exception("JWT expired: token has expired")

        with patch("api.auth.get_settings") as mock_settings, \
             patch("api.auth.is_supabase_configured", return_value=True), \
             patch("api.auth.get_supabase_client", return_value=mock_sb_client):
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = None
            mock_settings.return_value = mock_s

            response = self.client.get(
                "/api/v1/auth/me",
                headers={"Authorization": "Bearer mock_expired_token_abc"}
            )
            self.assertEqual(response.status_code, 401)
            body = response.json()
            self.assertFalse(body["success"])
            self.assertEqual(body["error"]["code"], ErrorCode.UNAUTHORIZED.value)
            self.assertEqual(body["error"]["message"], "Authentication token has expired.")

    # -------------------------------------------------------------------------
    # Test E: Valid Mocked Token / User
    # -------------------------------------------------------------------------
    def test_valid_token_offline_secret(self):
        """E1. Verifies valid JWT succeeds under offline secret mode returning user identity."""
        token = self._create_mock_jwt(
            sub=self.test_user_id,
            email=self.test_email,
            expires_in=3600
        )
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
            self.assertIn("data", body)
            self.assertEqual(body["data"]["user_id"], self.test_user_id)
            self.assertEqual(body["data"]["email"], self.test_email)
            self.assertTrue(body["data"]["authenticated"])

    def test_valid_token_online_supabase_auth_mock(self):
        """E2. Verifies valid token succeeds via online Supabase Auth client mock."""
        mock_user = MagicMock()
        mock_user.id = self.test_user_id
        mock_user.email = self.test_email
        mock_user.user_metadata = {"role": "technician"}

        mock_response = MagicMock()
        mock_response.user = mock_user

        mock_sb_client = MagicMock()
        mock_sb_client.auth.get_user.return_value = mock_response

        with patch("api.auth.get_settings") as mock_settings, \
             patch("api.auth.is_supabase_configured", return_value=True), \
             patch("api.auth.get_supabase_client", return_value=mock_sb_client):
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = None
            mock_settings.return_value = mock_s

            response = self.client.get(
                "/api/v1/auth/me",
                headers={"Authorization": "Bearer mock_valid_access_token_123"}
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertTrue(body["success"])
            self.assertEqual(body["data"]["user_id"], self.test_user_id)
            self.assertEqual(body["data"]["email"], self.test_email)
            self.assertTrue(body["data"]["authenticated"])

    # -------------------------------------------------------------------------
    # Test F: Authenticated Dependency Behavior
    # -------------------------------------------------------------------------
    def test_extract_bearer_token_direct(self):
        """F1. Verifies extract_bearer_token helper functions correctly across variations."""
        # Valid cases
        self.assertEqual(extract_bearer_token("Bearer mytoken123"), "mytoken123")
        self.assertEqual(extract_bearer_token("bearer lowercase_token"), "lowercase_token")
        self.assertEqual(extract_bearer_token("Bearer   spaced_token  "), "spaced_token")

        # Invalid cases
        with self.assertRaises(UnauthorizedError):
            extract_bearer_token(None)
        with self.assertRaises(UnauthorizedError):
            extract_bearer_token("")
        with self.assertRaises(UnauthorizedError):
            extract_bearer_token("   ")
        with self.assertRaises(UnauthorizedError):
            extract_bearer_token("Token xyz")
        with self.assertRaises(UnauthorizedError):
            extract_bearer_token("Bearer")
        with self.assertRaises(UnauthorizedError):
            extract_bearer_token("Bearer    ")

    def test_get_current_user_dependency_direct(self):
        """F2. Verifies get_current_user FastAPI dependency directly with Request mocks."""
        token = self._create_mock_jwt(sub=self.test_user_id, email=self.test_email)

        # Mock request with valid auth header
        mock_req = MagicMock(spec=Request)
        mock_req.headers = {"Authorization": f"Bearer {token}"}

        with patch("api.auth.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            user = get_current_user(mock_req)
            self.assertIsInstance(user, AuthenticatedUser)
            self.assertEqual(user.user_id, self.test_user_id)
            self.assertEqual(user.email, self.test_email)

        # Mock request without auth header
        mock_req_empty = MagicMock(spec=Request)
        mock_req_empty.headers = {}
        with self.assertRaises(UnauthorizedError):
            get_current_user(mock_req_empty)

    def test_get_optional_user_dependency_direct(self):
        """F3. Verifies get_optional_user returns None when header absent and validates when present."""
        # 1. Missing header -> returns None safely
        mock_req_none = MagicMock(spec=Request)
        mock_req_none.headers = {}
        self.assertIsNone(get_optional_user(mock_req_none))

        mock_req_blank = MagicMock(spec=Request)
        mock_req_blank.headers = {"Authorization": "   "}
        self.assertIsNone(get_optional_user(mock_req_blank))

        # 2. Valid header -> returns AuthenticatedUser
        token = self._create_mock_jwt(sub=self.test_user_id, email=self.test_email)
        mock_req_valid = MagicMock(spec=Request)
        mock_req_valid.headers = {"Authorization": f"Bearer {token}"}

        with patch("api.auth.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            user = get_optional_user(mock_req_valid)
            self.assertIsNotNone(user)
            self.assertEqual(user.user_id, self.test_user_id)

        # 3. Invalid header -> raises UnauthorizedError
        mock_req_invalid = MagicMock(spec=Request)
        mock_req_invalid.headers = {"Authorization": "Bearer bad_token"}

        with patch("api.auth.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            with self.assertRaises(UnauthorizedError):
                get_optional_user(mock_req_invalid)

    # -------------------------------------------------------------------------
    # Test G: No Secret / Token Leakage
    # -------------------------------------------------------------------------
    def test_no_secret_or_token_leakage_in_error_responses(self):
        """G. Verifies tokens, secrets, or internal paths never leak in error response JSON."""
        secret_sample = "sensitive_jwt_secret_998877"
        token_sample = "sensitive_raw_access_token_abc123"

        with patch("api.auth.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = secret_sample
            mock_settings.return_value = mock_s

            response = self.client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {token_sample}"}
            )
            raw_text = response.text
            self.assertNotIn(secret_sample, raw_text)
            self.assertNotIn(token_sample, raw_text)
            self.assertNotIn("Traceback", raw_text)
            self.assertNotIn("site-packages", raw_text)
            self.assertNotIn("Users", raw_text)

    # -------------------------------------------------------------------------
    # Test H: Existing Public Health Endpoints Behavior
    # -------------------------------------------------------------------------
    def test_public_health_endpoints_remain_accessible(self):
        """H. Verifies public system endpoints remain accessible without authentication."""
        # GET /
        resp_root = self.client.get("/")
        self.assertEqual(resp_root.status_code, 200)

        # GET /health
        resp_health = self.client.get("/health")
        self.assertEqual(resp_health.status_code, 200)
        self.assertEqual(resp_health.json()["status"], "healthy")

        # GET /api/v1/status
        resp_status = self.client.get("/api/v1/status")
        self.assertEqual(resp_status.status_code, 200)
        self.assertIn("status", resp_status.json()["data"])

    # -------------------------------------------------------------------------
    # Test I: Existing Classification API Regression
    # -------------------------------------------------------------------------
    def test_existing_classification_api_regression(self):
        """I. Verifies POST /api/v1/classify operates unauthenticated without breaking contract."""
        payload = {
            "description": "Bearing vibration on exhaust fan motor.",
            "mode": "local"
        }
        response = self.client.post("/api/v1/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["data"]["category"], "Mechanical Fault")

    # -------------------------------------------------------------------------
    # Test J: Existing History API Regression
    # -------------------------------------------------------------------------
    def test_existing_history_api_regression(self):
        """J. Verifies GET /api/v1/history functions unauthenticated without breaking contract."""
        with patch("persistence.repository.is_supabase_configured", return_value=True), \
             patch("persistence.repository.get_defect_reports", return_value=[]):
            response = self.client.get("/api/v1/history")
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertTrue(body["success"])
            self.assertEqual(body["data"]["total"], 0)

    # -------------------------------------------------------------------------
    # Test K: Existing Error Envelope Compatibility
    # -------------------------------------------------------------------------
    def test_error_envelope_compatibility(self):
        """K. Verifies 401 error adheres strictly to canonical Phase 3.5 ErrorResponse envelope."""
        response = self.client.get("/api/v1/auth/me")
        self.assertEqual(response.status_code, 401)
        body = response.json()

        # Strict envelope structure
        self.assertIn("success", body)
        self.assertFalse(body["success"])
        self.assertIn("error", body)
        self.assertIsInstance(body["error"], dict)
        self.assertIn("code", body["error"])
        self.assertIn("message", body["error"])
        self.assertIn("details", body["error"])
        self.assertEqual(body["error"]["code"], "UNAUTHORIZED")

    # -------------------------------------------------------------------------
    # Test L: Existing OpenAPI Compatibility
    # -------------------------------------------------------------------------
    def test_openapi_compatibility(self):
        """L. Verifies /api/v1/auth/me is registered in OpenAPI and spec builds cleanly."""
        schema = self.app.openapi()
        paths = schema.get("paths", {})

        # Ensure endpoint is registered
        self.assertIn("/api/v1/auth/me", paths)
        self.assertIn("get", paths["/api/v1/auth/me"])

        auth_op = paths["/api/v1/auth/me"]["get"]
        self.assertIn("Authentication", auth_op.get("tags", []))
        self.assertIn("401", auth_op.get("responses", {}))

        # Verify docs routes remain accessible
        docs_resp = self.client.get("/docs")
        self.assertEqual(docs_resp.status_code, 200)

        redoc_resp = self.client.get("/redoc")
        self.assertEqual(redoc_resp.status_code, 200)

        openapi_resp = self.client.get("/openapi.json")
        self.assertEqual(openapi_resp.status_code, 200)

    # -------------------------------------------------------------------------
    # Test M: Unconfigured Authentication Service
    # -------------------------------------------------------------------------
    def test_unconfigured_auth_service_returns_401(self):
        """M. Verifies clean 401 when neither JWT secret nor Supabase is configured."""
        with patch("api.auth.get_settings") as mock_settings, \
             patch("api.auth.is_supabase_configured", return_value=False):
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = None
            mock_settings.return_value = mock_s

            response = self.client.get(
                "/api/v1/auth/me",
                headers={"Authorization": "Bearer some_token_when_unconfigured"}
            )
            self.assertEqual(response.status_code, 401)
            body = response.json()
            self.assertFalse(body["success"])
            self.assertEqual(body["error"]["code"], ErrorCode.UNAUTHORIZED.value)
            self.assertIn("unconfigured", body["error"]["message"].lower())

    # -------------------------------------------------------------------------
    # Test N: AuthenticatedUser Domain Schema Integrity
    # -------------------------------------------------------------------------
    def test_authenticated_user_model_integrity(self):
        """N. Verifies AuthenticatedUser model contains required fields without password/secrets."""
        user = AuthenticatedUser(
            user_id="user-12345",
            email="tech@factory.com",
            metadata={"department": "QA"}
        )
        self.assertEqual(user.user_id, "user-12345")
        self.assertEqual(user.email, "tech@factory.com")
        self.assertEqual(user.metadata["department"], "QA")

        # Serializes cleanly
        user_dict = user.model_dump()
        self.assertNotIn("password", user_dict)
        self.assertNotIn("token", user_dict)
        self.assertNotIn("secret", user_dict)

        # Default metadata is dict
        minimal_user = AuthenticatedUser(user_id="user-999")
        self.assertEqual(minimal_user.user_id, "user-999")
        self.assertIsNone(minimal_user.email)
        self.assertEqual(minimal_user.metadata, {})


if __name__ == "__main__":
    unittest.main()
