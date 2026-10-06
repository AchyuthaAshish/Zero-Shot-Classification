"""Comprehensive API Security Test Suite for Industrial Defect Classification Backend.

Conforms to Phase 5 Step 5.7:
A. Security Headers:
   - X-Content-Type-Options: nosniff
   - X-Frame-Options: DENY
   - Referrer-Policy: strict-origin-when-cross-origin
   - Permissions-Policy: geolocation=(), camera=(), microphone=()
   - Content-Security-Policy: strict API CSP on REST routes, tailored docs CSP on /docs and /redoc
   - Strict-Transport-Security: conditional (omitted on local HTTP, present on HTTPS/production)
   - Tested across 200 (normal), 404 (not found), 422 (validation error), 500 (internal server error)

B. CORS Hardening:
   - Allowed development origins receive Access-Control-Allow-Origin & credentials
   - Disallowed origins do not receive CORS allow headers
   - Allowed methods: GET, POST, PATCH (preflight and response verification)
   - Explicit headers: Authorization, Content-Type, Accept, Origin, X-Requested-With
   - Wildcard origin rejection when credentials enabled
   - Production origin requirements and localhost rejection
   - Security headers remain present on CORS responses

C. Authentication & RFC 6750 Compliance:
   - 401 Unauthorized responses for missing, malformed, invalid, and expired tokens
   - Every HTTP 401 response includes 'WWW-Authenticate: Bearer'
   - HTTP 403 Forbidden responses do NOT include 'WWW-Authenticate'
   - Sanitized envelope: {success: false, error: {code, message, details}}
   - Zero token, secret, or key leakage in response body

D. Public vs Protected Routes:
   - Public: /, /health, /api/v1/status, /api/v1/classify, /docs, /redoc, /openapi.json
   - Protected: /api/v1/auth/me, /api/v1/auth/profile (GET & PATCH)

E. Error Information Disclosure Defense:
   - 500 unhandled exceptions do not disclose tracebacks, file paths, secrets, or DB strings
   - Standard sanitized error response envelope preserved
"""

import os
import time
import unittest
from unittest.mock import MagicMock, patch
import jwt
from fastapi import APIRouter, Depends
from fastapi.testclient import TestClient

from api.main import create_app
from api.config import reset_api_settings, load_api_settings
from api.dependencies import require_employee, get_current_user
from api.schemas.auth import AuthenticatedUser
from api.schemas.errors import ErrorCode
from core.exceptions import ConfigurationError


class TestComprehensiveAPISecurity(unittest.TestCase):
    """Full API Security verification suite covering headers, CORS, auth, routing, and error disclosure."""

    @classmethod
    def setUpClass(cls):
        reset_api_settings()
        cls.app = create_app()

        # Add an internal test route to verify 500 handling with simulated sensitive leaks
        fault_router = APIRouter(prefix="/test-security-faults")

        @fault_router.get("/trigger-sensitive-crash")
        def trigger_sensitive_crash():
            raise RuntimeError(
                "Database crash at postgresql://postgres:SecretPassword123@db.supabase.co:5432/postgres "
                "with JWT secret secret_token_xyz999 in C:\\Users\\achyu\\sensitive\\file.py"
            )

        # Add a test-only route protected by require_employee to explicitly test 403 vs 401
        @fault_router.get("/employee-guarded")
        def employee_guarded_endpoint(user: AuthenticatedUser = Depends(require_employee)):
            return {"success": True, "user_id": user.user_id, "role": user.role}

        cls.app.include_router(fault_router)
        cls.client = TestClient(cls.app, raise_server_exceptions=False)

        cls.mock_jwt_secret = "test_mock_jwt_secret_xyz789_32bytes_min_length"
        cls.test_user_id = "550e8400-e29b-41d4-a716-446655440099"
        cls.test_email = "security.tester@industrial.example.com"

    @classmethod
    def tearDownClass(cls):
        reset_api_settings()

    def _create_mock_jwt(self, expires_in: int = 3600, sub: str = None, email: str = None) -> str:
        """Helper to create a test HS256 JWT."""
        now = int(time.time())
        payload = {
            "sub": sub or self.test_user_id,
            "email": email or self.test_email,
            "iat": now,
            "exp": now + expires_in,
            "user_metadata": {"role": "employee", "full_name": "Security Test User"}
        }
        return jwt.encode(payload, self.mock_jwt_secret, algorithm="HS256")

    # =========================================================================
    # PART A: SECURITY HEADERS
    # =========================================================================

    def _verify_baseline_security_headers(self, response):
        """Helper to verify presence and correctness of baseline defensive headers."""
        headers = response.headers
        self.assertEqual(headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(headers.get("x-frame-options"), "DENY")
        self.assertEqual(headers.get("referrer-policy"), "strict-origin-when-cross-origin")
        self.assertEqual(headers.get("permissions-policy"), "geolocation=(), camera=(), microphone=()")
        self.assertIn("default-src 'none'", headers.get("content-security-policy", ""))

    def test_security_headers_on_normal_200_response(self):
        """A1. Verifies security headers on normal 200 response (GET /health)."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self._verify_baseline_security_headers(response)

    def test_security_headers_on_404_not_found(self):
        """A2. Verifies security headers on 404 response."""
        response = self.client.get("/api/v1/nonexistent-route-for-testing")
        self.assertEqual(response.status_code, 404)
        self._verify_baseline_security_headers(response)

    def test_security_headers_on_422_validation_error(self):
        """A3. Verifies security headers on 422 validation error."""
        response = self.client.post("/api/v1/classify", json={"bad_payload": True})
        self.assertEqual(response.status_code, 422)
        self._verify_baseline_security_headers(response)

    def test_security_headers_on_500_server_error(self):
        """A4. Verifies security headers on 500 application error."""
        response = self.client.get("/test-security-faults/trigger-sensitive-crash")
        self.assertEqual(response.status_code, 500)
        self._verify_baseline_security_headers(response)

    def test_hsts_omitted_on_local_http(self):
        """A5. Verifies Strict-Transport-Security is NOT sent on local HTTP development requests."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("strict-transport-security", response.headers)

    def test_hsts_enforced_on_https_and_forwarded_proto(self):
        """A6. Verifies Strict-Transport-Security is sent on HTTPS or X-Forwarded-Proto=https."""
        response = self.client.get("/health", headers={"X-Forwarded-Proto": "https"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("strict-transport-security", response.headers)
        self.assertEqual(response.headers["strict-transport-security"], "max-age=31536000; includeSubDomains")

    def test_docs_and_redoc_receive_tailored_docs_csp(self):
        """A7. Verifies Swagger UI /docs and ReDoc receive tailored docs CSP supporting UI assets."""
        docs_resp = self.client.get("/docs")
        self.assertEqual(docs_resp.status_code, 200)
        self.assertIn("https://cdn.jsdelivr.net", docs_resp.headers.get("content-security-policy", ""))
        self.assertEqual(docs_resp.headers.get("x-frame-options"), "DENY")

        redoc_resp = self.client.get("/redoc")
        self.assertEqual(redoc_resp.status_code, 200)
        self.assertIn("https://cdn.jsdelivr.net", redoc_resp.headers.get("content-security-policy", ""))

    # =========================================================================
    # PART B: CORS HARDENING
    # =========================================================================

    def test_cors_allowed_development_origin(self):
        """B1. Verifies allowed development origin receives Access-Control headers."""
        response = self.client.get("/health", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:5173")
        self.assertEqual(response.headers.get("access-control-allow-credentials"), "true")

    def test_cors_disallowed_origin(self):
        """B2. Verifies disallowed origin does NOT receive Access-Control-Allow-Origin."""
        response = self.client.get("/health", headers={"Origin": "https://attacker.evil.com"})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_cors_patch_preflight(self):
        """B3. Verifies CORS preflight OPTIONS request for PATCH succeeds with explicit headers."""
        response = self.client.options(
            "/api/v1/auth/profile",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "PATCH",
                "Access-Control-Request-Headers": "Authorization, Content-Type",
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:5173")
        allow_methods = [m.strip() for m in response.headers.get("access-control-allow-methods", "").split(",")]
        self.assertIn("PATCH", allow_methods)
        self.assertIn("GET", allow_methods)
        self.assertIn("POST", allow_methods)
        self.assertIn("OPTIONS", allow_methods)

        allow_headers = response.headers.get("access-control-allow-headers", "").lower()
        self.assertIn("authorization", allow_headers)
        self.assertIn("content-type", allow_headers)

    def test_cors_wildcard_rejection_with_credentials(self):
        """B4. Verifies wildcard '*' origin is rejected when credentials are enabled."""
        with patch.dict(os.environ, {"CORS_ORIGINS": "*"}, clear=False):
            with self.assertRaises(ConfigurationError) as ctx:
                load_api_settings()
            self.assertIn("wildcard", str(ctx.exception).lower())

    def test_cors_production_origin_requirements(self):
        """B5. Verifies production rejects empty CORS origins and rejects localhost origins."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "CORS_ORIGINS": ""}, clear=False):
            with self.assertRaises(ConfigurationError) as ctx:
                load_api_settings()
            self.assertIn("cors_origins", str(ctx.exception).lower())

        with patch.dict(os.environ, {"ENVIRONMENT": "production", "CORS_ORIGINS": "http://localhost:3000"}, clear=False):
            with self.assertRaises(ConfigurationError) as ctx:
                load_api_settings()
            self.assertIn("localhost", str(ctx.exception).lower())

    def test_cors_production_valid_explicit_origin(self):
        """B6. Verifies production accepts valid explicit domain origins."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "CORS_ORIGINS": "https://defects.enterprise.internal"}, clear=False):
            settings = load_api_settings()
            self.assertEqual(settings.cors_origins, ["https://defects.enterprise.internal"])

    def test_security_headers_remain_on_cors_responses(self):
        """B7. Verifies defensive security headers remain attached to CORS responses."""
        response = self.client.get("/health", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:5173")
        self._verify_baseline_security_headers(response)

    # =========================================================================
    # PART C: AUTHENTICATION & RFC 6750 WWW-AUTHENTICATE
    # =========================================================================

    def test_missing_token_returns_401_with_www_authenticate(self):
        """C1. Verifies missing token returns 401 with WWW-Authenticate: Bearer."""
        response = self.client.get("/api/v1/auth/me")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers.get("www-authenticate"), "Bearer")

        body = response.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], ErrorCode.UNAUTHORIZED.value)
        self.assertIn("authorization", body["error"]["message"].lower())

    def test_malformed_token_header_returns_401_with_www_authenticate(self):
        """C2. Verifies malformed Authorization header returns 401 with WWW-Authenticate: Bearer."""
        for malformed_header in [
            "Basic dXNlcjpwYXNz",
            "Bearer",
            "Bearer    ",
            "Token 123456",
        ]:
            response = self.client.get(
                "/api/v1/auth/me",
                headers={"Authorization": malformed_header}
            )
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.headers.get("www-authenticate"), "Bearer")
            body = response.json()
            self.assertFalse(body["success"])
            self.assertEqual(body["error"]["code"], ErrorCode.UNAUTHORIZED.value)

    def test_invalid_token_returns_401_with_www_authenticate(self):
        """C3. Verifies invalid/tampered token returns 401 with WWW-Authenticate: Bearer."""
        with patch("api.auth.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.get(
                "/api/v1/auth/me",
                headers={"Authorization": "Bearer invalid.jwt.signature"}
            )
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.headers.get("www-authenticate"), "Bearer")
            body = response.json()
            self.assertFalse(body["success"])
            self.assertEqual(body["error"]["code"], ErrorCode.UNAUTHORIZED.value)

    def test_expired_token_returns_401_with_www_authenticate(self):
        """C4. Verifies expired token returns 401 with WWW-Authenticate: Bearer."""
        expired_token = self._create_mock_jwt(expires_in=-3600)

        with patch("api.auth.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {expired_token}"}
            )
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.headers.get("www-authenticate"), "Bearer")
            body = response.json()
            self.assertFalse(body["success"])
            self.assertEqual(body["error"]["code"], ErrorCode.UNAUTHORIZED.value)
            self.assertIn("expired", body["error"]["message"].lower())

    def test_403_forbidden_does_not_have_www_authenticate(self):
        """C5. Verifies HTTP 403 Forbidden does NOT include WWW-Authenticate: Bearer."""
        valid_token = self._create_mock_jwt(expires_in=3600)

        # Authenticated user without profile -> 403 Forbidden
        with patch("api.auth.get_settings") as mock_settings, \
             patch("persistence.repository.is_supabase_configured", return_value=True), \
             patch("persistence.repository.get_profile", return_value=None), \
             patch("database.supabase_client.is_supabase_configured", return_value=True), \
             patch("database.supabase_client.get_profile", return_value=None):
            mock_s = MagicMock()
            mock_s.supabase_jwt_secret = self.mock_jwt_secret
            mock_settings.return_value = mock_s

            response = self.client.get(
                "/test-security-faults/employee-guarded",
                headers={"Authorization": f"Bearer {valid_token}"}
            )
            self.assertEqual(response.status_code, 403)
            self.assertNotIn("www-authenticate", response.headers)
            body = response.json()
            self.assertFalse(body["success"])
            self.assertEqual(body["error"]["code"], ErrorCode.FORBIDDEN.value)

    def test_auth_failure_responses_contain_no_secrets_or_tokens(self):
        """C6. Verifies authentication failure responses do not leak token strings or internal secrets."""
        leaky_token_string = "SUPER_SECRET_TOKEN_VALUE_NOT_TO_LEAK"
        response = self.client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {leaky_token_string}"}
        )
        self.assertEqual(response.status_code, 401)
        self.assertNotIn(leaky_token_string, response.text)
        self.assertNotIn(self.mock_jwt_secret, response.text)

    # =========================================================================
    # PART D: PUBLIC VS PROTECTED ROUTES
    # =========================================================================

    def test_public_routes_accessible_without_token(self):
        """D1. Verifies intended public routes are accessible without an Authorization header."""
        # /
        root_resp = self.client.get("/")
        self.assertEqual(root_resp.status_code, 200)

        # /health
        health_resp = self.client.get("/health")
        self.assertEqual(health_resp.status_code, 200)

        # /api/v1/status
        status_resp = self.client.get("/api/v1/status")
        self.assertEqual(status_resp.status_code, 200)

        # /docs
        docs_resp = self.client.get("/docs")
        self.assertEqual(docs_resp.status_code, 200)

        # /redoc
        redoc_resp = self.client.get("/redoc")
        self.assertEqual(redoc_resp.status_code, 200)

        # /openapi.json
        openapi_resp = self.client.get("/openapi.json")
        self.assertEqual(openapi_resp.status_code, 200)

    def test_public_classify_accessible_without_token(self):
        """D2. Verifies POST /api/v1/classify is a public endpoint accessible without auth."""
        from api.dependencies import get_classifier_service
        from core.schemas import ClassificationResult

        mock_classifier = MagicMock()
        mock_result = ClassificationResult(
            category="Surface Defect",
            reason="Clear scratch detected on surface",
            language="English",
            reliability="High",
            status="success",
            original_description="Deep scratch on aluminum housing surface",
            normalized_description="deep scratch on aluminum housing surface",
            model_source="local_ml",
            confidence=0.95
        )
        mock_classifier.classify.return_value = mock_result

        self.app.dependency_overrides[get_classifier_service] = lambda: mock_classifier
        try:
            response = self.client.post(
                "/api/v1/classify",
                json={"description": "Deep scratch on aluminum housing surface"}
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["data"]["category"], "Surface Defect")
        finally:
            self.app.dependency_overrides.pop(get_classifier_service, None)

    def test_protected_routes_require_authentication(self):
        """D3. Verifies protected routes reject unauthenticated requests with HTTP 401."""
        protected_endpoints = [
            ("GET", "/api/v1/auth/me", None),
            ("GET", "/api/v1/auth/profile", None),
            ("PATCH", "/api/v1/auth/profile", {"display_name": "Updated Operator"}),
        ]

        for method, endpoint, payload in protected_endpoints:
            with self.subTest(endpoint=endpoint, method=method):
                if method == "GET":
                    response = self.client.get(endpoint)
                elif method == "PATCH":
                    response = self.client.patch(endpoint, json=payload)
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.headers.get("www-authenticate"), "Bearer")
                body = response.json()
                self.assertFalse(body["success"])
                self.assertEqual(body["error"]["code"], ErrorCode.UNAUTHORIZED.value)

    # =========================================================================
    # PART E: ERROR INFORMATION DISCLOSURE DEFENSE
    # =========================================================================

    def test_500_error_does_not_disclose_internal_secrets_or_paths(self):
        """E1. Verifies 500 unhandled exceptions sanitize responses and conceal sensitive internals."""
        response = self.client.get("/test-security-faults/trigger-sensitive-crash")
        self.assertEqual(response.status_code, 500)

        raw_text = response.text
        # Must not disclose database connection URI with credentials
        self.assertNotIn("SecretPassword123", raw_text)
        self.assertNotIn("postgresql://", raw_text)

        # Must not disclose secrets or tokens
        self.assertNotIn("secret_token_xyz999", raw_text)

        # Must not disclose local filesystem paths
        self.assertNotIn("C:\\Users\\achyu", raw_text)
        self.assertNotIn("sensitive\\file.py", raw_text)

        # Must not disclose Python tracebacks
        self.assertNotIn("Traceback (most recent call last)", raw_text)

        # Must conform to sanitized envelope
        body = response.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["error"]["code"], ErrorCode.INTERNAL_SERVER_ERROR.value)
        self.assertEqual(body["error"]["message"], "An unexpected error occurred while processing the request. Please try again.")
        self.assertEqual(body["error"]["details"], {})


if __name__ == "__main__":
    unittest.main()
