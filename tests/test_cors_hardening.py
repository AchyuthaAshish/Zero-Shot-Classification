"""Unit and Integration Tests for Phase 5 Step 5.7.3: CORS Hardening.

Verifies:
1. GET request from an allowed development origin succeeds and receives Access-Control-Allow-Origin.
2. POST request from an allowed development origin succeeds and receives Access-Control-Allow-Origin.
3. PATCH preflight request succeeds (OPTIONS with Access-Control-Request-Method: PATCH).
4. PATCH is explicitly listed in Access-Control-Allow-Methods.
5. Authorization header is accepted in CORS preflight.
6. Content-Type header is accepted in CORS preflight.
7. Disallowed origins do NOT receive an Access-Control-Allow-Origin header.
8. Production configuration with missing/empty CORS_ORIGINS raises ConfigurationError.
9. Production configuration with localhost or loopback origins raises ConfigurationError.
10. Production configuration with valid explicit non-localhost origin succeeds.
11. Wildcard '*' origin is rejected when credentials are enabled (raises ConfigurationError).
12. Defensive security headers (from Step 5.7.2) remain present on CORS responses.
"""

import os
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from api.main import create_app
from api.config import reset_api_settings, load_api_settings, DEFAULT_CORS_METHODS, DEFAULT_CORS_HEADERS
from core.exceptions import ConfigurationError


class TestCORSHardening(unittest.TestCase):
    """Test suite validating production-safe CORS rules and development backward compatibility."""

    def setUp(self):
        reset_api_settings()
        self.app = create_app()
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def tearDown(self):
        reset_api_settings()

    # -------------------------------------------------------------------------
    # 1. GET from Allowed Development Origin
    # -------------------------------------------------------------------------
    def test_get_from_allowed_development_origin(self):
        """1. Verifies simple GET request from allowed dev origin receives CORS allow header."""
        response = self.client.get("/health", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:5173")
        self.assertEqual(response.headers.get("access-control-allow-credentials"), "true")

    # -------------------------------------------------------------------------
    # 2. POST from Allowed Development Origin
    # -------------------------------------------------------------------------
    def test_post_from_allowed_development_origin(self):
        """2. Verifies POST request from allowed dev origin receives CORS allow header."""
        response = self.client.post(
            "/api/v1/classify",
            json={"description": "   "},
            headers={"Origin": "http://localhost:3000"}
        )
        # Even on validation error 422, CORS headers must be attached
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:3000")
        self.assertEqual(response.headers.get("access-control-allow-credentials"), "true")

    # -------------------------------------------------------------------------
    # 3 & 4. PATCH Method Support in CORS Preflight
    # -------------------------------------------------------------------------
    def test_patch_preflight_succeeds(self):
        """3. Verifies CORS preflight OPTIONS request for PATCH method succeeds."""
        response = self.client.options(
            "/api/v1/auth/profile",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "PATCH",
                "Access-Control-Request-Headers": "Authorization, Content-Type"
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:5173")

    def test_patch_listed_in_allowed_methods(self):
        """4. Verifies PATCH is included in Access-Control-Allow-Methods and methods are restricted."""
        response = self.client.options(
            "/api/v1/auth/profile",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "PATCH"
            }
        )
        self.assertEqual(response.status_code, 200)
        allow_methods = [m.strip() for m in response.headers.get("access-control-allow-methods", "").split(",")]
        self.assertIn("PATCH", allow_methods)
        self.assertIn("GET", allow_methods)
        self.assertIn("POST", allow_methods)
        self.assertIn("OPTIONS", allow_methods)
        self.assertNotIn("*", allow_methods)

    # -------------------------------------------------------------------------
    # 5 & 6. Explicit Request Headers Allowlist
    # -------------------------------------------------------------------------
    def test_authorization_header_accepted(self):
        """5. Verifies Authorization header is accepted in preflight request."""
        response = self.client.options(
            "/api/v1/auth/me",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization"
            }
        )
        self.assertEqual(response.status_code, 200)
        allow_headers = response.headers.get("access-control-allow-headers", "").lower()
        self.assertIn("authorization", allow_headers)

    def test_content_type_header_accepted(self):
        """6. Verifies Content-Type header is accepted in preflight request."""
        response = self.client.options(
            "/api/v1/classify",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type"
            }
        )
        self.assertEqual(response.status_code, 200)
        allow_headers = response.headers.get("access-control-allow-headers", "").lower()
        self.assertIn("content-type", allow_headers)

    # -------------------------------------------------------------------------
    # 7. Disallowed Origin Rejection
    # -------------------------------------------------------------------------
    def test_disallowed_origin_rejected(self):
        """7. Verifies unauthorized origin does NOT receive Access-Control-Allow-Origin header."""
        response = self.client.get("/health", headers={"Origin": "https://malicious-site.example.com"})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("access-control-allow-origin", response.headers)

    # -------------------------------------------------------------------------
    # 8. Production Configuration Without Explicit CORS_ORIGINS
    # -------------------------------------------------------------------------
    def test_production_missing_cors_origins_raises_configuration_error(self):
        """8. Verifies production environment without explicit CORS_ORIGINS raises ConfigurationError."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "CORS_ORIGINS": ""}):
            reset_api_settings()
            with self.assertRaises(ConfigurationError) as ctx:
                load_api_settings()
            self.assertIn("explicit CORS_ORIGINS must be configured", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 9. Production Configuration with Localhost Origins
    # -------------------------------------------------------------------------
    def test_production_localhost_origins_rejected(self):
        """9. Verifies production environment with localhost origins raises ConfigurationError."""
        for bad_origin in ["http://localhost:3000", "http://127.0.0.1:5173", "http://[::1]:8000"]:
            with patch.dict(os.environ, {"ENVIRONMENT": "production", "CORS_ORIGINS": bad_origin}):
                reset_api_settings()
                with self.assertRaises(ConfigurationError) as ctx:
                    load_api_settings()
                self.assertIn("Localhost and loopback origins are not permitted in production", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 10. Production Configuration with Valid Explicit Non-Localhost Origin
    # -------------------------------------------------------------------------
    def test_production_valid_explicit_origins_accepted(self):
        """10. Verifies production environment accepts valid explicit domain origins."""
        prod_origins = "https://defect-platform.company.com, https://operator-portal.industrial.com"
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "CORS_ORIGINS": prod_origins}):
            reset_api_settings()
            cfg = load_api_settings()
            self.assertEqual(
                cfg.cors_origins,
                ["https://defect-platform.company.com", "https://operator-portal.industrial.com"]
            )

    # -------------------------------------------------------------------------
    # 11. Wildcard '*' Origin Protection with Credentials
    # -------------------------------------------------------------------------
    def test_wildcard_origin_rejected(self):
        """11. Verifies wildcard '*' origin raises ConfigurationError because credentials are enabled."""
        with patch.dict(os.environ, {"CORS_ORIGINS": "*"}):
            reset_api_settings()
            with self.assertRaises(ConfigurationError) as ctx:
                load_api_settings()
            self.assertIn("CORS wildcard '*' is not permitted", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 12. Security Headers Preserved on CORS Responses
    # -------------------------------------------------------------------------
    def test_security_headers_preserved_on_cors_responses(self):
        """12. Verifies Step 5.7.2 security headers remain intact on cross-origin requests."""
        response = self.client.get("/health", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(response.status_code, 200)

        # CORS header
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:5173")

        # Step 5.7.2 Security headers
        self.assertEqual(response.headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(response.headers.get("x-frame-options"), "DENY")
        self.assertEqual(response.headers.get("referrer-policy"), "strict-origin-when-cross-origin")
        self.assertEqual(response.headers.get("permissions-policy"), "geolocation=(), camera=(), microphone=()")
        self.assertIn("default-src 'none'", response.headers.get("content-security-policy", ""))


if __name__ == "__main__":
    unittest.main()
