"""Unit and Integration Tests for Phase 5 Step 5.7.2: Security Headers Middleware.

Verifies:
1. Standard security headers present on GET /health.
2. Standard security headers present on GET / (root).
3. Standard security headers present on normal API responses (GET /api/v1/status).
4. Standard security headers present on 4xx responses (404 and 422).
5. Standard security headers present on 5xx error responses.
6. HSTS is NOT present on normal local HTTP development requests.
7. HSTS is present on HTTPS requests (direct or forwarded proto).
8. HSTS is present when production environment is simulated/configured.
9. /docs (Swagger UI) responds successfully with tailored docs CSP.
10. /redoc responds successfully with tailored docs CSP.
11. REST API endpoints receive strict API CSP (disabling scripts, styles, frames).
"""

import os
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from api.main import create_app
from api.config import reset_api_settings


class TestSecurityHeadersMiddleware(unittest.TestCase):
    """Test suite verifying defensive HTTP security headers across the FastAPI application."""

    def setUp(self):
        reset_api_settings()
        self.app = create_app()
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def tearDown(self):
        reset_api_settings()

    # -------------------------------------------------------------------------
    # 1. Health Endpoint (GET /health)
    # -------------------------------------------------------------------------
    def test_health_endpoint_has_security_headers(self):
        """1. Verifies GET /health emits all baseline defensive security headers."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        headers = response.headers
        self.assertEqual(headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(headers.get("x-frame-options"), "DENY")
        self.assertEqual(headers.get("referrer-policy"), "strict-origin-when-cross-origin")
        self.assertEqual(headers.get("permissions-policy"), "geolocation=(), camera=(), microphone=()")
        self.assertIn("default-src 'none'", headers.get("content-security-policy", ""))

    # -------------------------------------------------------------------------
    # 2. Service Info Root (GET /)
    # -------------------------------------------------------------------------
    def test_root_endpoint_has_security_headers(self):
        """2. Verifies GET / emits all baseline defensive security headers."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

        headers = response.headers
        self.assertEqual(headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(headers.get("x-frame-options"), "DENY")
        self.assertEqual(headers.get("referrer-policy"), "strict-origin-when-cross-origin")
        self.assertEqual(headers.get("permissions-policy"), "geolocation=(), camera=(), microphone=()")
        self.assertIn("default-src 'none'", headers.get("content-security-policy", ""))

    # -------------------------------------------------------------------------
    # 3. Normal API Response (GET /api/v1/status)
    # -------------------------------------------------------------------------
    def test_normal_api_endpoint_has_security_headers(self):
        """3. Verifies normal REST API route (/api/v1/status) emits security headers."""
        response = self.client.get("/api/v1/status")
        self.assertEqual(response.status_code, 200)

        headers = response.headers
        self.assertEqual(headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(headers.get("x-frame-options"), "DENY")
        self.assertEqual(headers.get("referrer-policy"), "strict-origin-when-cross-origin")
        self.assertEqual(headers.get("permissions-policy"), "geolocation=(), camera=(), microphone=()")
        self.assertIn("default-src 'none'", headers.get("content-security-policy", ""))

    # -------------------------------------------------------------------------
    # 4. Application 4xx Responses
    # -------------------------------------------------------------------------
    def test_404_not_found_has_security_headers(self):
        """4a. Verifies HTTP 404 response emits security headers."""
        response = self.client.get("/api/v1/nonexistent-route-endpoint")
        self.assertEqual(response.status_code, 404)

        headers = response.headers
        self.assertEqual(headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(headers.get("x-frame-options"), "DENY")
        self.assertEqual(headers.get("referrer-policy"), "strict-origin-when-cross-origin")
        self.assertEqual(headers.get("permissions-policy"), "geolocation=(), camera=(), microphone=()")
        self.assertIn("default-src 'none'", headers.get("content-security-policy", ""))

    def test_422_validation_error_has_security_headers(self):
        """4b. Verifies HTTP 422 RequestValidationError response emits security headers."""
        response = self.client.post("/api/v1/classify", json={"description": ""})
        self.assertEqual(response.status_code, 422)

        headers = response.headers
        self.assertEqual(headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(headers.get("x-frame-options"), "DENY")
        self.assertEqual(headers.get("referrer-policy"), "strict-origin-when-cross-origin")
        self.assertEqual(headers.get("permissions-policy"), "geolocation=(), camera=(), microphone=()")

    # -------------------------------------------------------------------------
    # 5. Application 5xx Responses
    # -------------------------------------------------------------------------
    def test_500_unhandled_error_has_security_headers(self):
        """5. Verifies HTTP 500 unhandled exception response emits security headers."""
        test_app = create_app()

        @test_app.get("/test-unhandled-crash")
        def route_crash():
            raise RuntimeError("Database connection suddenly dropped")

        test_client = TestClient(test_app, raise_server_exceptions=False)
        response = test_client.get("/test-unhandled-crash")
        self.assertEqual(response.status_code, 500)

        headers = response.headers
        self.assertEqual(headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(headers.get("x-frame-options"), "DENY")
        self.assertEqual(headers.get("referrer-policy"), "strict-origin-when-cross-origin")
        self.assertEqual(headers.get("permissions-policy"), "geolocation=(), camera=(), microphone=()")
        self.assertIn("default-src 'none'", headers.get("content-security-policy", ""))

    # -------------------------------------------------------------------------
    # 6. HSTS Not Present on Local HTTP Development Requests
    # -------------------------------------------------------------------------
    def test_hsts_absent_on_local_http(self):
        """6. Verifies HSTS is NOT attached to local HTTP development requests."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("strict-transport-security", response.headers)

    # -------------------------------------------------------------------------
    # 7. HSTS Present on HTTPS Requests
    # -------------------------------------------------------------------------
    def test_hsts_present_on_forwarded_https(self):
        """7a. Verifies HSTS is attached when request indicates HTTPS via X-Forwarded-Proto."""
        response = self.client.get("/health", headers={"X-Forwarded-Proto": "https"})
        self.assertEqual(response.status_code, 200)

        hsts = response.headers.get("strict-transport-security")
        self.assertIsNotNone(hsts)
        self.assertEqual(hsts, "max-age=31536000; includeSubDomains")

    def test_hsts_present_on_direct_https_scheme(self):
        """7b. Verifies HSTS is attached when request scheme is directly https."""
        https_client = TestClient(self.app, base_url="https://api.industrial-ai.example.com")
        response = https_client.get("/health")
        self.assertEqual(response.status_code, 200)

        hsts = response.headers.get("strict-transport-security")
        self.assertIsNotNone(hsts)
        self.assertEqual(hsts, "max-age=31536000; includeSubDomains")

    # -------------------------------------------------------------------------
    # 8. HSTS Present when Production Configured
    # -------------------------------------------------------------------------
    def test_hsts_present_in_production_environment(self):
        """8. Verifies HSTS is attached when ENVIRONMENT is configured as production."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "CORS_ORIGINS": "https://app.industrial.example.com"}):
            reset_api_settings()
            prod_app = create_app()
            prod_client = TestClient(prod_app)
            response = prod_client.get("/health")

            self.assertEqual(response.status_code, 200)
            hsts = response.headers.get("strict-transport-security")
            self.assertIsNotNone(hsts)
            self.assertEqual(hsts, "max-age=31536000; includeSubDomains")

    # -------------------------------------------------------------------------
    # 9 & 10. Swagger UI (/docs) and ReDoc (/redoc) Compatibility
    # -------------------------------------------------------------------------
    def test_swagger_docs_responds_successfully_with_compatible_csp(self):
        """9. Verifies /docs responds HTTP 200 and receives the documentation-tailored CSP."""
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200)
        self.assertIn("html", response.headers.get("content-type", "").lower())

        csp = response.headers.get("content-security-policy", "")
        self.assertIn("cdn.jsdelivr.net", csp, "Docs CSP must allow jsdelivr CDN for Swagger assets.")
        self.assertIn("'unsafe-inline'", csp, "Docs CSP must allow inline scripts for Swagger initializer.")
        self.assertIn("fastapi.tiangolo.com", csp, "Docs CSP must allow tiangolo favicon.")

        # Baseline security headers must also remain present
        self.assertEqual(response.headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(response.headers.get("x-frame-options"), "DENY")
        self.assertEqual(response.headers.get("referrer-policy"), "strict-origin-when-cross-origin")

    def test_redoc_responds_successfully_with_compatible_csp(self):
        """10. Verifies /redoc responds HTTP 200 and receives the documentation-tailored CSP."""
        response = self.client.get("/redoc")
        self.assertEqual(response.status_code, 200)
        self.assertIn("html", response.headers.get("content-type", "").lower())

        csp = response.headers.get("content-security-policy", "")
        self.assertIn("cdn.jsdelivr.net", csp, "ReDoc CSP must allow jsdelivr CDN for ReDoc assets.")
        self.assertIn("fonts.googleapis.com", csp, "ReDoc CSP must allow Google Fonts.")
        self.assertIn("fonts.gstatic.com", csp, "ReDoc CSP must allow gstatic fonts.")

        # Baseline security headers must also remain present
        self.assertEqual(response.headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(response.headers.get("x-frame-options"), "DENY")
        self.assertEqual(response.headers.get("referrer-policy"), "strict-origin-when-cross-origin")

    # -------------------------------------------------------------------------
    # 11. REST API CSP Strictness
    # -------------------------------------------------------------------------
    def test_api_csp_is_strict_default_none(self):
        """11. Verifies REST API routes receive strict default-src 'none' policy."""
        response = self.client.get("/api/v1/status")
        csp = response.headers.get("content-security-policy", "")
        self.assertEqual(csp, "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")


if __name__ == "__main__":
    unittest.main()
