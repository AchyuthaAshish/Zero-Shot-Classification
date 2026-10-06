"""Unit and Integration Tests for Step 3.6 OpenAPI Documentation & API Contract.

Verifies:
A. OpenAPI generation succeeds
B. All expected endpoints exist
C. Correct HTTP methods exist
D. Classification request schema exists
E. Classification response schema exists
F. Multi-defect schemas exist
G. History schemas exist
H. Status schemas exist
I. Error schema exists
J. Error codes are documented
K. Taxonomy values are canonical
L. No secret strings appear in OpenAPI JSON
M. Examples are valid enough to serialize
N. /docs remains available
O. /redoc remains available
P. /openapi.json remains available
"""

import json
import unittest
from fastapi.testclient import TestClient

from api.main import create_app
from taxonomy.repository import get_taxonomy_repository


class TestAPIOpenAPI(unittest.TestCase):
    """Test suite for Phase 3 Step 3.6 OpenAPI Documentation & API Contract."""

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = TestClient(cls.app, raise_server_exceptions=False)
        cls.schema = cls.app.openapi()

    # -------------------------------------------------------------------------
    # Test A: OpenAPI Generation Succeeds
    # -------------------------------------------------------------------------
    def test_openapi_generation_succeeds(self):
        """A. Verifies OpenAPI specification generates successfully with title and version."""
        self.assertIsInstance(self.schema, dict)
        self.assertIn("openapi", self.schema)
        self.assertTrue(self.schema["openapi"].startswith("3."))

        info = self.schema.get("info", {})
        self.assertEqual(info.get("title"), "Industrial Defect Intelligence API")
        self.assertEqual(info.get("version"), "1.0.0")
        self.assertIn("description", info)
        self.assertIn("defect classification", info["description"].lower())

    # -------------------------------------------------------------------------
    # Test B: All Expected Endpoints Exist
    # -------------------------------------------------------------------------
    def test_all_expected_endpoints_exist(self):
        """B. Verifies all required endpoints are registered in OpenAPI paths."""
        paths = self.schema.get("paths", {})
        expected_endpoints = [
            "/",
            "/health",
            "/api/v1/status",
            "/api/v1/classify",
            "/api/v1/history",
            "/api/v1/history/{report_id}"
        ]
        for ep in expected_endpoints:
            self.assertIn(ep, paths, f"Missing endpoint in OpenAPI: {ep}")

    # -------------------------------------------------------------------------
    # Test C: Correct HTTP Methods Exist
    # -------------------------------------------------------------------------
    def test_correct_http_methods_exist(self):
        """C. Verifies correct HTTP methods are registered for each endpoint."""
        paths = self.schema.get("paths", {})
        self.assertIn("get", paths.get("/", {}))
        self.assertIn("get", paths.get("/health", {}))
        self.assertIn("get", paths.get("/api/v1/status", {}))
        self.assertIn("post", paths.get("/api/v1/classify", {}))
        self.assertIn("get", paths.get("/api/v1/history", {}))
        self.assertIn("get", paths.get("/api/v1/history/{report_id}", {}))

        # Ensure no unexpected mutations on classification
        self.assertNotIn("delete", paths.get("/api/v1/classify", {}))
        self.assertNotIn("put", paths.get("/api/v1/classify", {}))

    # -------------------------------------------------------------------------
    # Test D: Classification Request Schema Exists
    # -------------------------------------------------------------------------
    def test_classification_request_schema(self):
        """D. Verifies ClassificationRequest schema documents description constraints and modes."""
        schemas = self.schema.get("components", {}).get("schemas", {})
        self.assertIn("ClassificationRequest", schemas)

        req_schema = schemas["ClassificationRequest"]
        props = req_schema.get("properties", {})
        self.assertIn("description", props)
        self.assertIn("mode", props)

        # Check description constraints
        desc_prop = props["description"]
        self.assertEqual(desc_prop.get("maxLength"), 2000)
        self.assertEqual(desc_prop.get("minLength"), 1)

        # Check mode enum schema
        self.assertIn("ClassificationMode", schemas)
        mode_schema = schemas["ClassificationMode"]
        enum_values = mode_schema.get("enum", [])
        self.assertIn("local", enum_values)
        self.assertIn("gemini", enum_values)
        self.assertIn("hybrid", enum_values)
        self.assertIn("external", enum_values)

    # -------------------------------------------------------------------------
    # Test E: Classification Response Schema Exists
    # -------------------------------------------------------------------------
    def test_classification_response_schema(self):
        """E. Verifies ClassificationResponse and ClassificationData schemas document all key fields."""
        schemas = self.schema.get("components", {}).get("schemas", {})
        self.assertIn("ClassificationResponse", schemas)
        self.assertIn("ClassificationData", schemas)

        data_schema = schemas["ClassificationData"]
        props = data_schema.get("properties", {})

        key_fields = [
            "original_text",
            "is_multi_defect",
            "defect_count",
            "category",
            "status",
            "reliability",
            "explanation",
            "classification_mode",
            "provider",
            "model",
            "confidence",
            "confidence_level",
            "confidence_range",
            "raw_score",
            "calibrated_score",
            "top2_margin",
            "confidence_assessment",
            "ambiguity_assessment",
            "method",
            "validation",
            "defects"
        ]
        for field in key_fields:
            self.assertIn(field, props, f"Missing field in ClassificationData: {field}")

        # Check ConfidenceAssessmentSchema
        self.assertIn("ConfidenceAssessmentSchema", schemas)
        ca_props = schemas["ConfidenceAssessmentSchema"].get("properties", {})
        self.assertIn("level", ca_props)
        self.assertIn("approximate_range", ca_props)
        self.assertIn("raw_score", ca_props)
        self.assertIn("calibrated_prob", ca_props)
        self.assertIn("is_calibrated", ca_props)

    # -------------------------------------------------------------------------
    # Test F: Multi-Defect Schemas Exist
    # -------------------------------------------------------------------------
    def test_multidefect_schemas_exist(self):
        """F. Verifies multi-defect and child segment schemas are registered."""
        schemas = self.schema.get("components", {}).get("schemas", {})
        self.assertIn("MultiDefectValidationSchema", schemas)
        self.assertIn("PerDefectClassificationSchema", schemas)
        self.assertIn("ValidationIssueSchema", schemas)

        child_props = schemas["PerDefectClassificationSchema"].get("properties", {})
        self.assertIn("defect_id", child_props)
        self.assertIn("segment_id", child_props)
        self.assertIn("text", child_props)
        self.assertIn("category", child_props)
        self.assertIn("confidence_assessment", child_props)
        self.assertIn("source_start_char", child_props)
        self.assertIn("source_end_char", child_props)

    # -------------------------------------------------------------------------
    # Test G: History Schemas Exist
    # -------------------------------------------------------------------------
    def test_history_schemas_exist(self):
        """G. Verifies history list and detail response schemas are registered."""
        schemas = self.schema.get("components", {}).get("schemas", {})
        self.assertIn("HistoryListResponse", schemas)
        self.assertIn("HistoryListData", schemas)
        self.assertIn("HistoryDetailResponse", schemas)
        self.assertIn("DefectReportSummarySchema", schemas)
        self.assertIn("DefectReportDetailData", schemas)
        self.assertIn("DefectReportItemSchema", schemas)

        detail_props = schemas["DefectReportDetailData"].get("properties", {})
        self.assertIn("defects", detail_props)

    # -------------------------------------------------------------------------
    # Test H: Status Schemas Exist
    # -------------------------------------------------------------------------
    def test_status_schemas_exist(self):
        """H. Verifies health and diagnostics response schemas are registered."""
        schemas = self.schema.get("components", {}).get("schemas", {})
        self.assertIn("StatusResponse", schemas)
        self.assertIn("StatusData", schemas)
        self.assertIn("HealthChecks", schemas)
        self.assertIn("SubsystemCheck", schemas)
        self.assertIn("HealthStatus", schemas)

        health_enum = schemas["HealthStatus"].get("enum", [])
        self.assertEqual(set(health_enum), {"healthy", "degraded", "unhealthy"})

    # -------------------------------------------------------------------------
    # Test I: Error Schema Exists
    # -------------------------------------------------------------------------
    def test_error_schema_exists(self):
        """I. Verifies standardized ErrorResponse envelope is registered."""
        schemas = self.schema.get("components", {}).get("schemas", {})
        self.assertIn("ErrorResponse", schemas)
        self.assertIn("ErrorPayload", schemas)

        err_resp_props = schemas["ErrorResponse"].get("properties", {})
        self.assertIn("success", err_resp_props)
        self.assertIn("error", err_resp_props)

        payload_props = schemas["ErrorPayload"].get("properties", {})
        self.assertIn("code", payload_props)
        self.assertIn("message", payload_props)
        self.assertIn("details", payload_props)

    # -------------------------------------------------------------------------
    # Test J: Error Codes Are Documented
    # -------------------------------------------------------------------------
    def test_error_codes_documented(self):
        """J. Verifies all canonical error codes are defined in schemas or enums."""
        schemas = self.schema.get("components", {}).get("schemas", {})
        self.assertIn("ErrorCode", schemas)
        codes = schemas["ErrorCode"].get("enum", [])
        expected_codes = [
            "VALIDATION_ERROR",
            "NOT_FOUND",
            "SERVICE_UNAVAILABLE",
            "PROVIDER_ERROR",
            "DATABASE_ERROR",
            "INTERNAL_SERVER_ERROR",
            "HTTP_ERROR"
        ]
        for ec in expected_codes:
            self.assertIn(ec, codes, f"Missing error code: {ec}")

    # -------------------------------------------------------------------------
    # Test K: Taxonomy Values Are Canonical
    # -------------------------------------------------------------------------
    def test_taxonomy_values_are_canonical(self):
        """K. Verifies the approved 8-category industrial taxonomy is strictly preserved."""
        tax_repo = get_taxonomy_repository()
        categories = tax_repo.get_categories()
        expected_taxonomy = [
            "Mechanical Fault",
            "Electrical Fault",
            "Sensor Fault",
            "Temperature Fault",
            "Software Fault",
            "Power Supply Fault",
            "Communication Fault",
            "Unknown"
        ]
        self.assertEqual(len(categories), 8)
        self.assertEqual(set(categories), set(expected_taxonomy))

    # -------------------------------------------------------------------------
    # Test L: No Secret Strings Appear in OpenAPI JSON
    # -------------------------------------------------------------------------
    def test_no_secret_strings_appear_in_openapi_json(self):
        """L. Verifies no secret keys, JWT tokens, passwords, or credentials appear in OpenAPI."""
        raw_json = json.dumps(self.schema).lower()
        forbidden_substrings = [
            "eyj",  # Standard JWT token prefix
            "service_role",
            "postgres://",
            "postgresql://",
            "supabase_service",
            "supabase_key",
            "gemini_api_key",
            "aimlapi_key",
            "secret_key"
        ]
        for forbidden in forbidden_substrings:
            self.assertNotIn(forbidden, raw_json, f"Forbidden secret string detected in OpenAPI: {forbidden}")

    # -------------------------------------------------------------------------
    # Test M: Examples Are Valid Enough to Serialize
    # -------------------------------------------------------------------------
    def test_examples_are_valid_enough_to_serialize(self):
        """M. Verifies that all examples in the OpenAPI specification cleanly serialize to JSON."""
        raw_json = json.dumps(self.schema)
        self.assertIsInstance(raw_json, str)
        self.assertGreater(len(raw_json), 1000)

        # Re-parse to guarantee well-formedness
        reloaded = json.loads(raw_json)
        self.assertEqual(reloaded["info"]["title"], "Industrial Defect Intelligence API")

    # -------------------------------------------------------------------------
    # Test N: /docs Remains Available
    # -------------------------------------------------------------------------
    def test_docs_endpoint_available(self):
        """N. Verifies /docs Swagger UI is accessible and returns HTTP 200."""
        resp = self.client.get("/docs")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("html", resp.headers.get("content-type", "").lower())

    # -------------------------------------------------------------------------
    # Test O: /redoc Remains Available
    # -------------------------------------------------------------------------
    def test_redoc_endpoint_available(self):
        """O. Verifies /redoc ReDoc UI is accessible and returns HTTP 200."""
        resp = self.client.get("/redoc")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("html", resp.headers.get("content-type", "").lower())

    # -------------------------------------------------------------------------
    # Test P: /openapi.json Remains Available
    # -------------------------------------------------------------------------
    def test_openapi_json_endpoint_available(self):
        """P. Verifies /openapi.json is accessible and returns HTTP 200 with JSON schema."""
        resp = self.client.get("/openapi.json")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("application/json", resp.headers.get("content-type", "").lower())
        data = resp.json()
        self.assertEqual(data["info"]["title"], "Industrial Defect Intelligence API")


if __name__ == "__main__":
    unittest.main()
