"""Unit and Integration Tests for Phase 5 Step 5.8: Secrets & Security Audit.

Verifies:
1. Git repository tracking: No real .env files, private keys, or credential files are tracked.
2. .env.example hygiene: Only placeholder values are present for sensitive variables.
3. Configuration environment isolation: System settings dynamically pull from environment variables without hard-coded defaults.
4. Error sanitization defense: _sanitize_error reliably masks keys, URLs, JWT secrets, and connection strings.
5. Safe failure handling: Unconfigured client initializations raise controlled ConfigurationError without information disclosure.
"""

import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from config.settings import Settings, load_settings, reset_settings
from database.supabase_client import _sanitize_error, get_supabase_client, reset_supabase_client
from llm.client import GeminiLLMClient
from core.exceptions import ConfigurationError


class TestSecretsAndSecurityAudit(unittest.TestCase):
    """Test suite verifying secrets management, configuration security, and error sanitization."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent

    def setUp(self):
        reset_settings()
        reset_supabase_client()

    def tearDown(self):
        reset_settings()
        reset_supabase_client()

    # -------------------------------------------------------------------------
    # 1. Git Tracking Hygiene
    # -------------------------------------------------------------------------
    def test_no_real_env_files_tracked_in_git(self):
        """1. Verifies no .env file (other than .env.example) is tracked in Git."""
        result = subprocess.check_output(["git", "ls-files"], cwd=self.repo_root, text=True)
        tracked_files = [line.strip() for line in result.splitlines() if line.strip()]

        for path in tracked_files:
            lower = path.lower()
            if lower == ".env.example":
                continue
            self.assertFalse(
                lower.endswith(".env") or "/.env" in lower or ".env." in lower,
                f"Secret environment file is tracked by git: {path}"
            )

    def test_no_private_keys_or_credential_files_tracked(self):
        """2. Verifies no private keys (*.pem, *.key) or service account credentials are tracked."""
        result = subprocess.check_output(["git", "ls-files"], cwd=self.repo_root, text=True)
        tracked_files = [line.strip() for line in result.splitlines() if line.strip()]

        forbidden_extensions = (".pem", ".key", ".pkcs12", ".pfx", ".crt", ".cert")
        for path in tracked_files:
            lower = path.lower()
            for ext in forbidden_extensions:
                self.assertFalse(
                    lower.endswith(ext),
                    f"Sensitive key/cert file tracked by git: {path}"
                )
            self.assertNotEqual(
                os.path.basename(path),
                "credentials.json",
                f"credentials.json must not be tracked by git: {path}"
            )

    # -------------------------------------------------------------------------
    # 2. .env.example Sanitization & Placeholders
    # -------------------------------------------------------------------------
    def test_env_example_contains_only_placeholders(self):
        """3. Verifies .env.example contains only empty placeholder values for secret variables."""
        example_path = self.repo_root / ".env.example"
        self.assertTrue(example_path.is_file(), ".env.example file must exist.")

        sensitive_keys = {
            "LLM_API_KEY",
            "SUPABASE_URL",
            "SUPABASE_KEY",
            "SUPABASE_JWT_SECRET",
        }

        with open(example_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key in sensitive_keys:
                        self.assertEqual(
                            val,
                            "",
                            f"Sensitive variable '{key}' in .env.example must be an empty placeholder, found: {val}"
                        )

    # -------------------------------------------------------------------------
    # 3. Application Configuration Isolation
    # -------------------------------------------------------------------------
    def test_settings_credentials_default_to_none_when_unset(self):
        """4. Verifies Settings defaults sensitive credentials to None when environment is clean."""
        with patch.dict(os.environ, {}, clear=True):
            settings = load_settings(auto_load_env=False)
            self.assertIsNone(settings.llm_api_key)
            self.assertIsNone(settings.supabase_url)
            self.assertIsNone(settings.supabase_key)
            self.assertIsNone(settings.supabase_jwt_secret)

    def test_settings_correctly_reads_environment_credentials(self):
        """5. Verifies Settings reads credentials from environment variables."""
        test_env = {
            "LLM_API_KEY": "test_mock_gemini_key",
            "SUPABASE_URL": "https://test-project.supabase.co",
            "SUPABASE_KEY": "test_mock_supabase_anon_key",
            "SUPABASE_JWT_SECRET": "test_mock_jwt_secret_32bytes_long",
        }
        with patch.dict(os.environ, test_env, clear=True):
            settings = load_settings(auto_load_env=False)
            self.assertEqual(settings.llm_api_key, "test_mock_gemini_key")
            self.assertEqual(settings.supabase_url, "https://test-project.supabase.co")
            self.assertEqual(settings.supabase_key, "test_mock_supabase_anon_key")
            self.assertEqual(settings.supabase_jwt_secret, "test_mock_jwt_secret_32bytes_long")

    # -------------------------------------------------------------------------
    # 4. Error Sanitization Defense
    # -------------------------------------------------------------------------
    def test_sanitize_error_masks_all_configured_secrets(self):
        """6. Verifies _sanitize_error strips configured keys, URLs, and JWT secrets."""
        test_env = {
            "SUPABASE_KEY": "sb_secret_token_12345",
            "SUPABASE_URL": "https://secret-project-id.supabase.co",
            "SUPABASE_JWT_SECRET": "jwt_secret_token_67890",
            "LLM_API_KEY": "gemini_secret_api_key_abcde",
        }
        with patch.dict(os.environ, test_env, clear=True):
            raw_msg = (
                "Failed request to https://secret-project-id.supabase.co with key "
                "sb_secret_token_12345 and jwt jwt_secret_token_67890 and gemini_secret_api_key_abcde"
            )
            cleaned = _sanitize_error(raw_msg)
            self.assertNotIn("sb_secret_token_12345", cleaned)
            self.assertNotIn("https://secret-project-id.supabase.co", cleaned)
            self.assertNotIn("jwt_secret_token_67890", cleaned)
            self.assertNotIn("gemini_secret_api_key_abcde", cleaned)
            self.assertIn("***MASKED***", cleaned)
            self.assertIn("***URL***", cleaned)

    def test_sanitize_error_masks_database_connection_urls_and_api_keys(self):
        """7. Verifies _sanitize_error masks raw PostgreSQL connection URLs and Google API keys."""
        # Construct synthetic token at runtime to exercise sanitizer without static scanner false positives
        synthetic_google_key = "".join(["AI", "za"]) + ("SyntheticKeyFixture" * 3)[:35]
        raw_msg = (
            "Crash in postgresql://postgres:SuperSecretPassword99@db.supabase.co:5432/postgres "
            f"using key {synthetic_google_key}"
        )
        cleaned = _sanitize_error(raw_msg)
        self.assertNotIn("SuperSecretPassword99", cleaned)
        self.assertNotIn(synthetic_google_key, cleaned)
        self.assertIn("***MASKED***", cleaned)
        self.assertIn("***API_KEY_MASKED***", cleaned)

    # -------------------------------------------------------------------------
    # 5. Missing Credentials Safe Failure
    # -------------------------------------------------------------------------
    def test_supabase_client_raises_controlled_configuration_error_when_unconfigured(self):
        """8. Verifies get_supabase_client raises controlled ConfigurationError without leaks."""
        mock_s = Settings(supabase_url=None, supabase_key=None)
        with patch("database.supabase_client.get_settings", return_value=mock_s):
            with self.assertRaises(ConfigurationError) as ctx:
                get_supabase_client()
            err_msg = str(ctx.exception)
            self.assertIn("SUPABASE_URL", err_msg)
            self.assertIn("SUPABASE_KEY", err_msg)

    def test_gemini_client_raises_controlled_configuration_error_when_api_key_missing(self):
        """9. Verifies GeminiLLMClient raises controlled ConfigurationError when key is missing."""
        mock_s = Settings(llm_api_key=None)
        with patch("llm.client.get_settings", return_value=mock_s):
            client = GeminiLLMClient(api_key=None)
            with self.assertRaises(ConfigurationError) as ctx:
                client._get_client()
            err_msg = str(ctx.exception)
            self.assertIn("LLM_API_KEY", err_msg)


if __name__ == "__main__":
    unittest.main()
