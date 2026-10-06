"""Application Configuration for Zero-Shot Industrial Defect Classification.

Conforms to SRS Section 19 and PRD Section 11 (F1).
Centralizes system settings and provisional defaults pending project owner confirmation.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from core.exceptions import ConfigurationError

# Base directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent

# Provisional default input character limit per PRD Section 11 (F1) and SRS Section 4 (FR-001)
# Documented provisional default: 2,000 characters unless modified by project owner.
DEFAULT_MAX_INPUT_LENGTH: int = 2000

# Classification Modes: hybrid (local first with external fallback), local (offline ML only), external/gemini (API only)
DEFAULT_CLASSIFICATION_MODE: str = "hybrid"
SUPPORTED_CLASSIFICATION_MODES: frozenset[str] = frozenset(["hybrid", "local", "gemini", "external"])
DEFAULT_LOCAL_CONFIDENCE_THRESHOLD: float = 0.70

# Approved default LLM settings
DEFAULT_LLM_PROVIDER: str = "gemini"
DEFAULT_LLM_MODEL: str = "gemini-3.8-flash"
SUPPORTED_LLM_PROVIDERS: frozenset[str] = frozenset(["gemini", "aimlapi", "fake"])
DEFAULT_LLM_BASE_URL: str = "https://api.aimlapi.com/v1"


@dataclass(frozen=True)
class Settings:
    """System configuration container."""
    max_input_length: int = DEFAULT_MAX_INPUT_LENGTH
    taxonomy_file_path: Path = BASE_DIR / "taxonomy" / "defect_taxonomy.json"
    classification_mode: str = DEFAULT_CLASSIFICATION_MODE
    local_confidence_threshold: float = DEFAULT_LOCAL_CONFIDENCE_THRESHOLD
    llm_provider: str = DEFAULT_LLM_PROVIDER
    llm_model: str = DEFAULT_LLM_MODEL
    llm_api_key: Optional[str] = None
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    supabase_jwt_secret: Optional[str] = None
    semantic_retrieval_enabled: bool = False
    verification_enabled: bool = False


def load_env_file(env_path: Optional[Path] = None) -> None:
    """
    Lightweight, dependency-free local .env loader for development.
    Populates os.environ without overwriting pre-existing environment variables.
    """
    path = env_path or (BASE_DIR / ".env")
    if not path.is_file():
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception:
        # File access errors or encoding issues in local .env safely ignored
        pass


def load_settings(auto_load_env: bool = True) -> Settings:
    """
    Loads configuration settings, reading overrides from environment variables.
    Safely validates MAX_INPUT_LENGTH (falling back to 2,000 for invalid/zero/negative values).
    Strictly validates LLM_PROVIDER against SUPPORTED_LLM_PROVIDERS.
    """
    if auto_load_env:
        load_env_file()

    raw_max_len = os.getenv("MAX_INPUT_LENGTH")
    if raw_max_len is not None and raw_max_len.strip():
        try:
            parsed_len = int(raw_max_len.strip())
            if parsed_len <= 0:
                max_input_length = DEFAULT_MAX_INPUT_LENGTH
            else:
                max_input_length = parsed_len
        except ValueError:
            max_input_length = DEFAULT_MAX_INPUT_LENGTH
    else:
        max_input_length = DEFAULT_MAX_INPUT_LENGTH

    raw_tax_path = os.getenv("TAXONOMY_PATH")
    if raw_tax_path and raw_tax_path.strip():
        taxonomy_file_path = Path(raw_tax_path.strip())
    else:
        taxonomy_file_path = BASE_DIR / "taxonomy" / "defect_taxonomy.json"

    raw_mode = os.getenv("CLASSIFICATION_MODE", DEFAULT_CLASSIFICATION_MODE).strip().lower()
    if raw_mode not in SUPPORTED_CLASSIFICATION_MODES:
        raw_mode = DEFAULT_CLASSIFICATION_MODE

    raw_thresh = os.getenv("LOCAL_CONFIDENCE_THRESHOLD")
    if raw_thresh is not None and raw_thresh.strip():
        try:
            parsed_thresh = float(raw_thresh.strip())
            if 0.0 <= parsed_thresh <= 1.0:
                local_confidence_threshold = parsed_thresh
            else:
                local_confidence_threshold = DEFAULT_LOCAL_CONFIDENCE_THRESHOLD
        except ValueError:
            local_confidence_threshold = DEFAULT_LOCAL_CONFIDENCE_THRESHOLD
    else:
        local_confidence_threshold = DEFAULT_LOCAL_CONFIDENCE_THRESHOLD

    raw_provider = (
        os.getenv("EXTERNAL_PROVIDER")
        or os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
    ).strip().lower()

    if raw_provider in ("ai/ml api", "aiml_api", "aiml", "aimlapi", "openai", "glm", "glm-5-turbo", "glm5"):
        llm_provider = "aimlapi"
    elif raw_provider in ("gemini", "google", "google-genai"):
        llm_provider = "gemini"
    else:
        llm_provider = raw_provider

    if llm_provider not in SUPPORTED_LLM_PROVIDERS:
        supported_str = ", ".join(f"'{p}'" for p in sorted(list(SUPPORTED_LLM_PROVIDERS)))
        raise ConfigurationError(
            f"Unsupported LLM provider '{llm_provider}'. Supported providers are: {supported_str}."
        )

    raw_model = os.getenv("LLM_MODEL")
    if raw_model and raw_model.strip():
        llm_model = raw_model.strip()
    else:
        llm_model = "gemini-3.8-flash" if llm_provider == "gemini" else "z-ai/glm-5-turbo"

    llm_api_key = os.getenv("LLM_API_KEY")
    llm_base_url = os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL).strip()

    raw_sb_url = os.getenv("SUPABASE_URL")
    supabase_url = raw_sb_url.strip() if (raw_sb_url and raw_sb_url.strip()) else None

    raw_sb_key = os.getenv("SUPABASE_KEY")
    supabase_key = raw_sb_key.strip() if (raw_sb_key and raw_sb_key.strip()) else None

    raw_sb_jwt = os.getenv("SUPABASE_JWT_SECRET")
    supabase_jwt_secret = raw_sb_jwt.strip() if (raw_sb_jwt and raw_sb_jwt.strip()) else None

    return Settings(
        max_input_length=max_input_length,
        taxonomy_file_path=taxonomy_file_path,
        classification_mode=raw_mode,
        local_confidence_threshold=local_confidence_threshold,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        supabase_jwt_secret=supabase_jwt_secret,
        semantic_retrieval_enabled=False,
        verification_enabled=False
    )



_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Returns the singleton application settings instance."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings

def reset_settings() -> None:
    """Resets the singleton application settings instance (useful in tests)."""
    global _settings
    _settings = None
