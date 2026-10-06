"""Provider-neutral LLM Client with Google Gemini Implementation and Fake Client.

Conforms to SRS Section 12 (External API Requirements), NFR-004 (Testability),
NFR-007 (Fault tolerance), and FR-014 (Error handling).

Isolates all Google GenAI SDK specifics to this module.
Provides a FakeLLMClient for zero-cost, deterministic offline unit and integration testing.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
from core.schemas import LLMRawResponse
from core.exceptions import ModelError, ConfigurationError
from config.settings import get_settings, Settings
from llm.prompts import GeminiClassificationOutput

# Suppress harmless internal Google GenAI SDK recommendation logger regarding AFC in generate_content
logging.getLogger("google_genai.models").setLevel(logging.ERROR)


class BaseLLMClient(ABC):
    """Abstract provider-neutral LLM client interface."""

    @abstractmethod
    def classify(self, prompt: str, system_instruction: str) -> LLMRawResponse:
        """Submits a classification request and returns a structured LLMRawResponse."""
        pass


class GeminiLLMClient(BaseLLMClient):
    """Google Gemini Developer API client using google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        settings = get_settings()
        self.api_key = api_key or settings.llm_api_key
        self.model_name = model_name or settings.llm_model or "gemini-3.8-flash"
        self._client = None

    def _get_client(self):
        """Initializes the google.genai Client on demand, validating API key."""
        if self._client is not None:
            return self._client

        if not self.api_key or not self.api_key.strip():
            raise ConfigurationError(
                "Gemini API key is missing. Please set the LLM_API_KEY environment variable."
            )

        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key.strip())
            return self._client
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Google GenAI client: {e}") from e

    def verify_model_availability(self, model_name: Optional[str] = None) -> str:
        """
        Verifies that the requested model is accessible with the client's API key.
        Returns verified model name or raises ConfigurationError/ModelError.
        """
        target_model = model_name or self.model_name
        client = self._get_client()
        try:
            model_info = client.models.get(model=target_model)
            return getattr(model_info, "name", target_model)
        except Exception as e:
            err_str = str(e)
            if "API_KEY_INVALID" in err_str or "PERMISSION_DENIED" in err_str:
                raise ConfigurationError(f"Gemini API key is invalid or unauthorized: {e}") from e
            # Try listing available models to suggest valid alternatives
            try:
                available = [m.name for m in client.models.list() if "gemini" in getattr(m, "name", "").lower()]
                models_str = ", ".join(available[:5])
                raise ModelError(f"Model '{target_model}' unavailable for account. Available Gemini models: {models_str}") from e
            except ConfigurationError:
                raise
            except Exception:
                raise ModelError(f"Gemini model '{target_model}' verification failed: {e}") from e

    def classify(self, prompt: str, system_instruction: str) -> LLMRawResponse:
        """Calls the Gemini API with structured JSON output schema and bounded retries."""
        client = self._get_client()
        import time

        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=GeminiClassificationOutput,
                temperature=0.0,
                max_output_tokens=300
            )

            # Retry transient 429/5xx errors up to 2 times with backoff
            max_retries = 2
            response = None
            for attempt in range(max_retries + 1):
                try:
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=config
                    )
                    break
                except Exception as e:
                    err_msg = str(e)
                    is_transient = any(
                        m in err_msg.lower()
                        for m in ["resource_exhausted", "rate limit", "429", "503", "500", "deadline_exceeded", "unavailable"]
                    )
                    if is_transient and attempt < max_retries:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    raise

            raw_text = response.text if response else None
            if not raw_text or not raw_text.strip():
                raise ModelError("Gemini returned an empty response.")

            parsed = json.loads(raw_text)
            return LLMRawResponse(
                proposed_category=parsed.get("category", "Unknown"),
                reason=parsed.get("reason", "No reason provided."),
                detected_language=parsed.get("language"),
                raw_content=raw_text,
                reliability=parsed.get("reliability")
            )

        except json.JSONDecodeError as e:
            raise ModelError(f"Gemini response could not be parsed as JSON: {e}") from e
        except ConfigurationError:
            raise
        except Exception as e:
            # Map API errors, timeouts, rate limits to controlled ModelError
            err_msg = str(e)
            if "RESOURCE_EXHAUSTED" in err_msg or "rate limit" in err_msg.lower():
                raise ModelError(f"Gemini API rate limit exceeded: {e}") from e
            if "DEADLINE_EXCEEDED" in err_msg or "timeout" in err_msg.lower():
                raise ModelError(f"Gemini API request timed out: {e}") from e
            if "API_KEY_INVALID" in err_msg or "PERMISSION_DENIED" in err_msg:
                raise ConfigurationError(f"Gemini API key invalid or unauthorized: {e}") from e
            raise ModelError(f"Gemini API call failed: {e}") from e


class FakeLLMClient(BaseLLMClient):
    """
    Deterministic fake LLM client for testing without live API calls.
    Supports predefined mapping, rule-based fallback, and controlled error simulation.
    """

    def __init__(
        self,
        canned_responses: Optional[Dict[str, Union[Dict[str, str], Exception]]] = None,
        default_category: str = "Unknown"
    ):
        self.canned_responses = canned_responses or {}
        self.default_category = default_category
        self.call_history = []

    def classify(self, prompt: str, system_instruction: str) -> LLMRawResponse:
        self.call_history.append({"prompt": prompt, "system_instruction": system_instruction})

        # Check for specific simulated errors or canned responses
        lower_prompt = prompt.lower()
        for key, resp in self.canned_responses.items():
            if key.lower() in lower_prompt:
                if isinstance(resp, Exception):
                    raise resp
                if isinstance(resp, dict):

                    return LLMRawResponse(
                        proposed_category=resp.get("category", self.default_category),
                        reason=resp.get("reason", "Canned fake response."),
                        detected_language=resp.get("language", "English"),
                        raw_content=json.dumps(resp),
                        reliability=resp.get("reliability", "High" if resp.get("category", self.default_category) != "Unknown" else "Medium")
                    )

        # Rule-based fallback for standard industrial phrases if no canned match
        lower_prompt = prompt.lower()

        if "vibration" in lower_prompt or "bearing" in lower_prompt or "gear" in lower_prompt or "వైబ్రేట్" in lower_prompt:
            category = "Mechanical Fault"
            reason = "Mechanical abnormality indicated by vibration or component noise."
        elif "short circuit" in lower_prompt or "wiring" in lower_prompt or "sparking" in lower_prompt:
            category = "Electrical Fault"
            reason = "Electrical circuit or wiring failure detected."
        elif "sensor" in lower_prompt or "reading incorrect" in lower_prompt or "calibration" in lower_prompt:
            category = "Sensor Fault"
            reason = "Measurement or sensor reading defect indicated."
        elif "overheating" in lower_prompt or "temperature above" in lower_prompt or "thermal" in lower_prompt:
            category = "Temperature Fault"
            reason = "Abnormal physical temperature condition reported."
        elif "software" in lower_prompt or "crash" in lower_prompt or "firmware" in lower_prompt or "bug" in lower_prompt:
            category = "Software Fault"
            reason = "Program logic or software failure reported."
        elif "power not reaching" in lower_prompt or "power supply" in lower_prompt or "battery" in lower_prompt:
            category = "Power Supply Fault"
            reason = "Loss or instability of incoming electrical power."
        elif "wifi" in lower_prompt or "mqtt" in lower_prompt or "can bus" in lower_prompt or "communication" in lower_prompt:
            category = "Communication Fault"
            reason = "Data link or network communication disruption reported."
        else:
            category = self.default_category
            reason = "Insufficient evidence to distinguish a specific fault category."

        fake_rel = "Medium" if category == self.default_category else "High"
        return LLMRawResponse(
            proposed_category=category,
            reason=reason,
            detected_language="English",
            raw_content=json.dumps({"category": category, "reason": reason, "reliability": fake_rel}),
            reliability=fake_rel
        )


class AIMLAPIClient(BaseLLMClient):
    """AI/ML API client using OpenAI-compatible chat completions over httpx."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0
    ):
        settings = get_settings()
        self.api_key = api_key or settings.llm_api_key
        raw_model = model_name or settings.llm_model or "z-ai/glm-5-turbo"
        # Map user-friendly model aliases to the AI/ML API accepted model identifier
        if raw_model.strip().lower() in ("glm 5 turbo", "glm-5-turbo", "glm5-turbo", "glm-5 turbo"):
            self.model_name = "z-ai/glm-5-turbo"
        else:
            self.model_name = raw_model.strip()
        self.base_url = (base_url or getattr(settings, "llm_base_url", "https://api.aimlapi.com/v1")).rstrip("/")
        self.timeout = timeout

    def classify(self, prompt: str, system_instruction: str) -> LLMRawResponse:
        """Calls the AI/ML API chat completions endpoint with structured JSON output."""
        if not self.api_key or not self.api_key.strip():
            raise ConfigurationError(
                "AI/ML API key is missing. Please set the LLM_API_KEY environment variable."
            )

        import httpx

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json"
        }

        full_system = (
            f"{system_instruction}\n\n"
            "STRICT RESPONSE FORMAT:\n"
            "You MUST respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "category": "<one of the 8 approved taxonomy categories>",\n'
            '  "reason": "<concise evidence-based reason>",\n'
            '  "language": "<English | Telugu | Telugu-English>",\n'
            '  "reliability": "High" | "Medium" | "Low"\n'
            "}\n"
            "Do NOT output markdown code blocks, backticks, or text outside the JSON object."
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": full_system},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=payload)

            if response.status_code == 401:
                raise ConfigurationError(
                    "AI/ML API key is invalid or unauthorized. Please verify your credentials."
                )
            elif response.status_code == 403:
                body_lower = response.text.lower()
                if any(w in body_lower for w in ["run out of funds", "balance", "billing", "payment", "quota"]):
                    raise ModelError(
                        "AI/ML API credit balance is exhausted. Please top up your account balance or use Local ML mode."
                    )
                raise ConfigurationError(
                    "AI/ML API request forbidden. Please verify your account permissions."
                )
            elif response.status_code == 429:
                raise ModelError(
                    "AI/ML API rate limit exceeded. Please wait or switch to Local ML mode."
                )
            elif response.status_code >= 400:
                raise ModelError(
                    f"AI/ML API returned HTTP {response.status_code}."
                )

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise ModelError("AI/ML API returned an empty choices list.")

            content = choices[0].get("message", {}).get("content", "")
            if not content or not content.strip():
                raise ModelError("AI/ML API returned empty content.")

            cleaned = content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            return LLMRawResponse(
                proposed_category=parsed.get("category", "Unknown"),
                reason=parsed.get("reason", "No reason provided."),
                detected_language=parsed.get("language"),
                raw_content=content,
                reliability=parsed.get("reliability")
            )

        except ConfigurationError:
            raise
        except ModelError:
            raise
        except json.JSONDecodeError as e:
            raise ModelError(f"AI/ML API response could not be parsed as JSON: {e}") from e
        except httpx.TimeoutException as e:
            raise ModelError("AI/ML API request timed out.") from e
        except Exception as e:
            # Mask API key if it ever appears in exception string
            safe_err = str(e).replace(self.api_key.strip(), "***MASKED***")
            raise ModelError(f"AI/ML API request failed: {safe_err}") from e


def get_llm_client(force_fake: bool = False) -> BaseLLMClient:
    """
    Factory creating configured LLM client.
    Does NOT silently fall back to FakeLLMClient unless explicitly requested via
    force_fake=True or LLM_PROVIDER=fake.
    """
    if force_fake:
        return FakeLLMClient()

    settings = get_settings()
    provider = settings.llm_provider

    if provider == "gemini":
        return GeminiLLMClient()
    elif provider in ("aimlapi", "openai"):
        return AIMLAPIClient()
    elif provider == "fake":
        return FakeLLMClient()
    else:
        raise ConfigurationError(
            f"Unsupported LLM provider '{provider}'. Supported providers are: 'gemini', 'aimlapi', 'fake'."
        )

