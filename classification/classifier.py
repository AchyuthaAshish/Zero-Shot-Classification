"""Classification Orchestrator for Industrial Defect Classification.

Supports:
1. LOCAL mode: Free offline ML classification (Zero Gemini calls).
2. GEMINI mode: Zero-shot LLM classification with strict taxonomy validation.
3. HYBRID mode (default): Fast local ML first; falls back to Gemini if
   confidence is below threshold; safely falls back to local ML if Gemini is
   unavailable, unconfigured, or quota-exhausted.

Conforms to SRS Section 3 (C6), Section 4 (FR-004, FR-005, FR-014),
and BR-001, BR-002, BR-006, BR-009, BR-010.
"""

import os
import sys
from typing import Optional
from core.schemas import ClassificationResult, PreprocessedInput

# Ensure standard output supports multilingual UTF-8 display (e.g. Telugu script) on Windows terminals
if hasattr(sys, "stdout") and sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from core.exceptions import (
    InputError,
    ModelError,
    ConfigurationError,
    ValidationError,
    TaxonomyError,
    DefectClassificationException
)
from core.decision_engine import DecisionEngine
from preprocessing.text_processor import get_text_processor, TextProcessor
from taxonomy.repository import get_taxonomy_repository, TaxonomyRepository
from llm.client import BaseLLMClient, get_llm_client
from llm.prompts import build_system_instruction, build_classification_prompt
from config.settings import get_settings
from ml.local_classifier import LocalDefectClassifier, get_local_classifier


class DefectClassifier:
    """Orchestrates defect classification with Local ML and Gemini fallback support."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        text_processor: Optional[TextProcessor] = None,
        taxonomy_repo: Optional[TaxonomyRepository] = None,
        decision_engine: Optional[DecisionEngine] = None,
        local_classifier: Optional[LocalDefectClassifier] = None,
        mode: Optional[str] = None
    ):
        self.text_processor = text_processor or get_text_processor()
        self.taxonomy_repo = taxonomy_repo or get_taxonomy_repository()
        self.decision_engine = decision_engine or DecisionEngine()
        self._explicit_llm_client = (llm_client is not None)
        self.llm_client = llm_client or get_llm_client()
        self._local_classifier = local_classifier
        self._mode = mode

    @property
    def local_classifier(self) -> LocalDefectClassifier:
        """Lazy-loaded local classifier."""
        if self._local_classifier is None:
            self._local_classifier = get_local_classifier()
        return self._local_classifier

    @property
    def active_mode(self) -> str:
        """Resolves active classification mode."""
        if self._mode is not None:
            return self._mode.lower()
        env_mode = os.getenv("CLASSIFICATION_MODE")
        if env_mode and env_mode.strip():
            return env_mode.strip().lower()
        # If an explicit llm_client was injected (e.g. in test suites testing LLM mocks)
        # and CLASSIFICATION_MODE was not explicitly set in os.environ,
        # default to "gemini" to preserve LLM mock unit tests.
        if self._explicit_llm_client:
            return "gemini"
        return get_settings().classification_mode.lower()

    def _classify_gemini(
        self,
        raw_defect_text: Optional[str],
        processed_input: Optional[PreprocessedInput] = None,
        _is_subsegment: bool = False
    ) -> ClassificationResult:
        """Executes zero-shot Gemini classification path."""
        safe_raw_text = raw_defect_text or ""

        # 1. Preprocessing if not already provided
        if processed_input is None:
            try:
                processed_input = self.text_processor.preprocess(raw_defect_text)
            except InputError as e:
                return self.decision_engine.create_error_result(
                    raw_text=safe_raw_text,
                    error_message=e.message,
                    error_type="input_error"
                )
            except Exception as e:
                return self.decision_engine.create_error_result(
                    raw_text=safe_raw_text,
                    error_message=f"Unexpected input processing error: {e}",
                    error_type="input_error"
                )

        # 2. Build Controlled System Instruction & Prompt
        try:
            system_instruction = build_system_instruction(self.taxonomy_repo)
            prompt = build_classification_prompt(processed_input.normalized_text)
        except TaxonomyError as e:
            return self.decision_engine.create_error_result(
                raw_text=safe_raw_text,
                error_message=e.message,
                error_type="taxonomy_error"
            )

        # 3. Invoke LLM Client
        try:
            llm_raw_response = self.llm_client.classify(
                prompt=prompt,
                system_instruction=system_instruction
            )
        except ConfigurationError as e:
            return self.decision_engine.create_error_result(
                raw_text=safe_raw_text,
                error_message=e.message,
                error_type="configuration_error"
            )
        except ModelError as e:
            return self.decision_engine.create_error_result(
                raw_text=safe_raw_text,
                error_message=e.message,
                error_type="model_error"
            )
        except Exception as e:
            return self.decision_engine.create_error_result(
                raw_text=safe_raw_text,
                error_message=f"Unexpected external model failure: {e}",
                error_type="model_error"
            )

        # 4. Resolve Decision and Strict Validation
        try:
            res = self.decision_engine.decide_from_llm(
                processed_input=processed_input,
                llm_response=llm_raw_response,
                verification_feedback=None,
                _is_subsegment=_is_subsegment,
                defect_classifier=self
            )
            provider = get_settings().llm_provider
            res.model_source = "aimlapi" if (provider == "aimlapi" or self.active_mode in ("aimlapi", "openai")) else "gemini"
            return res
        except Exception as e:
            return self.decision_engine.create_error_result(
                raw_text=safe_raw_text,
                error_message=f"Classification decision error: {e}",
                error_type="system_error"
            )

    def classify(
        self,
        raw_defect_text: Optional[str],
        mode_override: Optional[str] = None,
        _is_subsegment: bool = False
    ) -> ClassificationResult:
        """
        Executes classification according to configured mode (LOCAL, GEMINI/EXTERNAL, or HYBRID).
        """
        mode = (mode_override or self.active_mode).lower()
        safe_raw_text = raw_defect_text or ""

        # Validate input with text_processor first
        try:
            processed_input = self.text_processor.preprocess(raw_defect_text)
        except InputError as e:
            return self.decision_engine.create_error_result(
                raw_text=safe_raw_text,
                error_message=e.message,
                error_type="input_error"
            )
        except Exception as e:
            return self.decision_engine.create_error_result(
                raw_text=safe_raw_text,
                error_message=f"Unexpected input processing error: {e}",
                error_type="input_error"
            )

        # Mode 1: Pure LOCAL ML (Zero API requests)
        if mode == "local":
            return self.local_classifier.classify(raw_defect_text, _is_subsegment=_is_subsegment)

        # Mode 2: Pure external LLM zero-shot (Gemini or AI/ML API)
        if mode in ("gemini", "external", "aimlapi"):
            return self._classify_gemini(raw_defect_text, processed_input=processed_input, _is_subsegment=_is_subsegment)

        # Mode 3: HYBRID (Local ML first; external LLM fallback for low confidence or Unknown)
        local_result = self.local_classifier.classify(raw_defect_text, _is_subsegment=_is_subsegment)

        # Handle multi-defect in hybrid mode
        if not _is_subsegment and local_result.segmentation_result and local_result.segmentation_result.is_multi_defect and local_result.segmentation_result.segment_count >= 2:
            from core.per_defect_classifier import get_per_defect_classifier
            multi_defect_classification = get_per_defect_classifier().classify_segments(
                segments=local_result.segmentation_result.segments,
                original_text=safe_raw_text,
                mode="hybrid",
                multi_defect_assessment=local_result.multi_defect_assessment,
                segmentation_result=local_result.segmentation_result,
                local_classifier=self.local_classifier,
                defect_classifier=self
            )
            local_result.multi_defect_classification = multi_defect_classification
            return local_result

        threshold = get_settings().local_confidence_threshold

        is_high_conf = (
            local_result.confidence is not None
            and local_result.confidence >= threshold
            and local_result.category != "Unknown"
            and local_result.status == "success"
            and not (local_result.ambiguity_assessment and local_result.ambiguity_assessment.is_ambiguous)
        )

        # High confidence, unambiguous known category -> Return local result immediately
        if is_high_conf:
            return local_result

        # Check if local result is Unknown due to non-defect / unrelated text (Phase 1 Step 1.3)
        from core.unknown_detector import get_unknown_detector
        ev = get_unknown_detector().analyze_evidence(raw_defect_text)
        if local_result.category == "Unknown" and (ev.is_unrelated_or_non_defect or (ev.is_trivial_length and ev.total_evidence_count == 0)):
            # Do not make unnecessary external API calls for greetings / non-defect / trivial inputs
            return local_result

        # Low confidence, ambiguous, or technical unknown -> Request external LLM fallback
        try:
            ext_result = self._classify_gemini(raw_defect_text, processed_input=processed_input, _is_subsegment=_is_subsegment)
            # If external model succeeds with a valid result, return it
            if ext_result.status in ("success", "unknown"):
                provider = get_settings().llm_provider
                ext_result.model_source = "local_ml_fallback_aimlapi" if provider == "aimlapi" else "local_ml_fallback_gemini"
                return ext_result
            else:
                # External model produced an error. Safely fall back to the local prediction!
                local_result.model_source = "local_ml_fallback"
                return local_result
        except Exception:
            # External provider unavailable/unconfigured/network down -> Safe fallback to local prediction
            local_result.model_source = "local_ml_fallback"
            return local_result


# Global singleton classifier instance
_default_classifier: Optional[DefectClassifier] = None


def get_defect_classifier(force_fake: bool = False, mode: Optional[str] = None) -> DefectClassifier:
    """Returns singleton or newly configured DefectClassifier instance."""
    global _default_classifier
    if force_fake:
        return DefectClassifier(llm_client=get_llm_client(force_fake=True), mode=mode or "gemini")
    if mode is not None:
        return DefectClassifier(mode=mode)
    if _default_classifier is None:
        _default_classifier = DefectClassifier()
    return _default_classifier


def reset_defect_classifier() -> None:
    """Resets the singleton classifier instance (useful for test isolation)."""
    global _default_classifier
    _default_classifier = None


def classify_defect(
    description: Optional[str],
    classifier: Optional[DefectClassifier] = None,
    mode: Optional[str] = None
) -> ClassificationResult:
    """
    Public reusable service function to classify an industrial defect description.
    Delegates to the configured DefectClassifier pipeline and returns canonical ClassificationResult.
    """
    active_classifier = classifier or get_defect_classifier(mode=mode)
    return active_classifier.classify(description, mode_override=mode)


def classify_per_defect(
    description: Optional[str],
    mode: Optional[str] = None
):
    """
    Public convenience service function to independently classify each defect segment in a report.
    Returns MultiDefectClassificationResult.
    """
    from core.per_defect_classifier import get_per_defect_classifier
    classifier = get_per_defect_classifier()
    return classifier.classify_report(description, mode=mode or "local")
