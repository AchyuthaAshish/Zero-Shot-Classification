"""Input Preprocessor for Zero-Shot Industrial Defect Classification.

Conforms to PRD Section 9 (F1, F2), SRS Section 3 (C2), FR-001, FR-002,
and Business Rules BR-006, BR-008.

Responsibilities:
1. Validates non-empty input and preserves original user text.
2. Performs lightweight normalization without modifying technical vocabulary.
3. Detects English, Telugu script, and Telugu-English code-switched input.
4. Ensures non-English/code-switched inputs are NOT rejected.

NOTE ON LOCAL-LANGUAGE SCOPE:
English, Telugu script, and Telugu-English code-switch detection/preparation are implemented.
Multilingual semantic classification still requires LLM integration and evaluation.
"""


import re
from typing import Optional
from core.schemas import PreprocessedInput
from core.exceptions import InputError
from config.settings import get_settings

# Telugu Unicode block range: \u0c00 - \u0c7f
TELUGU_SCRIPT_REGEX = re.compile(r"[\u0c00-\u0c7f]")

# Common transliterated / Romanized Telugu markers found in industrial code-switched reporting
# Examples: 'Motor lo unusual sound vastundi', 'chala heat avtundi', 'bearing lo noise vachindi'
TELUGU_ROMANIZED_MARKERS = {
    "lo", "undi", "undhi", "vastundi", "vasthundi", "vastundhi", "vachindi", "vachindhi",
    "chesindi", "chesindhi", "avtundi", "avthundi", "avutondi", "avutundi", "ayindi", "chala",
    "baga", "bagundi", "ledhu", "ledu", "kuda", "kani", "mariyu"
}


class TextProcessor:
    """Preprocesses and analyzes natural-language defect descriptions."""

    def __init__(self, max_length: Optional[int] = None):
        if max_length is not None:
            self._max_length = max_length
        else:
            self._max_length = get_settings().max_input_length


    def preprocess(self, text: Optional[str]) -> PreprocessedInput:
        """
        Validates, normalizes, and identifies language context of raw defect text.

        Raises:
            InputError: If text is None, empty, or whitespace-only.
        """
        if text is None:
            raise InputError("Defect description cannot be None.")

        # Preservation of raw text exactly as provided
        raw_text = text

        # Check for whitespace-only
        stripped = raw_text.strip()
        if not stripped:
            raise InputError("Defect description cannot be empty or whitespace-only.")

        if len(stripped) > self._max_length:
            raise InputError(f"Defect description exceeds maximum length of {self._max_length} characters.")

        # Lightweight normalization: collapse consecutive whitespace into single space
        normalized = re.sub(r"\s+", " ", stripped)

        detected_language, is_code_switched = self._detect_language(normalized)

        words = normalized.split()
        char_count = len(normalized)
        word_count = len(words)

        return PreprocessedInput(
            raw_text=raw_text,
            normalized_text=normalized,
            detected_language=detected_language,
            is_code_switched=is_code_switched,
            char_count=char_count,
            word_count=word_count
        )

    def _detect_language(self, text: str) -> tuple[str, bool]:
        """Detects whether text is Telugu script, Telugu-English code-switched, or English."""
        # Check for native Telugu Unicode script
        has_telugu_script = bool(TELUGU_SCRIPT_REGEX.search(text))
        has_latin_script = bool(re.search(r"[a-zA-Z]", text))

        if has_telugu_script and has_latin_script:
            return "Telugu-English Mixed Script", True

        if has_telugu_script:
            return "Telugu", False

        # If only Latin script, check for Romanized Telugu code-switching
        lower_tokens = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
        matched_markers = lower_tokens.intersection(TELUGU_ROMANIZED_MARKERS)

        if matched_markers:
            return "Telugu-English (Code-Switched)", True

        # Default standard English if Latin
        if has_latin_script:
            return "English", False

        return "Unknown/Other", False


# Singleton instance
_default_text_processor: Optional[TextProcessor] = None

def get_text_processor() -> TextProcessor:
    """Returns singleton instance of TextProcessor."""
    global _default_text_processor
    if _default_text_processor is None:
        _default_text_processor = TextProcessor()
    return _default_text_processor
