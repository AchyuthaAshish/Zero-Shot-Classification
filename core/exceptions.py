"""Domain exception types for the Zero-Shot Industrial Defect Classification system.

Conforms to SRS Section 4 (FR-014) and SRS Section 20.
"""

class DefectClassificationException(Exception):
    """Base exception for all defect classification errors."""
    def __init__(self, message: str, code: str = "SYSTEM_ERROR"):
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class InputError(DefectClassificationException):
    """Raised when user defect input is empty, invalid, or violates length constraints."""
    def __init__(self, message: str = "Invalid or empty defect description input."):
        super().__init__(message, code="INPUT_ERROR")


class TaxonomyError(DefectClassificationException):
    """Raised when the centralized taxonomy cannot be loaded or is corrupted."""
    def __init__(self, message: str = "Taxonomy definition error or unable to load taxonomy."):
        super().__init__(message, code="TAXONOMY_ERROR")


class ModelError(DefectClassificationException):
    """Raised when external LLM call fails, times out, rate limits, or returns malformed response."""
    def __init__(self, message: str = "External LLM service failed or returned an invalid response."):
        super().__init__(message, code="MODEL_ERROR")


class ValidationError(DefectClassificationException):
    """Raised when classification response fails strict schema or taxonomy validation."""
    def __init__(self, message: str = "Classification output failed validation against approved taxonomy."):
        super().__init__(message, code="VALIDATION_ERROR")


class VerificationError(DefectClassificationException):
    """Raised when the optional secondary verification pipeline encounters a failure."""
    def __init__(self, message: str = "Verification service encountered an error."):
        super().__init__(message, code="VERIFICATION_ERROR")


class PersistenceError(DefectClassificationException):
    """Raised when saving or querying classification records fails."""
    def __init__(self, message: str = "Persistence storage failure."):
        super().__init__(message, code="PERSISTENCE_ERROR")


class ConfigurationError(DefectClassificationException):
    """Raised when required configuration settings or API credentials are missing."""
    def __init__(self, message: str = "Configuration error or missing environment setting."):
        super().__init__(message, code="CONFIGURATION_ERROR")
