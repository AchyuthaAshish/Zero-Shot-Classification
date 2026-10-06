from typing import Generator, Optional
from fastapi import Request, Depends
from api.config import APISettings, get_api_settings
from api.schemas.auth import AuthenticatedUser
from api.auth import extract_bearer_token, verify_supabase_jwt
from api.errors import UnauthorizedError, ForbiddenError


def get_api_config() -> APISettings:
    """Dependency that provides the current API settings."""
    return get_api_settings()


def get_classifier_service():
    """Dependency that provides the reusable DefectClassifier instance."""
    from classification.classifier import get_defect_classifier
    return get_defect_classifier()


def get_current_user(request: Request) -> AuthenticatedUser:
    """Dependency enforcing mandatory Supabase authentication.
    
    Extracts and validates the Bearer token from the incoming request.
    Raises UnauthorizedError (HTTP 401) on missing, malformed, or invalid tokens.
    """
    auth_header = request.headers.get("Authorization")
    token = extract_bearer_token(auth_header)
    return verify_supabase_jwt(token)


def get_optional_user(request: Request) -> Optional[AuthenticatedUser]:
    """Dependency providing optional authentication context.
    
    If Authorization header is present, validates the token (raising UnauthorizedError on invalid/expired).
    If Authorization header is absent, safely returns None without blocking the request.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.strip():
        return None
    token = extract_bearer_token(auth_header)
    return verify_supabase_jwt(token)


def _require_role(required_role: str, current_user: AuthenticatedUser) -> AuthenticatedUser:
    """Internal helper validating application role against authoritative profile.

    Extensible foundation for role-based authorization (e.g. employee, reviewer, admin).
    """
    if not current_user:
        raise UnauthorizedError("Authentication credentials were not provided.")

    import persistence.repository as repository
    if repository.is_supabase_configured():
        try:
            profile = repository.get_profile(user_id=current_user.user_id)
        except Exception:
            profile = None

        if profile is None:
            raise ForbiddenError(
                message="User application profile not found. Access forbidden.",
                details={"reason": "profile_not_found"}
            )

        role = profile.get("role") or "employee"
        if role != required_role:
            raise ForbiddenError(
                message=f"Access forbidden. {required_role.capitalize()} role required.",
                details={"required_role": required_role, "actual_role": role}
            )
        current_user.role = role
    else:
        if getattr(current_user, "role", None) != required_role:
            raise ForbiddenError(
                message=f"Access forbidden. {required_role.capitalize()} role required.",
                details={"required_role": required_role, "actual_role": getattr(current_user, "role", None)}
            )

    return current_user


def require_employee(
    current_user: AuthenticatedUser = Depends(get_current_user)
) -> AuthenticatedUser:
    """Dependency enforcing that the authenticated user possesses the 'employee' role.

    Verifies:
    1. Request has a valid authenticated user (HTTP 401 via get_current_user if invalid/missing)
    2. User's application profile exists in public.profiles (HTTP 403 if profile missing)
    3. User's authoritative role is 'employee' (HTTP 403 if role mismatch)
    """
    return _require_role("employee", current_user)



