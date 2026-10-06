"""Supabase Authentication Service and JWT Verification for FastAPI Backend.

Conforms to Phase 5 Step 5.1:
- Extracts and validates Bearer tokens from incoming HTTP Authorization headers
- Supports both offline cryptographic JWT verification (via SUPABASE_JWT_SECRET)
  and online Supabase Auth verification (via client.auth.get_user)
- Standardized HTTP 401 UnauthorizedError mapping adhering to Phase 3.5 envelope
- Absolute security: raw tokens and cryptographic secrets are never logged, printed, or disclosed
"""

import logging
from typing import Optional, Dict, Any
import jwt

from config.settings import get_settings
from database.supabase_client import is_supabase_configured, get_supabase_client, _sanitize_error
from api.errors import UnauthorizedError
from api.schemas.auth import AuthenticatedUser

logger = logging.getLogger("api.auth")


def extract_bearer_token(auth_header: Optional[str]) -> str:
    """Extracts and validates Bearer token format from HTTP Authorization header.
    
    Raises UnauthorizedError if header is missing, malformed, or empty.
    Never logs or leaks the token content.
    """
    if not auth_header or not auth_header.strip():
        raise UnauthorizedError(
            message="Authentication credentials were not provided. Missing Authorization header.",
            details={"required_header": "Authorization: Bearer <token>"}
        )

    parts = auth_header.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise UnauthorizedError(
            message="Invalid Authorization header format. Expected 'Bearer <token>'.",
            details={"expected_format": "Bearer <token>"}
        )

    token = parts[1].strip()
    if not token:
        raise UnauthorizedError(
            message="Bearer token is missing or empty.",
            details={"expected_format": "Bearer <token>"}
        )

    return token


def _resolve_user_role(user_id: str) -> str:
    """Resolves application role from the authoritative public.profiles record.

    Defaults to 'employee' if database is unconfigured or profile record is not yet present.
    Never trusts client-provided claims for role escalation.
    """
    if not is_supabase_configured():
        return "employee"
    try:
        from database.supabase_client import get_profile
        profile = get_profile(user_id)
        if profile and profile.get("role"):
            return str(profile.get("role"))
    except Exception as e:
        logger.debug("Could not resolve role from profile for user %s: %s", user_id, _sanitize_error(str(e)))
    return "employee"


def verify_supabase_jwt(token: str) -> AuthenticatedUser:
    """Verifies a Supabase Auth access token and returns an AuthenticatedUser representation.
    
    Strategy:
    1. If SUPABASE_JWT_SECRET is configured, performs fast offline cryptographic verification via PyJWT.
    2. Otherwise, delegates verification to Supabase Auth API via client.auth.get_user(jwt=token).
    
    Raises UnauthorizedError on missing/invalid/expired tokens.
    Never exposes token contents, internal secrets, or raw exceptions.
    """
    if not token or not str(token).strip():
        raise UnauthorizedError(
            message="Bearer token is missing or empty.",
            details={"expected_format": "Bearer <token>"}
        )

    settings = get_settings()
    jwt_secret = getattr(settings, "supabase_jwt_secret", None)

    # Strategy 1: Offline cryptographic verification with JWT secret if available
    if jwt_secret and jwt_secret.strip():
        try:
            # Supabase Auth tokens are typically HS256 signed JWTs with sub (user UUID)
            payload = jwt.decode(
                token,
                jwt_secret.strip(),
                algorithms=["HS256"],
                options={"verify_aud": False}
            )
            user_id = payload.get("sub")
            if not user_id:
                raise UnauthorizedError("Invalid authentication token: missing subject identity.")

            email = payload.get("email")
            metadata = payload.get("user_metadata") or {}
            role = _resolve_user_role(str(user_id))
            return AuthenticatedUser(
                user_id=str(user_id),
                email=email,
                role=role,
                metadata=metadata if isinstance(metadata, dict) else {}
            )
        except jwt.ExpiredSignatureError:
            raise UnauthorizedError("Authentication token has expired.")
        except jwt.InvalidTokenError:
            raise UnauthorizedError("Invalid authentication token.")
        except UnauthorizedError:
            raise
        except Exception as e:
            logger.warning("JWT verification error: %s", _sanitize_error(str(e)))
            raise UnauthorizedError("Invalid authentication token.")

    # Strategy 2: Online Supabase Auth server verification
    if not is_supabase_configured():
        raise UnauthorizedError("Authentication service is currently unconfigured.")

    try:
        sb_client = get_supabase_client()
        user_response = sb_client.auth.get_user(jwt=token)
        if user_response and getattr(user_response, "user", None):
            sb_user = user_response.user
            user_id = str(getattr(sb_user, "id", None) or "")
            if not user_id:
                raise UnauthorizedError("Invalid authentication token: missing user identifier.")
            email = getattr(sb_user, "email", None)
            metadata = getattr(sb_user, "user_metadata", None) or {}
            role = _resolve_user_role(user_id)
            return AuthenticatedUser(
                user_id=user_id,
                email=email,
                role=role,
                metadata=metadata if isinstance(metadata, dict) else {}
            )
        raise UnauthorizedError("Invalid authentication token.")
    except UnauthorizedError:
        raise
    except Exception as exc:
        err_str = str(exc).lower()
        if "expired" in err_str:
            raise UnauthorizedError("Authentication token has expired.")
        elif any(k in err_str for k in ("invalid", "jwt", "token", "unauthorized", "bad", "malformed", "not found")):
            raise UnauthorizedError("Invalid authentication token.")
        else:
            logger.warning("Supabase Auth API error during token validation: %s", _sanitize_error(str(exc)))
            raise UnauthorizedError("Invalid or rejected authentication token.")
