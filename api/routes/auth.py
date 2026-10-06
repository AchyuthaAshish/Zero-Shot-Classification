"""Authentication Router for FastAPI Backend.

Provides:
- GET /api/v1/auth/me: Returns verified Supabase Auth user identity representation.
  Protected by mandatory get_current_user dependency.
- Rejects missing, malformed, invalid, or expired tokens with standardized HTTP 401 UnauthorizedError.
- Never discloses JWT tokens, passwords, database credentials, or internal exceptions.
"""

import logging
from fastapi import APIRouter, Depends, status

from api.dependencies import get_current_user
from api.schemas.auth import AuthenticatedUser, UserResponse, UserResponseData
from api.schemas.profile import ProfileData, ProfileResponse, ProfileUpdateRequest
from api.schemas.errors import ErrorResponse

logger = logging.getLogger("api.routes.auth")

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.get(
    "/me",
    summary="Get Authenticated User Identity",
    description=(
        "Validates the incoming Supabase Bearer access token and returns the "
        "verified user identity representation. Rejects unauthenticated or invalid requests with HTTP 401."
    ),
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "model": UserResponse,
            "description": "Successfully authenticated user identity.",
        },
        401: {
            "model": ErrorResponse,
            "description": "Authentication failure (missing, malformed, invalid, or expired token).",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal server error.",
        }
    }
)
def get_my_user(
    current_user: AuthenticatedUser = Depends(get_current_user)
) -> UserResponse:
    """Returns the authenticated user details for the verified Supabase Auth session."""
    return UserResponse(
        success=True,
        data=UserResponseData(
            user_id=current_user.user_id,
            email=current_user.email,
            role=current_user.role,
            authenticated=True
        )
    )


@router.get(
    "/profile",
    summary="Get Authenticated User Profile",
    description=(
        "Retrieves the application profile associated with the authenticated user ID. "
        "Rejects unauthenticated requests with HTTP 401."
    ),
    response_model=ProfileResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "model": ProfileResponse,
            "description": "User profile successfully retrieved.",
        },
        401: {
            "model": ErrorResponse,
            "description": "Authentication failure (missing, malformed, invalid, or expired token).",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal server error.",
        }
    }
)
def get_my_profile(
    current_user: AuthenticatedUser = Depends(get_current_user)
) -> ProfileResponse:
    """Returns the authenticated user's profile record from public.profiles."""
    import persistence.repository as repository

    profile_record = None
    if repository.is_supabase_configured():
        try:
            profile_record = repository.get_profile(user_id=current_user.user_id)
        except Exception as e:
            logger.warning(f"Profile query failed, using session fallback: {e}")
            profile_record = None

    if profile_record:
        return ProfileResponse(
            success=True,
            data=ProfileData(
                id=str(profile_record.get("id", current_user.user_id)),
                display_name=str(profile_record.get("display_name") or "Operator"),
                email=profile_record.get("email") or current_user.email,
                role=str(profile_record.get("role") or current_user.role),
                created_at=str(profile_record.get("created_at")) if profile_record.get("created_at") else None,
                updated_at=str(profile_record.get("updated_at")) if profile_record.get("updated_at") else None
            )
        )

    # Fallback when profile record not yet created in database
    default_name = (
        current_user.metadata.get("full_name")
        or current_user.metadata.get("name")
        or (current_user.email.split("@")[0] if current_user.email and "@" in current_user.email else "Operator")
    )
    return ProfileResponse(
        success=True,
        data=ProfileData(
            id=current_user.user_id,
            display_name=default_name,
            email=current_user.email,
            role=current_user.role,
            created_at=None,
            updated_at=None
        )
    )


@router.patch(
    "/profile",
    summary="Update Authenticated User Profile",
    description="Updates the authenticated user's profile display name in public.profiles.",
    response_model=ProfileResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "model": ProfileResponse,
            "description": "User profile successfully updated.",
        },
        401: {
            "model": ErrorResponse,
            "description": "Authentication failure.",
        },
        422: {
            "model": ErrorResponse,
            "description": "Validation error on update payload.",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal server error.",
        }
    }
)
def update_my_profile(
    body: ProfileUpdateRequest,
    current_user: AuthenticatedUser = Depends(get_current_user)
) -> ProfileResponse:
    """Updates profile attributes for the authenticated user session."""
    import persistence.repository as repository

    if repository.is_supabase_configured():
        updated = repository.update_profile(
            user_id=current_user.user_id,
            display_name=body.display_name
        )
        if updated:
            return ProfileResponse(
                success=True,
                data=ProfileData(
                    id=str(updated.get("id", current_user.user_id)),
                    display_name=str(updated.get("display_name", body.display_name)),
                    email=updated.get("email") or current_user.email,
                    role=str(updated.get("role") or current_user.role),
                    created_at=str(updated.get("created_at")) if updated.get("created_at") else None,
                    updated_at=str(updated.get("updated_at")) if updated.get("updated_at") else None
                )
            )

    return ProfileResponse(
        success=True,
        data=ProfileData(
            id=current_user.user_id,
            display_name=body.display_name,
            email=current_user.email,
            role=current_user.role,
            created_at=None,
            updated_at=None
        )
    )

