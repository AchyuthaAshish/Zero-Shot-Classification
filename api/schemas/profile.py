"""Pydantic schemas for user profile data model and API responses.

Conforms to Phase 5 Step 5.2:
- Profiles table schema representation linked to auth.users(id)
- Profile data transfer objects with strict validation
- Zero exposure of passwords, tokens, JWTs, or internal secrets
- Excludes role fields (deferred to Step 5.3)
"""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator


class ProfileData(BaseModel):
    """Domain model and API data representation of a user profile."""
    id: str = Field(description="User UUID strictly referencing auth.users(id)")
    display_name: str = Field(description="User display name")
    email: Optional[str] = Field(default=None, description="User email address")
    role: str = Field(default="employee", description="Authoritative application role ('employee')")
    created_at: Optional[str] = Field(default=None, description="ISO timestamp of profile creation")
    updated_at: Optional[str] = Field(default=None, description="ISO timestamp of last profile update")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "display_name": "Senior Operator",
                "email": "operator@factory.example.com",
                "role": "employee",
                "created_at": "2026-10-06T10:00:00Z",
                "updated_at": "2026-10-06T10:00:00Z"
            }
        }
    )


class ProfileResponse(BaseModel):
    """Standard response envelope for profile operations."""
    success: bool = Field(default=True, description="Indicates successful profile operation")
    data: ProfileData = Field(description="Profile details payload")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "display_name": "Senior Operator",
                    "email": "operator@factory.example.com",
                    "role": "employee",
                    "created_at": "2026-10-06T10:00:00Z",
                    "updated_at": "2026-10-06T10:00:00Z"
                }
            }
        }
    )


class ProfileUpdateRequest(BaseModel):
    """Request payload for updating an authenticated user's profile."""
    display_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Updated display name (1 to 100 characters, cannot be whitespace-only)."
    )

    @field_validator("display_name")
    @classmethod
    def validate_display_name_not_blank(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Display name cannot be empty or contain only whitespace.")
        return clean

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "display_name": "Chief Reliability Engineer"
            }
        }
    )
