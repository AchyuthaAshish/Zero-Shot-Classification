"""Pydantic schemas for authentication and user identity.

Conforms to Phase 5 Step 5.1:
- Minimal, clean authenticated user representation (UUID, email, safe metadata)
- Standardized response envelope for user details
- Zero exposure of tokens, passwords, or internal security credentials
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class AuthenticatedUser(BaseModel):
    """Internal domain representation of a verified Supabase Auth user."""
    user_id: str = Field(description="Supabase Auth user UUID (sub claim)")
    email: Optional[str] = Field(default=None, description="Authenticated user email address if available")
    role: str = Field(default="employee", description="Authoritative application role ('employee')")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Safe user metadata dictionary")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "email": "operator@industrial.example.com",
                "role": "employee",
                "metadata": {}
            }
        }
    )


class UserResponseData(BaseModel):
    """Payload for authenticated user identity response."""
    user_id: str = Field(description="Supabase Auth user UUID")
    email: Optional[str] = Field(default=None, description="Authenticated user email address")
    role: str = Field(default="employee", description="Authoritative application role ('employee')")
    authenticated: bool = Field(default=True, description="Indicates verified authentication status")


class UserResponse(BaseModel):
    """Standard response envelope for user profile/identity endpoint."""
    success: bool = Field(default=True, description="Indicates successful user identity retrieval")
    data: UserResponseData = Field(description="User profile details")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "data": {
                    "user_id": "550e8400-e29b-41d4-a716-446655440000",
                    "email": "operator@industrial.example.com",
                    "role": "employee",
                    "authenticated": True
                }
            }
        }
    )
