"""Security Headers Middleware for Industrial Defect Intelligence FastAPI Backend.

Conforms to Phase 5 Step 5.7.2:
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- Referrer-Policy: strict-origin-when-cross-origin
- Permissions-Policy: geolocation=(), camera=(), microphone=()
- Content-Security-Policy: Tailored policy separating API endpoints and interactive documentation
- Strict-Transport-Security: Enforced on HTTPS or explicitly configured production environments
"""

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from api.config import get_api_settings


# Maximum protection for REST API endpoints (disables scripts, styles, frames, and objects)
API_CSP = (
    "default-src 'none'; "
    "frame-ancestors 'none'; "
    "base-uri 'none'; "
    "form-action 'none'"
)

# Tailored policy for Swagger UI (/docs) and ReDoc (/redoc)
# Allows required CDN assets and inline initialization scripts while denying external frames and plugins
DOCS_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
    "img-src 'self' data: https://fastapi.tiangolo.com; "
    "font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net data:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "object-src 'none'; "
    "base-uri 'self'"
)


def apply_security_headers(response: Response, request: Request) -> Response:
    """Applies defensive HTTP security headers to the given response."""
    # 1. Prevent MIME-type sniffing
    response.headers.setdefault("X-Content-Type-Options", "nosniff")

    # 2. Defend against clickjacking
    response.headers.setdefault("X-Frame-Options", "DENY")

    # 3. Referrer privacy protection
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")

    # 4. Restrict sensitive browser and device hardware APIs
    response.headers.setdefault(
        "Permissions-Policy",
        "geolocation=(), camera=(), microphone=()"
    )

    # 5. Content Security Policy (API vs. Interactive Documentation)
    path = request.url.path if request else ""
    if path.startswith("/docs") or path.startswith("/redoc") or path == "/openapi.json":
        response.headers.setdefault("Content-Security-Policy", DOCS_CSP)
    else:
        response.headers.setdefault("Content-Security-Policy", API_CSP)

    # 6. HTTP Strict Transport Security (HSTS)
    # Applied when request is HTTPS (direct or forwarded) or when environment is production
    if request:
        is_https = (
            request.url.scheme == "https"
            or request.headers.get("x-forwarded-proto", "").lower() == "https"
        )
        settings = get_api_settings()
        is_prod = getattr(settings, "environment", "").lower() == "production"

        if is_https or is_prod:
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains"
            )

    return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware attaching defensive HTTP security headers to all application responses."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        return apply_security_headers(response, request)
