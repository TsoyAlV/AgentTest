import secrets

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

bearer = HTTPBearer(auto_error=False)


def current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> int:
    settings = request.app.state.settings
    if not settings.dev_auth_enabled:
        raise HTTPException(503, "Authentication is not configured")
    expected = settings.dev_api_token.get_secret_value()
    if credentials is None or not secrets.compare_digest(
        credentials.credentials.encode(), expected.encode()
    ):
        raise HTTPException(401, "Invalid bearer token", headers={"WWW-Authenticate": "Bearer"})
    # Never accept a user ID or roles from the request body or LLM.
    return settings.dev_user_id


def dev_admin(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    cfg = request.app.state.settings
    token = cfg.dev_admin_token.get_secret_value()
    if cfg.app_env != "local" or not cfg.dev_auth_enabled or len(token) < 24 or token.startswith('CHANGE_ME'):
        raise HTTPException(503, "Set DEV_ADMIN_TOKEN to enable local user experiments")
    if credentials is None or not secrets.compare_digest(credentials.credentials.encode(), token.encode()):
        raise HTTPException(401, "Invalid admin bearer token", headers={"WWW-Authenticate": "Bearer"})
