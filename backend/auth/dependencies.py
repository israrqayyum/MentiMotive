"""
FastAPI dependency: validates the Bearer JWT and injects the current user.

Usage in any route:
    from auth.dependencies import get_current_user
    from db.models import User

    @router.get("/protected")
    async def protected(current_user: User = Depends(get_current_user)):
        ...
"""
import logging
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from auth.jwt_handler import decode_access_token
from db.database import get_db
from db.models import User

logger = logging.getLogger(__name__)

# Uses the standard "Bearer <token>" Authorization header scheme
bearer_scheme = HTTPBearer(auto_error=False)

_CREDENTIALS_EXCEPTION = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or expired token. Please log in again.",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Dependency that:
    1. Extracts the Bearer token from the Authorization header.
    2. Decodes and validates the JWT.
    3. Loads the corresponding User from the database.

    Raises 401 if the token is missing, malformed, expired, or the user no longer exists.
    """
    if credentials is None:
        raise _CREDENTIALS_EXCEPTION

    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise _CREDENTIALS_EXCEPTION

    user_id: str = payload.get("sub")
    if not user_id:
        raise _CREDENTIALS_EXCEPTION

    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    user = result.scalars().first()

    if user is None:
        logger.warning(f"Token valid but user not found: {user_id}")
        raise _CREDENTIALS_EXCEPTION

    return user
