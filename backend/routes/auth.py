"""
Authentication routes:
  POST /auth/login   → verify credentials, return JWT + user info
  POST /auth/logout  → stateless acknowledgement (client discards token)
"""
import logging
import bcrypt

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from db.models import User
from models.schemas import LoginRequest, LoginResponse, LogoutResponse
from services.user_service import get_user_by_email
from auth.jwt_handler import create_access_token
from auth.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ── Login ────────────────────────────────────────────────────────────────────

@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Login",
    description=(
        "Authenticate with **email + password**. "
        "On success returns a JWT `access_token` (Bearer) and the user's profile. "
        "Pass the token in the `Authorization: Bearer <token>` header on subsequent requests."
    ),
    responses={
        401: {"description": "Invalid email or password."},
    },
)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)) -> LoginResponse:
    # 1. Look up user
    user: User | None = await get_user_by_email(db, payload.email)

    # 2. Validate password — use bcrypt directly to avoid passlib/bcrypt>=4.x issue
    if user is None or not bcrypt.checkpw(
        payload.password.encode("utf-8"),
        user.password_hash.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3. Mint JWT  (sub = user_id as string)
    token = create_access_token({"sub": str(user.id)})

    logger.info(f"User logged in: {user.email}")
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        user_id=str(user.id),
        name=user.name,
        email=user.email,
        role=user.role,
    )


# ── Logout ───────────────────────────────────────────────────────────────────

@router.post(
    "/logout",
    response_model=LogoutResponse,
    summary="Logout",
    description=(
        "Stateless logout — the server does not maintain a token blacklist. "
        "The client is responsible for discarding the token after receiving this response."
    ),
)
async def logout(_: User = Depends(get_current_user)) -> LogoutResponse:
    """
    Requires a valid Bearer token (just to confirm the caller is authenticated).
    Returns a 200 OK with a confirmation message.
    """
    return LogoutResponse(message="Logged out successfully. Please discard your token.")
