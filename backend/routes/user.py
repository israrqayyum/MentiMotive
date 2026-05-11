from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import get_db
from models.schemas import UserCreate, UserResponse, SessionListResponse
from services.user_service import create_user, get_all_users
from db.models import Session, User
from sqlalchemy.future import select
from typing import List

from auth.dependencies import get_current_user

router = APIRouter(prefix="/users", tags=["Users"])

@router.post("", response_model=UserResponse, summary="Create a new user")
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    db_user = await create_user(db, user)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    return db_user

@router.get("/me", response_model=UserResponse, summary="Get current logged in user")
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user

@router.get("", response_model=List[UserResponse], summary="Get all users")
async def get_users(db: AsyncSession = Depends(get_db)):
    users = await get_all_users(db)
    return users

@router.get("/{user_id}/sessions", response_model=List[SessionListResponse], summary="Get all active sessions for a user")
async def get_user_sessions(user_id: str, db: AsyncSession = Depends(get_db)):
    # We query the DB ordered by updated_at descending
    stmt = select(Session).where(Session.user_id == user_id).order_by(Session.updated_at.desc())
    result = await db.execute(stmt)
    sessions = result.scalars().all()
    return sessions
