from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import bcrypt
import logging
from db.models import User
from models.schemas import UserCreate

logger = logging.getLogger(__name__)

def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

async def create_user(db: AsyncSession, user: UserCreate) -> User:
    """Create a new user."""
    # Check if user already exists
    stmt = select(User).where(User.email == user.email)
    result = await db.execute(stmt)
    existing_user = result.scalars().first()
    
    if existing_user:
        logger.warning(f"Attempt to create duplicate user with email: {user.email}")
        return None
        
    hashed_password = get_password_hash(user.password)
    db_user = User(
        name=user.name,
        email=user.email,
        password_hash=hashed_password,
        role=user.role
    )
    
    try:
        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)
        logger.info(f"User created: {db_user.email} with role: {db_user.role}")
        return db_user
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating user: {e}")
        raise

async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    stmt = select(User).where(User.email == email)
    result = await db.execute(stmt)
    return result.scalars().first()

async def get_all_users(db: AsyncSession):
    stmt = select(User).order_by(User.created_at.desc())
    result = await db.execute(stmt)
    return result.scalars().all()
