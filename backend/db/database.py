import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

from config import settings

# In case someone runs scripts from root or other dirs, make sure DB is accessible or fallback gracefully.
DATABASE_URL = settings.DATABASE_URL

# Create async engine for PostgreSQL
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # set to true for debugging SQL queries
    future=True
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

# Base model for declarative mapping
Base = declarative_base()

async def init_db():
    """
    Initialize database, creates all tables.
    To be called in the lifespan function of FastAPI.
    """
    async with engine.begin() as conn:
        # Create all tables if they don't exist
        # WARNING: In production, rely on Alembic migrations instead of create_all()
        await conn.run_sync(Base.metadata.create_all)
        
    # Seed default user for backward compatibility when user_id is missing
    from db.models import User
    from sqlalchemy.future import select
    
    DEFAULT_USER_ID = "11111111-1111-1111-1111-111111111111"
    async with AsyncSessionLocal() as session:
        stmt = select(User).where(User.id == DEFAULT_USER_ID)
        result = await session.execute(stmt)
        user = result.scalars().first()
        if not user:
            import bcrypt
            import logging
            # Use bcrypt directly to avoid passlib/bcrypt>=4.x compatibility issue
            hashed = bcrypt.hashpw(b"default123", bcrypt.gensalt()).decode("utf-8")
            default_user = User(
                id=DEFAULT_USER_ID,
                name="Default User",
                email="default@mentimotive.local",
                password_hash=hashed,
                role="admin"
            )
            session.add(default_user)
            await session.commit()
            logging.info("✅ Default user seeded successfully")
        else:
            import logging
            logging.info("✅ Default user already exists")


async def get_db():
    """
    Dependency to yield an async database session piece by piece.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except:
            await session.rollback()
            raise
        finally:
            await session.close()
