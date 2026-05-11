from typing import Dict, Optional, List
from datetime import datetime
import logging
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, delete

from db.models import Session, Message, User
from config import settings

logger = logging.getLogger(__name__)

class SessionManagerDB:
    """Manages chat sessions stored in PostgreSQL."""
    
    async def get_or_create_session(self, db: AsyncSession, user_id: str, session_id: Optional[str] = None) -> Session:
        """Get existing session or create new one."""
        if session_id:
            stmt = select(Session).where(Session.id == session_id, Session.user_id == user_id)
            result = await db.execute(stmt)
            session = result.scalars().first()
            if session:
                logger.info(f"Retrieved existing session: {session_id}")
                return session

        # Generate new session_id if not provided or not found
        new_session_id = session_id or str(uuid.uuid4())
        logger.info(f"Creating new session: {new_session_id}")
        
        session = Session(id=new_session_id, user_id=user_id)
        db.add(session)
        await db.commit()
        await db.refresh(session)
        return session
    
    async def get_session(self, db: AsyncSession, session_id: str, user_id: str) -> Session:
        """Get existing session (raises error if not found)."""
        stmt = select(Session).where(Session.id == session_id, Session.user_id == user_id)
        result = await db.execute(stmt)
        session = result.scalars().first()
        if not session:
            raise ValueError(f"Session {session_id} not found")
        return session
    
    async def add_message(
        self, 
        db: AsyncSession, 
        session_id: str, 
        user_id: str, 
        role: str, 
        content: str, 
        source: str = "text",
        text_emotion: Optional[str] = None,
        voice_emotion: Optional[str] = None,
        face_emotion: Optional[str] = None
    ) -> Message:
        """Add message to session history with multi-modal emotion states."""
        # Ensure session exists
        session = await self.get_or_create_session(db, user_id, session_id)
        
        new_message = Message(
            session_id=session.id,
            role=role,
            content=content,
            source=source,
            text_emotion=text_emotion,
            voice_emotion=voice_emotion,
            face_emotion=face_emotion
        )
        
        # Update session activity timestamp
        session.updated_at = datetime.utcnow()
        
        db.add(new_message)
        await db.commit()
        await db.refresh(new_message)
        
        return new_message
    
    async def get_messages(self, db: AsyncSession, session_id: str, user_id: str) -> List[Message]:
        """Get all messages from session."""
        session = await self.get_session(db, session_id, user_id)
        # Fetching messages securely via the db
        stmt = select(Message).where(Message.session_id == session.id).order_by(Message.timestamp)
        result = await db.execute(stmt)
        return list(result.scalars().all())
    
    async def get_langchain_history(self, db: AsyncSession, session_id: str, user_id: str) -> List[Dict]:
        """Get messages in LangChain format."""
        messages = await self.get_messages(db, session_id, user_id)
        # Assuming limiting to MAX_MESSAGES for LLM Context explicitly here
        max_messages = settings.MAX_MESSAGES
        recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
        
        return [
            {"role": msg.role, "content": msg.content}
            for msg in recent_messages
        ]
    
    async def clear_session(self, db: AsyncSession, session_id: str, user_id: str):
        """Clear a single session completely."""
        session = await self.get_session(db, session_id, user_id)
        await db.delete(session)
        await db.commit()
        logger.info(f"Cleared session: {session_id}")

# Global instance for easy imports where dependecy injection is bypassed (prefer Depends(get_db) though)
session_manager = SessionManagerDB()
