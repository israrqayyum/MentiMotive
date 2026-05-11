import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from db.database import Base

def gen_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="user", nullable=False)
    
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid, index=True)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String, default="New Chat", index=True)
    
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    
    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan", order_by="Message.timestamp")


class Message(Base):
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid, index=True)
    session_id = Column(UUID(as_uuid=False), ForeignKey("sessions.id"), nullable=False, index=True)
    
    role = Column(String, nullable=False)  # 'user' or 'system' (bot)
    content = Column(String, nullable=False) # The actual content
    timestamp = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    source = Column(String, default="text") # 'text', 'voice', etc.
    
    # Triple-Emotion Model fields
    text_emotion = Column(String, nullable=True)
    voice_emotion = Column(String, nullable=True)
    face_emotion = Column(String, nullable=True)
    
    # Allows for optional structural or RAG references, context, etc.
    metadata_json = Column(JSON, nullable=True, default={})
    
    session = relationship("Session", back_populates="messages")