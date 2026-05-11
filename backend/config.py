from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    """
    Application configuration settings.
    These settings are automatically populated from your local environment 
    variables or a `.env` file via `pydantic-settings`.
    """
    
    # ==========================================
    # LLM General Settings
    # ==========================================
    LLM_PROVIDER: str = Field(
        default="openai", 
        description="Which LLM provider to use. Valid options: 'gemini' or 'openai'"
    )
    LLM_TEMPERATURE: float = Field(
        default=0.7, 
        description="Controls prediction randomness. 0.0 is deterministic, 1.0 is highly creative."
    )
    
    # ==========================================
    # Provider-Specific API Keys & Models
    # ==========================================
    # Gemini
    GEMINI_API_KEY: str = Field(default="", description="Your Google Gemini API Key")
    GEMINI_MODEL: str = Field(default="gemini-2.5-flash", description="Gemini model version (e.g., gemini-2.5-flash, gemini-1.5-pro)")
    
    # OpenAI
    OPENAI_API_KEY: str = Field(default="", description="Your OpenAI API Key")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini", description="OpenAI model version (e.g., gpt-4o-mini, gpt-4o, gpt-3.5-turbo)")
    
    # ==========================================
    # Audio & Text-to-Speech (TTS)
    # ==========================================
    TTS_ENGINE: str = Field(
        default="edge", 
        description="TTS Engine. 'edge' for cloud-based Microsoft Edge TTS, 'tiny' for offline local TinyTTS."
    )
    
    # ==========================================
    # Session & Memory
    # ==========================================
    MAX_MESSAGES: int = Field(
        default=50, 
        description="Maximum number of conversational messages to keep in memory per active chat session."
    )
    
    # ==========================================
    # Retrieval-Augmented Generation (RAG)
    # ==========================================
    RAG_TOP_K: int = Field(
        default=3, 
        description="Number of most relevant document chunks to fetch during semantic context retrieval."
    )
    CHUNK_SIZE: int = Field(
        default=500, 
        description="Character count for splitting large uploaded documents into searchable parts."
    )
    CHUNK_OVERLAP: int = Field(
        default=50, 
        description="Number of characters to overlap between document chunks to preserve flowing context."
    )
    
    # ==========================================
    # Vector Embeddings & Database
    # ==========================================
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", 
        description="HuggingFace model ID for generating text embeddings."
    )
    ONNX_EMBEDDING_MODEL_PATH: str = Field(
        default="/onnx_models/all-minilm-l6-v2", 
        description="Local path pointing to the offline, quantized ONNX embedding model."
    )
    
    CHROMA_PERSIST_DIR: str = Field(
        default="./backend/chroma_db", 
        description="Local root directory where ChromaDB persistently stores vector data."
    )
    COLLECTION_NAME: str = Field(
        default="mental_health_docs", 
        description="Internal name of the ChromaDB collection holding the system's documents."
    )
    
    # ==========================================
    # Database (PostgreSQL)
    # ==========================================
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/mentimdb",
        description="Async PostgreSQL connection string."
    )

    # ==========================================
    # Authentication (JWT)
    # ==========================================
    SECRET_KEY: str = Field(
        default="change-me-in-production-use-a-long-random-string",
        description="Secret key for signing JWT tokens. Override in .env with a secure random value."
    )
    ALGORITHM: str = Field(
        default="HS256",
        description="JWT signing algorithm."
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=60 * 24 * 7,   # 7 days
        description="JWT access token TTL in minutes."
    )

    # Configuration for reading the .env file automatically
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
