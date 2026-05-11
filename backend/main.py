# ------------------------------------------------------------
# FastAPI backend for MentiMotive - Voice & Text Emotion Analysis

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Configure Logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

# Load environment variables from root directory .env file
root_env = Path(__file__).parent.parent / ".env"
if root_env.exists():
    load_dotenv(root_env)
    logging.info(f"✅ Loaded .env from: {root_env}")
    # Verify API key is loaded (hide actual value)
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        logging.info(f"✅ GEMINI_API_KEY loaded (length: {len(api_key)})")
    else:
        logging.error("❌ GEMINI_API_KEY not found in .env file")
else:
    logging.warning(f"⚠️ No .env file found at: {root_env}")

from routes import health, text_emotion, voice_emotion, parallel_sentiment, chat, rag, user, session, auth
from services.text_emotion_distilBERT import get_analyzer as get_text_analyzer
from services.voice_emotion_service import get_analyzer as get_voice_analyzer
from services.parallel_sentiment_service import get_analyzer as get_parallel_analyzer
from services.onnx_embeddings_service import get_onnx_embeddings
from services.langchain_service import langchain_service
from services.rag_service import rag_service
from db.database import init_db

# 2. Lifespan Event Handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML models during server startup and init database."""
    logging.info("🔄 Preloading ML models and initializing DB...")
    
    try:
        # Initialize the database schemas
        await init_db()
        logging.info("✅ Database schema initialized.")
    except Exception as e:
        logging.error(f"❌ Failed to initialize database: {e}")

    try:
        # Load text emotion model
        get_text_analyzer()
        logging.info("✅ Text emotion model loaded")
        
        # Load voice emotion models (STT + Text + Audio)
        get_voice_analyzer()
        logging.info("✅ Voice emotion models loaded")
        
        # Load parallel sentiment models (STT + Text + Audio)
        get_parallel_analyzer()
        logging.info("✅ Parallel sentiment models loaded")
        
        # Load ONNX embeddings model
        get_onnx_embeddings()
        logging.info("✅ ONNX embeddings model loaded")
        
        # Load LangChain service
        langchain_service.initialize()
        logging.info("✅ LangChain service loaded")
        
        # Load RAG service
        rag_service.initialize()
        logging.info("✅ RAG service loaded")
        
        logging.info("🎉 All models and services loaded successfully!")
    except Exception as e:
        logging.error(f"❌ Error loading models: {e}")
        raise
    
    yield
    
    # Cleanup (if needed)
    logging.info("🔄 Shutting down...")

# 3. Initialize FastAPI App
app = FastAPI(
    title="MentiMotive API",
    version="2.0",
    description="""
MentiMotive is a **voice and text** emotion-aware assistant backend.

### What this API provides
- **Health**: liveness checks and a simple banner at `/`.
- **Text emotion**: `POST /classify/text` classifies a single string.
- **Voice analysis**: `POST /analyze/voice` returns transcript + text emotion + acoustic emotion.
- **Parallel sentiment**: `POST /analyze/parallel-sentiment` runs STT + audio emotion in parallel, then text emotion on the transcript, and returns timing stats.
- **Chat**: session-based `/chat/text` and `/chat/voice` using LangChain and retrieval; `/chat/voice` optionally includes TTS audio.
- **Users**: basic user creation and session fetching.
- **Sessions**: robust session storage powered by PostgreSQL.
- **RAG ingestion & stats** (under the `/chat/...` prefix): `POST /chat/ingest`, `GET /chat/collection/stats`.

### Conventions (response field names)
- `POST /classify/text` → `emotion` + `confidence_score` (top label, lowercased).
- `POST /analyze/voice` and `POST /analyze/parallel-sentiment` → nested results with `label` + `confidence`.
- `POST /chat/text` and `POST /chat/voice` → per-modality `emotion` + `confidence` inside the `emotions` object.

Swagger UI: `/docs` • ReDoc: `/redoc`
""",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "Health",
            "description": "Process liveness and a simple service banner. `/health` does not validate ML subsystems.",
        },
        {
            "name": "Users",
            "description": "Manage user accounts.",
        },
        {
            "name": "Sessions",
            "description": "Manage session history and details.",
        },
        {
            "name": "Text Emotion",
            "description": "Classify plain text into an emotion label with a confidence score.",
        },
        {
            "name": "Voice Analysis",
            "description": "Upload audio; get transcript and both text-based and acoustic emotion outputs (label/confidence).",
        },
        {
            "name": "Parallel Sentiment",
            "description": "Upload audio; STT and audio emotion run in parallel, then text emotion runs on the transcript. Includes timing stats.",
        },
        {
            "name": "Chat",
            "description": "Session-based text/voice chat. Voice chat can synthesize audio (TTS) and returns emotions per modality.",
        },
        {
            "name": "RAG",
            "description": "Ingest PDF/TXT into the vector store and inspect collection stats. Routes are under `/chat/...`.",
        },
    ],
)

# 3. Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Include Routers
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(user.router)
app.include_router(session.router)
app.include_router(text_emotion.router)
app.include_router(voice_emotion.router)
app.include_router(parallel_sentiment.router)
app.include_router(chat.router)
app.include_router(rag.router)

# 6. Run the server
if __name__ == "__main__":
    import uvicorn
    logging.info("🚀 Starting MentiMotive API server...")
    # Use reload_dirs to enable auto-reload
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True, 
        reload_dirs=["backend"]
    )