# 🧠 MentiMotive - Backend System Architecture & Setup

This document provides a concise overview of the MentiMotive FastAPI backend, including configurations, AI models, processing pipelines, and APIs.

*(Note: The frontend is a lightweight Streamlit application primarily used for testing and interacting with these backend services).*

---

## ⚙️ Environment & Configuration

The system uses `pydantic-settings` to centralize configurations in `backend/config.py`. Environment variables are loaded from a `.env` file at the root of the project.

**`.env` Setup Example:**
```env
# LLM Provider Selection
LLM_PROVIDER=openai      # Choices: openai or gemini
OPENAI_MODEL=gpt-4o-mini
GEMINI_MODEL=gemini-2.5-flash

# API Keys (Provide based on selected LLM_PROVIDER)
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Text-to-Speech Engine Selection
TTS_ENGINE=edge          # Choices: edge (cloud) or tiny (local offline)

# RAG Configuration
RAG_TOP_K=3              # Number of documents to retrieve
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Chat limitations
MAX_MESSAGES=50          # Max messages per session

# PostgreSQL (Multi-Session Database)
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/mentimdb
```

---

## 🗄️ Database Architecture (PostgreSQL + Multi-Session Support)

The backend uses **asynchronous SQLAlchemy (`asyncpg`)** with PostgreSQL to persist all chat session data — replacing the previous in-memory dictionary store. The database is bootstrapped automatically at server startup via FastAPI's `lifespan` function.

### Schema: Triple-Emotion Model

```
┌──────────────────────────────────────────────────────────────┐
│  users                                                       │
│  ─────────────────────────────────────────────────────────   │
│  id           UUID        (PK)                               │
│  name         String                                         │
│  email        String      (unique, indexed)                  │
│  password_hash String     (bcrypt hashed)                    │
│  created_at   DateTime                                       │
│  updated_at   DateTime                                       │
└──────────────────────────────────┬───────────────────────────┘
                                   │ 1-to-Many
┌──────────────────────────────────▼───────────────────────────┐
│  sessions                                                    │
│  ─────────────────────────────────────────────────────────   │
│  id           UUID        (PK)                               │
│  user_id      UUID        (FK → users.id)                    │
│  title        String      (default: "New Chat")              │
│  created_at   DateTime                                       │
│  updated_at   DateTime                                       │
└──────────────────────────────────┬───────────────────────────┘
                                   │ 1-to-Many
┌──────────────────────────────────▼───────────────────────────┐
│  messages                                                    │
│  ─────────────────────────────────────────────────────────   │
│  id            UUID       (PK)                               │
│  session_id    UUID       (FK → sessions.id)                 │
│  role          String     ('user' or 'assistant')            │
│  content       String                                        │
│  source        String     ('text' or 'voice')               │
│  timestamp     DateTime                                      │
│  text_emotion  String     (nullable) ← Triple-Emotion Model  │
│  voice_emotion String     (nullable) ← Triple-Emotion Model  │
│  face_emotion  String     (nullable) ← Future video support  │
│  metadata_json JSON       (optional RAG sources, etc.)       │
└──────────────────────────────────────────────────────────────┘
```

> **Triple-Emotion Model:** Each `Message` row can store the emotional signal from up to three independent modalities (text, voice waveform, and face expression). `face_emotion` defaults to `None` today and is reserved for future video/face pipeline integration.

### Backward Compatibility

A **Default User** (`id = 11111111-1111-1111-1111-111111111111`) is seeded automatically at startup. All chat endpoints accept an **optional** `user_id`. If omitted, the default user is used — ensuring existing clients with no `user_id` do not break.

---

## 🤖 Models Used

The backend orchestrates multiple specialized AI models:

1. **Speech-To-Text (STT):** OpenAI Whisper (`base` or `tiny`, CPU-optimized) for transcribing audio.
2. **Voice Emotion:** Wav2Vec2 (`superb/wav2vec2-base-superb-er`) for analyzing acoustic emotional tone (e.g., angry, sad, happy, neutral).
3. **Text Emotion:** DistilBERT for classifying text-based emotional sentiment.
4. **Embeddings (RAG):** `all-MiniLM-L6-v2` (via ONNX) to convert document text into vector embeddings for similarity searches.
5. **LLM (Generation):** OpenAI (e.g., GPT-4o-mini) or Google Gemini (e.g., 2.5 Flash) via LangChain to generate empathetic responses.
6. **Text-to-Speech (TTS):** Microsoft Edge TTS or a local TinyTTS model to synthesize the AI's response into voice.

---

## 🔄 Pipeline Steps & Flow

### 1. Document Ingestion (RAG Pipeline)
- **Purpose:** Enhances the LLM's knowledge base with specific mental health documents.
- **Flow:** User uploads PDF/TXT → Text extracted → Split into chunks (500 chars) → Converted to embeddings (`all-MiniLM-L6-v2`) → Stored persistently in ChromaDB vector database.

### 2. Text Chat Pipeline
- **Purpose:** Handle user text input, analyze emotion, and return an empathetic AI response.
- **Flow:**
  1. Receive user text, optional `session_id`, and optional `user_id`.
  2. Resolve `user_id` (use default if omitted). Get or create a DB session via `SessionManagerDB`.
  3. Analyze **Text Emotion** using DistilBERT.
  4. Load LangChain history from **PostgreSQL** (`get_langchain_history`).
  5. Perform **RAG Retrieval** in ChromaDB to find relevant document context.
  6. Pass text, emotion context (with dissonance detection), and RAG context to **LangChain/LLM**.
  7. **Persist** user message + AI response to PostgreSQL with `text_emotion`, `voice_emotion=None`, `face_emotion=None`.
  8. Return response + sources + detected emotions.

### 3. Voice Chat Pipeline
- **Purpose:** Handle spoken user input, extract multiple emotion layers, and return an AI response.
- **Flow:**
  1. Receive audio file, optional `session_id`, and optional `user_id` via form data.
  2. Resolve `user_id`. Get or create a DB session via `SessionManagerDB`.
  3. **Transcribe** audio + concurrently analyze **Audio Emotion** (Wav2Vec2) via `ParallelSentimentAnalyzer`.
  4. Extract **Text Emotion** from the transcribed text via DistilBERT (part of the parallel pipeline).
  5. Load LangChain history from **PostgreSQL** (`get_langchain_history`).
  6. RAG Retrieval → LangChain/LLM response generation.
  7. **Persist** user message with `text_emotion` + `voice_emotion`; assistant message saved alongside.
  8. Synthesize AI response to audio via TTS (Edge or TinyTTS).
  9. Return transcript, dual emotions, AI response, TTS audio (base64), and RAG sources.

### 4. Emotional Dissonance Detection
When building the LLM prompt context, `build_emotion_context()` inspects all available modalities. If more than one modality is detected (e.g. text says "happy" but voice says "sad"), an explicit instruction is injected into the system prompt:

> *"If you detect 'emotional dissonance' (e.g. text says happy, but voice/face implies anxiety or sadness), explicitly and empathetically address this conflict."*

This ensures the LLM never ignores cross-modal contradictions.

---

## 📡 API Endpoints

### 👤 User Management (`/users`)

- **`POST /users`** — Register a new user. Accepts `name`, `email`, `password` (hashed with bcrypt), and optional `role`. Returns `UserResponse` with `role`.
- **`GET /users`** — Retrieve a list of all registered users. Returns `List[UserResponse]`.
- **`GET /users/me`** — Retrieve the currently authenticated user based on the provided JWT token.
- **`GET /users/{user_id}/sessions`** — Retrieve all sessions belonging to a user, ordered by most recently active (`updated_at` desc). Returns `List[SessionListResponse]`.

### 🔐 Authentication (`/auth`)

- **`POST /auth/login`** — Authenticate with email and password. Returns a JWT `access_token` and the user profile.
- **`POST /auth/logout`** — Stateless logout; prompts the client to discard the JWT.

### 💬 Session Management (`/sessions`)

- **`GET /sessions/{session_id}`** — Retrieve the full conversation history for a session (with all multi-modal emotion data).
- **`DELETE /sessions/{session_id}`** — Permanently delete a session and all its messages (cascading).
- **`PATCH /sessions/{session_id}/title`** — Update the title of a session. Designed for LLM auto-generated titles (e.g., after first turn).

  All session endpoints accept an optional `?user_id=` query param (defaults to the shared default user if omitted).

  **Session History Example:**
  ```json
  {
    "id": "a1b2c3d4-e5f6-7890-1234-56789abcdef0",
    "user_id": "11111111-1111-1111-1111-111111111111",
    "title": "New Chat",
    "created_at": "2026-04-26T10:15:30Z",
    "updated_at": "2026-04-26T10:16:45Z",
    "message_count": 2,
    "messages": [
      {
        "id": "msg-uuid-1",
        "session_id": "a1b2c3d4-...",
        "role": "user",
        "content": "I am feeling anxious about work.",
        "source": "text",
        "timestamp": "2026-04-26T10:15:30Z",
        "text_emotion": "fear",
        "voice_emotion": null,
        "face_emotion": null
      },
      {
        "id": "msg-uuid-2",
        "session_id": "a1b2c3d4-...",
        "role": "assistant",
        "content": "I hear that you're feeling anxious...",
        "source": "text",
        "timestamp": "2026-04-26T10:15:35Z",
        "text_emotion": null,
        "voice_emotion": null,
        "face_emotion": null
      }
    ]
  }
  ```

### 💬 Chat & Conversational Interfaces (`/chat`)

- **`POST /chat/text`** — Send a text message.
  - Body: `{ "text": "...", "session_id": null, "user_id": null }`
  - Returns: response, detected emotions (`text`), RAG sources, `session_id`.

- **`POST /chat/voice`** — Send a voice recording.
  - Form data: `audio` (file), `session_id` (optional), `user_id` (optional).
  - Returns: transcript, dual emotions (`audio` + `text`), response, TTS audio (base64), `session_id`.

- **`POST /chat/tts/engine`** — Switch the active TTS engine at runtime (`edge` or `tiny`).
- **`GET /chat/tts/engine`** — Get the current TTS engine and available options.

### 📚 RAG Operations (`/chat`)
- **`POST /chat/ingest`** — Upload documents (PDF/TXT) to parse, embed, and store in ChromaDB.
- **`GET /chat/collection/stats`** — Retrieve statistics about ingested documents.

### 🔬 Direct Analysis Endpoints
- **`POST /classify/text`** — Direct text emotion classification (standalone, no session).
- **`POST /analyze/voice`** — Direct audio analysis: transcript + text emotion + voice emotion.
- **`POST /analyze/parallel-sentiment`** — STT and audio emotion in parallel; includes timing diagnostics.

---

## 🏗️ Directory Structure

```
backend/
├── main.py                          # FastAPI app: lifespan, CORS, router registration
├── config.py                        # Pydantic Settings (all env vars, incl. DATABASE_URL)
│
├── auth/
│   ├── __init__.py                  # Auth module
│   ├── dependencies.py              # FastAPI JWT dependencies (get_current_user)
│   └── jwt_handler.py               # Token creation and decoding logic
│
├── db/
│   ├── database.py                  # Async engine, AsyncSession factory, init_db(), get_db()
│   └── models.py                    # SQLAlchemy ORM: User, Session, Message (Triple-Emotion)
│
├── models/
│   └── schemas.py                   # Pydantic DTOs: UserCreate/Response, SessionListResponse,
│                                    #   MessageResponse, SessionHistoryResponse, Login, Chat schemas
│
├── routes/
│   ├── auth.py                      # POST /auth/login, POST /auth/logout
│   ├── user.py                      # POST /users, GET /users, GET /users/me, GET /users/{id}/sessions
│   ├── session.py                   # GET|DELETE /sessions/{id}, PATCH /sessions/{id}/title
│   ├── chat.py                      # POST /chat/text, POST /chat/voice, TTS engine
│   ├── rag.py                       # POST /chat/ingest, GET /chat/collection/stats
│   ├── text_emotion.py              # POST /classify/text
│   ├── voice_emotion.py             # POST /analyze/voice
│   ├── parallel_sentiment.py        # POST /analyze/parallel-sentiment
│   └── health.py                    # GET /health
│
├── services/
│   ├── session_manager.py           # SessionManagerDB: async CRUD against PostgreSQL
│   ├── user_service.py              # create_user(), get_user_by_email() with bcrypt
│   ├── langchain_service.py         # LLM chat via LangChain (Gemini / OpenAI)
│   ├── rag_service.py               # ChromaDB retrieval with ONNX embeddings
│   ├── parallel_sentiment_service.py# Whisper STT + Wav2Vec2 in parallel
│   ├── voice_emotion_service.py     # Wav2Vec2 acoustic emotion
│   ├── text_emotion_distilBERT.py   # DistilBERT text emotion
│   ├── onnx_embeddings_service.py   # ONNX-optimized MiniLM embeddings
│   └── tts_service.py               # Edge TTS / TinyTTS synthesis
│
└── utils/
    └── prompts.py                   # System prompt template + build_emotion_context()
                                     #   (dissonance detection logic lives here)
```

---

## 🚀 Setup & Run Guide

### 1. Prerequisites
- Python 3.10+
- FFmpeg (must be installed on your system for audio processing)
- PostgreSQL 14+ running locally (or a remote connection string in `.env`)

### 2. Installation Steps
```bash
# 1. Clone the repository and navigate to the project directory
git clone https://github.com/mehdikhan55/fyp-mental-health-chatbot.git
cd fyp-mental-health-chatbot

# 2. Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\Activate.ps1
# On Linux/Mac:
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
```bash
cp .env.example .env
```
Open `.env` and fill in your configuration:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_actual_api_key_here
TTS_ENGINE=edge
DATABASE_URL=postgresql+asyncpg://postgres:yourpassword@localhost:5432/mentimdb
```

> **Database is auto-created** — `init_db()` calls `Base.metadata.create_all()` on startup (dev mode). No manual migrations needed to get started. For production, use Alembic (`alembic` is already in `requirements.txt`).

### 4. Running the Application
**Start the FastAPI Backend:**
```bash
# From the project root
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
*Wait until you see:* `✅ All models and services loaded successfully!`

The backend APIs will be accessible at `http://127.0.0.1:8000`.  
Interactive Swagger UI: `http://127.0.0.1:8000/docs`

**(Optional) Start the Testing Frontend:**
```bash
streamlit run frontend/app.py
```
Opens the testing interface at `http://localhost:8501`.