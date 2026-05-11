# MentiMotive - AI Mental Health Chatbot

An empathetic AI chatbot that analyzes emotions across multiple modalities (text, voice, facial expressions) to provide mental health support with context-aware responses.

## Architecture

```mermaid
graph TB
    subgraph Client["Client Layer"]
        UI[Streamlit UI]
        API_Client[API Client]
    end

    subgraph API["FastAPI Backend"]
        Auth[Auth Service<br/>JWT + bcrypt]
        Chat[Chat Routes<br/>text/voice]
        Session[Session Manager]
        RAG_Route[RAG Routes]
    end

    subgraph AI["AI Processing Pipeline"]
        STT[Whisper STT<br/>Speech-to-Text]
        TxtEmo[DistilBERT<br/>Text Emotion]
        VoiceEmo[Wav2Vec2<br/>Voice Emotion]
        Dissonance[Dissonance<br/>Detection]
        LLM[LLM<br/>GPT-4o/Gemini]
        TTS[TTS Engine<br/>Edge/Tiny]
    end

    subgraph Storage["Data Layer"]
        PG[(PostgreSQL<br/>Users/Sessions/Messages)]
        Chroma[(ChromaDB<br/>Vector Store)]
    end

    subgraph Models["ML Models"]
        Embeddings[ONNX MiniLM<br/>Embeddings]
    end

    UI --> API_Client
    API_Client --> Auth
    API_Client --> Chat
    API_Client --> RAG_Route

    Chat --> Session
    Chat --> STT
    Chat --> TxtEmo
    Chat --> VoiceEmo
    
    STT --> TxtEmo
    TxtEmo --> Dissonance
    VoiceEmo --> Dissonance
    
    Dissonance --> LLM
    RAG_Route --> Embeddings
    Embeddings --> Chroma
    Chroma --> LLM
    
    LLM --> TTS
    
    Session --> PG
    Auth --> PG
    
    style AI fill:#e1f5ff
    style Storage fill:#fff4e1
    style Models fill:#f0e1ff
```

## Features

- **Multi-Modal Emotion Analysis**: Detects emotions from text, voice tone, and facial expressions (future)
- **Emotional Dissonance Detection**: Identifies conflicts between what users say vs. how they sound
- **RAG-Enhanced Responses**: Retrieves relevant mental health documents for context-aware replies
- **Multi-Session Support**: PostgreSQL-backed persistent chat sessions with user authentication
- **Flexible LLM**: Supports OpenAI GPT and Google Gemini models
- **Voice I/O**: Speech-to-text input and text-to-speech output (Edge TTS or local TinyTTS)

## Tech Stack

**Backend**: FastAPI, SQLAlchemy (async), PostgreSQL  
**AI/ML**: Transformers, Whisper, Wav2Vec2, DistilBERT, LangChain, ChromaDB  
**Auth**: JWT, bcrypt  
**Frontend**: Streamlit (testing interface)

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 14+
- FFmpeg (for audio processing)

### Installation

```bash
# Clone and navigate
git clone https://github.com/mehdikhan55/fyp-mental-health-chatbot.git
cd fyp-mental-health-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/mentimotive
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
TTS_ENGINE=edge
```

### Run

```bash
# Start backend
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Start frontend (optional)
streamlit run frontend/app.py
```

Access:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- UI: http://localhost:8501

## API Endpoints

### Authentication
- `POST /auth/login` - Login with email/password
- `POST /auth/logout` - Logout

### Users
- `POST /users` - Register new user
- `GET /users/me` - Get current user
- `GET /users/{user_id}/sessions` - List user sessions

### Chat
- `POST /chat/text` - Send text message
- `POST /chat/voice` - Send voice message
- `POST /chat/ingest` - Upload documents for RAG

### Sessions
- `GET /sessions/{session_id}` - Get conversation history
- `DELETE /sessions/{session_id}` - Delete session
- `PATCH /sessions/{session_id}/title` - Update session title

## Database Schema

```
users (id, name, email, password_hash, created_at)
  ↓ 1:N
sessions (id, user_id, title, created_at, updated_at)
  ↓ 1:N
messages (id, session_id, role, content, source, timestamp,
          text_emotion, voice_emotion, face_emotion, metadata_json)
```

**Triple-Emotion Model**: Each message stores emotions from text, voice, and face (future) independently.

## Project Structure

```
├── backend/
│   ├── main.py              # FastAPI app entry
│   ├── config.py            # Environment settings
│   ├── auth/                # JWT authentication
│   ├── db/                  # Database models & connection
│   ├── routes/              # API endpoints
│   ├── services/            # Business logic & ML models
│   └── utils/               # Prompts & helpers
├── frontend/
│   └── app.py               # Streamlit testing UI
├── requirements.txt
└── .env.example
```

## License

MIT

## Contributors

Built as a Final Year Project for mental health support research.
