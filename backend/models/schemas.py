from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
import uuid

# ---------------------------------------------------------------------------
# Users & Sessions (DB Multi-Session Support)
# ---------------------------------------------------------------------------

class UserCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Jane Doe",
                "email": "jane.doe@example.com",
                "password": "securepassword123"
            }
        }
    )
    name: str = Field(..., description="The user's full name.")
    email: str = Field(..., description="A unique email address used for login.", pattern=r"^\S+@\S+\.\S+$")
    password: str = Field(..., min_length=6, description="A raw password that will be hashed using bcrypt before storage.")
    role: Literal["user", "admin"] = Field(default="user", description="User role (user or admin).")

class UserResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "11111111-1111-1111-1111-111111111111",
                "name": "Jane Doe",
                "email": "jane.doe@example.com",
                "role": "user",
                "created_at": "2026-04-26T10:00:00Z",
                "updated_at": "2026-04-26T10:00:00Z"
            }
        }
    )
    id: str = Field(..., description="Database generated UUID for the user.")
    name: str = Field(..., description="The user's registered name.")
    email: str = Field(..., description="The user's registered email.")
    role: str = Field(..., description="The user's role.")
    created_at: datetime = Field(..., description="Timestamp when the user was created.")
    updated_at: datetime = Field(..., description="Timestamp when the user was last updated.")

class SessionListResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "session-uuid-here",
                "title": "Discussion about Anxiety",
                "updated_at": "2026-04-26T10:15:30Z"
            }
        }
    )
    id: str = Field(..., description="Database generated UUID for the active session.")
    title: str = Field(..., description="A concise title for the conversation (default 'New Chat' or auto-named by LLM).")
    updated_at: datetime = Field(..., description="Timestamp of the most recent message exchanged in this session.")


# ---------------------------------------------------------------------------
# Authentication: POST /auth/login, POST /auth/logout
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "jane.doe@example.com",
                "password": "securepassword123"
            }
        }
    )
    email: str = Field(..., description="Registered email address.")
    password: str = Field(..., description="Plain-text password (sent over HTTPS, never stored).")


class LoginResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGci...",
                "token_type": "bearer",
                "user_id": "11111111-1111-1111-1111-111111111111",
                "name": "Jane Doe",
                "email": "jane.doe@example.com",
                "role": "user"
            }
        }
    )
    access_token: str = Field(..., description="JWT access token. Include as `Authorization: Bearer <token>` in subsequent requests.")
    token_type: str = Field(default="bearer", description="Always 'bearer'.")
    user_id: str = Field(..., description="Authenticated user's UUID.")
    name: str = Field(..., description="User's registered name.")
    email: str = Field(..., description="User's registered email.")
    role: str = Field(..., description="User's role.")


class LogoutResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Logged out successfully. Please discard your token."
            }
        }
    )
    message: str = Field(..., description="Confirmation message.")

# ---------------------------------------------------------------------------
# Shared: emotion and chat building blocks
# ---------------------------------------------------------------------------


class EmotionData(BaseModel):
    """Chat/session shape: the model’s top class is stored as the string `emotion` (not `label`). Casing is whatever the underlying classifier returns; only `POST /classify/text` lowercases the label in its own response model."""
    model_config = ConfigDict(
        json_schema_extra={"example": {"emotion": "happy", "confidence": 0.91}}
    )
    emotion: str = Field(..., description="Predicted emotion label (e.g. happy, sad, neutral).")
    confidence: float = Field(
        ...,
        ge=0.0,
        description="Model score for the top `emotion` (0–1 in practice). In voice chat, `text` is from the transcript model and `audio` from the acoustic model.",
    )
    all_scores: Optional[Dict[str, float]] = Field(
        None, description="Optional per-label scores when the model exposes them."
    )


class EmotionResultLabel(BaseModel):
    """Classifier output when the API uses `label` + `confidence` (voice / parallel analysis)."""
    model_config = ConfigDict(
        json_schema_extra={"example": {"label": "happy", "confidence": 0.88}}
    )
    label: str = Field(..., description="Top predicted class label from the model.")
    confidence: float = Field(
        ..., ge=0.0, description="Confidence for the `label` (0–1 scale, rounded in services)."
    )


class MessageResponse(BaseModel):
    """Single turn in a conversation, for API response."""
    id: str
    session_id: str
    role: str
    content: str
    source: Optional[str] = None
    timestamp: datetime
    
    text_emotion: Optional[str] = None
    voice_emotion: Optional[str] = None
    face_emotion: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class SessionHistoryResponse(BaseModel):
    """Session history containing multiple messages, returned by GET /sessions/{id}."""
    id: str = Field(..., description="Session UUID.")
    user_id: str = Field(..., description="Owner user UUID.")
    title: str = Field(..., description="Session title (default: 'New Chat').")
    created_at: datetime = Field(..., description="When the session was created.")
    updated_at: datetime = Field(..., description="When the session was last updated.")
    messages: List[MessageResponse] = Field(default=[], description="Ordered message history.")
    message_count: int = Field(default=0, description="Total number of messages in the session.")

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Text emotion: POST /classify/text
# ---------------------------------------------------------------------------


class TextEmotionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"text": "I am so excited for the weekend!"}}
    )
    text: str = Field(
        ...,
        min_length=1,
        description="Raw text. An empty string fails **request validation (422)** before the route. If only spaces, trim can yield 400 in the route handler.",
    )


class TextEmotionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"emotion": "joy", "confidence_score": 0.87}}
    )
    emotion: str = Field(
        ..., description="Predicted class label, lowercased (e.g. joy, sadness, anger)."
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Top-label score from the `sentiment-analysis` pipeline (see `onnx_models/text_emotion`); usually 0–1.",
    )


# ---------------------------------------------------------------------------
# Voice analysis: POST /analyze/voice
# ---------------------------------------------------------------------------


class VoiceEmotionAnalysisResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transcript": "I feel great today",
                "text_emotion": {"label": "joy", "confidence": 0.92},
                "voice_emotion": {"label": "happy", "confidence": 0.78},
                "voice_emotion_all": [
                    {"label": "happy", "confidence": 0.78},
                    {"label": "neutral", "confidence": 0.12},
                ],
            }
        }
    )
    transcript: str = Field(
        ..., description="Speech-to-text result from the audio file (Whisper, full utterance joined)."
    )
    text_emotion: EmotionResultLabel = Field(
        ..., description="Emotion from the **transcribed text** (text modality)."
    )
    voice_emotion: EmotionResultLabel = Field(
        ..., description="Top **acoustic** emotion (tone) from the audio model."
    )
    voice_emotion_all: List[EmotionResultLabel] = Field(
        default_factory=list,
        description="All ranked audio-emotion labels from the model (highest first).",
    )


# ---------------------------------------------------------------------------
# Parallel sentiment: POST /analyze/parallel-sentiment
# ---------------------------------------------------------------------------


class ParallelTimingBreakdownMs(BaseModel):
    audio_loading: float = Field(..., description="Time to load and prepare audio, in milliseconds.")
    stt_whisper: float = Field(
        ..., description="Speech-to-text (Whisper) segment, in milliseconds."
    )
    audio_emotion: float = Field(..., description="Audio emotion model time, in milliseconds.")
    text_emotion: float = Field(..., description="Text emotion (from transcript) time, in milliseconds.")


class ParallelStats(BaseModel):
    total_processing_ms: float = Field(..., description="End-to-end processing time in milliseconds.")
    breakdown_ms: ParallelTimingBreakdownMs
    parallel_check: str = Field(
        ...,
        description="Heuristic string indicating whether STT and audio runs overlapped (debugging / perf).",
    )


class ParallelSentimentResponse(BaseModel):
    """Success payload from parallel sentiment (both paths on the same file)."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "transcript": "I am frustrated",
                "path1_text_emotion": {"label": "anger", "confidence": 0.86},
                "path2_audio_emotion": {"label": "sad", "confidence": 0.71},
                "path2_audio_emotion_all": [
                    {"label": "sad", "confidence": 0.71}
                ],
                "stats": {
                    "total_processing_ms": 1234.5,
                    "breakdown_ms": {
                        "audio_loading": 20.0,
                        "stt_whisper": 500.0,
                        "audio_emotion": 300.0,
                        "text_emotion": 50.0,
                    },
                    "parallel_check": "✅ Success",
                },
            }
        }
    )
    success: Literal[True] = Field(
        True, description="When present and true, the analysis completed and the remaining fields are valid."
    )
    transcript: str
    path1_text_emotion: EmotionResultLabel = Field(
        ..., description="Path 1: STT → text → **text** emotion (what was said)."
    )
    path2_audio_emotion: EmotionResultLabel = Field(
        ..., description="Path 2: **acoustic** emotion (top label from the audio model)."
    )
    path2_audio_emotion_all: List[EmotionResultLabel] = Field(
        ..., description="All ranked audio emotion labels (same ordering as the underlying model)."
    )
    stats: ParallelStats = Field(
        ..., description="Timing and parallelization diagnostics (useful for performance tuning)."
    )


class VoiceChatRequest(BaseModel):
    """
    **Not** bound in OpenAPI: voice chat uses multipart `session_id` form field.
    Kept for client codegen / parity with the text `session_id` semantics.
    """
    session_id: Optional[str] = Field(
        None,
        description="Same as `TextChatRequest.session_id`: empty string is treated as missing; new UUID issued.",
    )


class TextChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"text": "How can I improve my sleep?", "session_id": None, "user_id": None}}
    )
    user_id: Optional[str] = Field(
        None,
        description="The UUID of the user making the request. Uses a default if omitted.",
    )
    session_id: Optional[str] = Field(
        None,
        description="Existing id, or omit / `null`. **Empty string `""`** is treated the same as missing (a **new** UUID is generated) — see `get_or_create_session` truthiness check.",
    )
    text: str = Field(
        ...,
        min_length=1,
        description="User message. Empty string is **422** (Pydantic). A whitespace-only string is accepted; the text classifier maps strip-empty to neutral. **Label casing in `emotions` is the raw model label** (this route does not lowercase; only `POST /classify/text` lowercases the top label).",
    )
    face_emotion: Optional[str] = Field(
        None,
        description="Face emotion detected via the client camera at send time. Stored on the message; future-friendly for multimodal context.",
    )


class EmotionsVoiceChat(BaseModel):
    """Emotions object returned by **voice** chat (parallel path + text path)."""
    audio: EmotionData = Field(
        ..., description="From parallel analysis: `emotion` and `confidence` for the **acoustic** path."
    )
    text: EmotionData = Field(
        ...,
        description="From parallel analysis: `emotion` and `confidence` from **transcript** text.",
    )


class ChatResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "8f0e2c4a-1b2d-4c5e-9f0a-123456789abc",
                "response": "Here are a few ideas...",
                "emotions": {
                    "text": {"emotion": "neutral", "confidence": 0.62, "all_scores": None}
                },
                "rag_sources": ["/tmp/notes.pdf", "/data/guide.txt"],
            }
        }
    )
    session_id: str = Field(..., description="Server session id (new or the one you passed).")
    response: str = Field(
        ..., description="Assistant text from the LangChain pipeline. `langchain_service.chat` always sets `use_rag=True` for this route; `rag_sources` may be empty with no error."
    )
    emotions: Dict[str, Any] = Field(
        ...,
        description="**Text chat:** `{\"text\": { \"emotion\", \"confidence\" }}` where `emotion` is the classifier’s **raw** label string (casing as returned, unlike `POST /classify/text` which lowercases).",
    )
    rag_sources: List[str] = Field(
        default_factory=list,
        description="From `langchain_service.chat` → RAG: always a **list** (possibly empty). Values are per-chunk `metadata['source']` (often a path). Empty on no matches **or** when retrieval throws (caught in `rag_service` with a placeholder context, `sources` still `[]`).",
    )
    transcript: Optional[str] = Field(
        None, description="Omitted in current text handler; reserved for possible combined shapes."
    )
    audio: Optional[str] = Field(
        None, description="Omitted in text handler. Voice-style flows would use a separate `VoiceChatResponse`."
    )
    audio_format: Optional[str] = Field(
        None, description="Omitted for text. Only meaningful for responses that include synthesized audio."
    )
    tts_engine: Optional[str] = Field(
        None, description="Omitted for `POST /chat/text`. `edge` / `tiny` when a voice response would carry TTS (see `VoiceChatResponse`)."
    )


class VoiceChatResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "...",
                "transcript": "I feel anxious",
                "emotions": {
                    "audio": {"emotion": "sad", "confidence": 0.7},
                    "text": {"emotion": "fear", "confidence": 0.65},
                },
                "response": "I'm sorry you're feeling that way...",
                "rag_sources": [],
                "audio": "<base64...>",
                "audio_format": "mp3",
                "tts_engine": "edge",
            }
        }
    )
    session_id: str
    transcript: str = Field(
        ..., description="Whisper STT on the temp upload (from `ParallelSentimentAnalyzer.analyze_parallel` — same pipeline as `POST /analyze/parallel-sentiment` for this file)."
    )
    emotions: EmotionsVoiceChat
    response: str = Field(..., description="Assistant text reply (also spoken when TTS succeeds).")
    rag_sources: List[str] = Field(
        default_factory=list,
        description="Same as text chat: always a list from `langchain_service.chat` (per-chunk metadata `source` strings, possibly empty).",
    )
    audio: Optional[str] = Field(
        None, description="Base64-encoded TTS output, or `null` if TTS failed (text is still returned)."
    )
    audio_format: Optional[str] = Field(
        None,
        description="`mp3` for Edge, `wav` for TinyTTS, or `null` if there is no audio payload (e.g. both TTS engines failed).",
    )
    tts_engine: str = Field(
        ...,
        description="Engine that produced the audio: `edge`, `tiny`, or `none` if both engines failed and `audio` is null.",
    )



class TtsEngineGetResponse(BaseModel):
    engine: str = Field(..., description="Current default engine: `edge` or `tiny`.")
    available_engines: List[str] = Field(
        default_factory=lambda: ["edge", "tiny"],
        description="All engines the server binary supports switching between.",
    )


class TtsEngineSetResponse(BaseModel):
    status: str = Field("success", description="Always `success` when the switch succeeded.")
    engine: str
    message: str


# ---------------------------------------------------------------------------
# RAG: POST /chat/ingest, GET /chat/collection/stats
# ---------------------------------------------------------------------------


class IngestResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "files_processed": 2,
                "chunks_created": 48,
                "collection": "mentimotive_docs",
            }
        }
    )
    status: str = Field(
        "success",
        description="Always `success` for HTTP 200. Check `files_processed` and `chunks_created` — a run that skips or drops all files can still return 200 with zeros.",
    )
    files_processed: int = Field(
        ...,
        description="Incremented only when `rag_service.ingest_document` completes for that file. **Skipped** extensions and files whose ingest raises in the per-file `try` are not counted. Non-PDF/TXT are skipped without incrementing.",
    )
    chunks_created: int = Field(
        ..., description="Total vector chunks added across all successfully processed files."
    )
    collection: str = Field(
        ..., description="Chroma collection name used for RAG (from `COLLECTION_NAME` / settings)."
    )


class RAGCollectionStatsResponse(BaseModel):
    """RAG service stats. On some failures the service may return only `error` (check both shapes)."""
    model_config = ConfigDict(
        json_schema_extra={"example": {"collection_name": "mentimotive_docs", "document_count": 120}}
    )
    collection_name: Optional[str] = Field(
        None, description="Chroma collection name from settings (omitted on internal error in service)."
    )
    document_count: Optional[int] = Field(
        None, description="Chroma `count()` for the collection (omitted when `error` is set; not an estimate in normal operation)."
    )
    error: Optional[str] = Field(
        None,
        description="Error message if stats could not be read (rare: still returned in body by the service).",
    )


# ---------------------------------------------------------------------------
# Generic root / health
# ---------------------------------------------------------------------------


class RootResponse(BaseModel):
    message: str = Field(..., description="Human-readable API identity string.")
    version: str = Field(..., description="API version string (same as OpenAPI `info.version` when in sync).")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Always `ok` when the process is up.")
    message: str = Field(..., description="Short liveness string for load balancers or probes.")


class ClearSessionResponse(BaseModel):
    status: str = "success"
    message: str = Field(..., description="Confirmation string (session delete is idempotent if the id was unknown).")
