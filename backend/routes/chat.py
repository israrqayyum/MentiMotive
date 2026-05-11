from sqlalchemy.ext.asyncio import AsyncSession
from db.database import get_db
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    status,
    Query,
    Depends
)
from typing import Optional
from models.schemas import (
    TextChatRequest,
    ChatResponse,
    EmotionData,
    VoiceChatResponse,
    EmotionsVoiceChat,
    TtsEngineGetResponse,
    TtsEngineSetResponse,
)
from services.session_manager import session_manager
from services.langchain_service import langchain_service
from services.parallel_sentiment_service import get_analyzer as get_parallel_analyzer
from services.text_emotion_distilBERT import get_analyzer as get_text_analyzer
from services.tts_service import tts_service
import logging
import tempfile
import os
import time
import base64

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = logging.getLogger(__name__)


@router.post(
    "/voice",
    response_model=VoiceChatResponse,
    summary="Conversational voice turn (STT, emotions, RAG, TTS)",
    description="Multipart: **one audio** file, optional `session_id`. **Step 1** calls the same `ParallelSentimentAnalyzer` as `POST /analyze/parallel-sentiment` (Whisper + audio + text-on-transcript, then `path1` / `path2` **labels**). **Step 2** remaps those to `emotion` + `confidence` and runs LangChain with RAG. **Step 3** TTS. "
    "`emotions` in the JSON are **always** `emotion`+`confidence` (chat shape), not `label`+`confidence` like the bare analyze route.",
    responses={
        500: {
            "description": "Parallel analysis failed (`success: false`), **LLM `invoke` raised**, or another uncaught error. RAG **retrieval** errors are **not** returned as 500: `rag_service.retrieve_context` swallows them into context text and empty `rag_sources` (the handler may still return 200 unless the LLM also fails). TTS failure yields **200** with text and `audio: null`."
        }
    },
)
async def voice_chat(
    audio: UploadFile = File(
        ...,
        description="User utterance, typically `wav` (any extension saved as temp input for the analyzer).",
    ),
    session_id: Optional[str] = Form(
        None,
        description="Omit, or pass an existing id. **Empty string** is treated like omit (a **new** id is generated). Same as `TextChatRequest.session_id`.",
    ),
    user_id: Optional[str] = Form(
        None,
        description="The UUID of the user making the request. Uses a default if omitted."
    ),
    face_emotion: Optional[str] = Form(
        None,
        description="Face emotion detected via the client camera at send time (e.g. happy, sad, neutral). Omitted when camera is off or no face was detected.",
    ),
    db: AsyncSession = Depends(get_db)
) -> VoiceChatResponse:
    try:
        total_start = time.time()
        logger.info("\n" + "=" * 60)
        logger.info("🎤 VOICE CHAT REQUEST - PIPELINE")
        logger.info("=" * 60)
        logger.info(
            f"📥 Request fields:"
            f"\n   session_id   : {session_id!r}"
            f"\n   user_id      : {user_id!r}"
            f"\n   face_emotion : {face_emotion!r}"
            f"\n   audio        : <{audio.filename or 'upload'}, {audio.content_type}> (bytes omitted)"
        )

        effective_user_id = user_id or "11111111-1111-1111-1111-111111111111"
        session = await session_manager.get_or_create_session(db, effective_user_id, session_id)
        session_id = str(session.id)

        # Derive a file extension from the upload's content-type so downstream
        # libraries (soundfile, ffmpeg) get a meaningful hint.
        _CT_TO_EXT = {
            "audio/wav": ".wav", "audio/x-wav": ".wav",
            "audio/webm": ".webm", "audio/ogg": ".ogg",
            "audio/mp4": ".m4a", "audio/m4a": ".m4a",
            "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
            "audio/amr": ".amr", "audio/3gpp": ".3gp",
            "audio/aac": ".aac",
        }
        ct = (audio.content_type or "").split(";")[0].strip().lower()
        suffix = _CT_TO_EXT.get(ct, ".wav")
        logger.info(f"   content_type={ct!r} → temp suffix={suffix!r}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            content = await audio.read()
            temp_audio.write(content)
            temp_path = temp_audio.name

        t1 = time.time()
        analyzer = get_parallel_analyzer()
        result = await analyzer.analyze_parallel(temp_path)
        t2 = time.time()
        logger.info(f"\n⏱️  Step 1: Voice Analysis: {round((t2 - t1) * 1000, 2)} ms")

        os.unlink(temp_path)

        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Voice analysis failed: {result.get('error', 'Unknown error')}",
            )

        transcript = result["transcript"]

        emotions = {
            "audio": {
                "emotion": result["path2_audio_emotion"]["label"],
                "confidence": result["path2_audio_emotion"]["confidence"],
            },
            "text": {
                "emotion": result["path1_text_emotion"]["label"],
                "confidence": result["path1_text_emotion"]["confidence"],
            },
        }
        if face_emotion:
            emotions["face"] = {"emotion": face_emotion, "confidence": 1.0}

        history = await session_manager.get_langchain_history(db, session_id, effective_user_id)

        logger.info(f"\n⏱️  Step 2: Generating AI Response...")
        t3 = time.time()
        response_text, sources = langchain_service.chat(
            user_message=transcript,
            emotions=emotions,
            history=history,
            use_rag=True,
        )
        t4 = time.time()
        logger.info(f"⏱️  Step 2 Total: {round((t4 - t3) * 1000, 2)} ms")

        logger.info(f"\n⏱️  Step 3: Generating Audio Response...")
        t5 = time.time()
        success, audio_bytes, tts_time_ms, engine_used = await tts_service.generate_speech(
            response_text
        )
        t6 = time.time()
        logger.info(
            f"⏱️  Step 3 Total: {round((t6 - t5) * 1000, 2)} ms (Engine: {engine_used})"
        )

        audio_base64 = None
        if success and audio_bytes:
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        else:
            logger.warning("⚠️  TTS failed, returning text-only response")

        await session_manager.add_message(
            db=db,
            session_id=session_id,
            user_id=effective_user_id,
            role="user",
            content=transcript,
            source="voice",
            text_emotion=emotions["text"]["emotion"],
            voice_emotion=emotions["audio"]["emotion"],
            face_emotion=face_emotion
        )

        await session_manager.add_message(
            db=db,
            session_id=session_id,
            user_id=effective_user_id,
            role="assistant",
            content=response_text,
            source="text"
        )

        total_end = time.time()
        logger.info(f"\n📊 VOICE CHAT SUMMARY")
        logger.info(f"   Total Time: {round((total_end - total_start) * 1000, 2)} ms")
        logger.info(f"   Transcript: {transcript[:50]}...")
        face_em_label = emotions.get("face", {}).get("emotion", "n/a")
        logger.info(
            f"   Emotions: Audio={emotions['audio']['emotion']}, Text={emotions['text']['emotion']}, Face={face_em_label}"
        )
        logger.info(f"   TTS Engine: {engine_used}")
        logger.info("=" * 60 + "\n")

        return VoiceChatResponse(
            session_id=session_id,
            transcript=transcript,
            emotions=EmotionsVoiceChat(
                audio=EmotionData(**emotions["audio"]),
                text=EmotionData(**emotions["text"]),
            ),
            response=response_text,
            rag_sources=sources,
            audio=audio_base64,
            audio_format="mp3" if engine_used == "edge" else "wav",
            tts_engine=engine_used,
        )

    except Exception as e:
        logger.error(f"Voice chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e


@router.post(
    "/text",
    response_model=ChatResponse,
    summary="Conversational text turn (emotion, RAG, no TTS)",
    description="JSON body. Creates or continues a `session_id`. Same LangChain+retrieval path as **voice** (no TTS). Text emotion uses the **standalone** text analyzer from `text_emotion_distilBERT` (not the parallel service).",
    responses={
        500: {
            "description": "LLM `invoke` or another uncaught error. RAG lookup failures do **not** 500: they are converted to error context inside `langchain_service` (empty `rag_sources` possible)."
        }
    },
)
async def text_chat(request: TextChatRequest, db: AsyncSession = Depends(get_db)) -> ChatResponse:
    try:
        total_start = time.time()
        logger.info("\n" + "=" * 60)
        logger.info("💬 TEXT CHAT REQUEST - PIPELINE")
        logger.info("=" * 60)

        effective_user_id = request.user_id or "11111111-1111-1111-1111-111111111111"
        session = await session_manager.get_or_create_session(db, effective_user_id, request.session_id)
        session_id = str(session.id)

        logger.info(f"\n⏱️  Step 1: Text Emotion Analysis...")
        t1 = time.time()
        analyzer = get_text_analyzer()
        emotion_result = analyzer.classify(request.text)
        t2 = time.time()
        logger.info(f"⏱️  Step 1 Total: {round((t2 - t1) * 1000, 2)} ms")

        emotion_data = {
            "emotion": emotion_result["label"],
            "confidence": emotion_result["score"],
        }

        emotions = {
            "text": emotion_data,
        }

        history = await session_manager.get_langchain_history(db, session_id, effective_user_id)

        logger.info(f"\n⏱️  Step 2: Generating AI Response...")
        t3 = time.time()
        response_text, sources = langchain_service.chat(
            user_message=request.text,
            emotions=emotions,
            history=history,
            use_rag=True,
        )
        t4 = time.time()
        logger.info(f"⏱️  Step 2 Total: {round((t4 - t3) * 1000, 2)} ms")

        await session_manager.add_message(
            db=db,
            session_id=session_id,
            user_id=effective_user_id,
            role="user",
            content=request.text,
            source="text",
            text_emotion=emotions["text"]["emotion"],
            voice_emotion=None,
            face_emotion=request.face_emotion
        )

        await session_manager.add_message(
            db=db,
            session_id=session_id,
            user_id=effective_user_id,
            role="assistant",
            content=response_text,
            source="text"
        )

        total_end = time.time()
        logger.info(f"\n📊 TEXT CHAT SUMMARY")
        logger.info(f"   Total Time: {round((total_end - total_start) * 1000, 2)} ms")
        logger.info(f"   User Message: {request.text[:50]}...")
        logger.info(
            f"   Emotion: {emotions['text']['emotion']} ({emotions['text']['confidence']})"
        )
        logger.info("=" * 60 + "\n")

        return ChatResponse(
            session_id=session_id,
            emotions=emotions,
            response=response_text,
            rag_sources=sources,
        )

    except Exception as e:
        logger.error(f"Text chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e





@router.post(
    "/tts/engine",
    response_model=TtsEngineSetResponse,
    summary="Set active TTS engine (query param)",
    description="**Query parameter** `engine` = `edge` (Microsoft Edge / higher quality) or `tiny` (local). Matches `requests.post(..., params={\"engine\": ...})` clients. Does not re-test the network; applies for subsequent TTS in voice chat.",
    responses={400: {"description": "Value other than `edge` or `tiny`."}},
)
async def set_tts_engine(
    engine: str = Query(
        ...,
        description="`edge` — online Edge TTS (e.g. mp3); `tiny` — offline, wav output from TinyTTS.",
        example="edge",
    )
) -> TtsEngineSetResponse:
    if engine not in ["edge", "tiny"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid engine. Must be 'edge' or 'tiny'",
        )

    try:
        # Note: this switches the in-memory engine used by `tts_service`.
        # It does not persist back into `.env` or environment variables.
        tts_service.set_engine(engine)
        logger.info(f"TTS engine switched to: {engine}")

        return TtsEngineSetResponse(
            status="success",
            engine=engine,
            message=f"TTS engine set to {engine}",
        )
    except Exception as e:
        logger.error(f"Set TTS engine error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e


@router.get(
    "/tts/engine",
    response_model=TtsEngineGetResponse,
    summary="Get current TTS engine",
    description="Returns the active engine and the list of values accepted by the POST endpoint above.",
)
async def get_tts_engine() -> TtsEngineGetResponse:
    return TtsEngineGetResponse(
        engine=tts_service.tts_engine,
        available_engines=["edge", "tiny"],
    )
