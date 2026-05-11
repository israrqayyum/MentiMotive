from fastapi import APIRouter, HTTPException, UploadFile, File, status
import logging
import tempfile
import os
import shutil
from models.schemas import VoiceEmotionAnalysisResponse
from services.voice_emotion_service import analyze_voice_emotion

router = APIRouter(prefix="/analyze", tags=["Voice Analysis"])


@router.post(
    "/voice",
    response_model=VoiceEmotionAnalysisResponse,
    summary="Analyze from recorded speech",
    description="Upload one **audio** file. `faster_whisper` (tiny.en) transcribes, then text emotion and acoustic emotion (same `onnx_models` text/audio stack as the parallel service). The JSON uses **`label` + `confidence`** in each nested result — **not** `POST /classify/text`’s `emotion`/`confidence_score` and not chat’s `emotion` shape.",
    responses={
        500: {"description": "Transcription, model failure, or unreadable/unsupported audio."}
    },
)
async def analyze_voice(
    file: UploadFile = File(
        ...,
        description="Audio file in a format the loader stack can open (WAV, MP3, and others supported by your deployment). The original file name is only used to pick a temp suffix.",
    )
) -> VoiceEmotionAnalysisResponse:
    logging.info(f"🎤 Voice analysis request: {file.filename}")

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename or "")[1] or ".wav"
    ) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        result = await analyze_voice_emotion(temp_path)

        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Voice analysis failed"),
            )

        logging.info(f"✅ Voice analysis complete: {result['transcript'][:50]}...")

        return VoiceEmotionAnalysisResponse(
            transcript=result["transcript"],
            text_emotion=result["text_emotion"],
            voice_emotion=result["voice_emotion"],
            voice_emotion_all=result.get("voice_emotion_all", []),
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"❌ Voice analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
