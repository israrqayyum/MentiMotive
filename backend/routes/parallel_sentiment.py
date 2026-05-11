from fastapi import APIRouter, HTTPException, UploadFile, File, status
import logging
import tempfile
import os
import shutil
from models.schemas import ParallelSentimentResponse
from services.parallel_sentiment_service import analyze_parallel_sentiment

router = APIRouter(prefix="/analyze", tags=["Parallel Sentiment"])


@router.post(
    "/parallel-sentiment",
    response_model=ParallelSentimentResponse,
    summary="Audio → parallel text + audio emotion (timed)",
    description="**STT (Whisper) and the audio emotion model** run in parallel on the file; after transcription, **text emotion** is run on the transcript. "
    "The fields `path1_text_emotion` and `path2_audio_emotion` are the two modalities. "
    "HTTP **200** only returns `success: true`; a failed run is **500** with `detail` (no `success: false` in the body for this route).",
    responses={500: {"description": "Pipeline or model error; `detail` carries the message from the service."}},
)
async def analyze_parallel_sentiment_endpoint(
    file: UploadFile = File(
        ...,
        description="Audio file. Same format constraints as other `/analyze/...` routes. Filename is used for a temp file suffix only.",
    )
) -> ParallelSentimentResponse:
    logging.info(f"🎯 Parallel sentiment analysis request: {file.filename}")

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.filename)[1]
    ) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        result = await analyze_parallel_sentiment(temp_path)

        if not result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Unknown error"),
            )

        os.unlink(temp_path)
        return result

    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        logging.error(f"❌ Parallel sentiment analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e
