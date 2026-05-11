from fastapi import APIRouter, HTTPException, status
from models.schemas import TextEmotionRequest, TextEmotionResponse
import logging
from services.text_emotion_distilBERT import classify_text_emotion

router = APIRouter(prefix="/classify", tags=["Text Emotion"])


@router.post(
    "/text",
    response_model=TextEmotionResponse,
    summary="Classify emotion from text",
    description="Runs the local ONNX text emotion classifier (see `onnx_models/text_emotion`). The **response** `emotion` field is the top label, lowercased.",
    responses={
        422: {"description": "Validation error (e.g. empty `text` — violates `min_length: 1.`)."},
        400: {"description": "Non-empty body but `text` is only whitespace; empty after `strip`."},
        500: {"description": "Classifier returned no `label` or the model is misconfigured."},
    },
)
async def classify_text(request: TextEmotionRequest) -> TextEmotionResponse:
    user_text = request.text.strip()

    if not user_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text input cannot be empty.",
        )

    logging.info(f"🔹 Text classification request: {user_text}")

    result = classify_text_emotion(user_text)

    if not result or "label" not in result:
        logging.error("❌ Text classification failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to classify text. Check model configuration.",
        )

    emotion = result["label"].lower()
    confidence = float(result.get("score", 0.0))

    logging.info(f"✅ Emotion: {emotion} | Confidence: {confidence:.2f}")

    return TextEmotionResponse(
        emotion=emotion,
        confidence_score=confidence,
    )
