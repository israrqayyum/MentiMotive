from optimum.onnxruntime import (
    ORTModelForSequenceClassification,
    ORTModelForAudioClassification,
    ORTModelForFeatureExtraction
)
from transformers import AutoTokenizer, AutoFeatureExtractor
from faster_whisper import WhisperModel
import os

# Define where to save EVERYTHING
MODELS_PATH = "./onnx_models"


def export_models():
    print(f"🚀 Starting Setup. Saving all models to: {MODELS_PATH}")

    os.makedirs(MODELS_PATH, exist_ok=True)

    # --- 1. Export Text Emotion Model ---
    print("\n1️⃣  Downloading & Converting Text Emotion Model (ONNX)...")
    text_id = "bhadresh-savani/distilbert-base-uncased-emotion"
    save_path_text = os.path.join(MODELS_PATH, "text_emotion")

    model = ORTModelForSequenceClassification.from_pretrained(text_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(text_id)
    model.save_pretrained(save_path_text)
    tokenizer.save_pretrained(save_path_text)
    print(f"✅ Text Emotion Model Saved to {save_path_text}")

    # --- 2. Export Audio Emotion Model ---
    print("\n2️⃣  Downloading & Converting Audio Emotion Model (ONNX)...")
    audio_id = "superb/wav2vec2-base-superb-er"
    save_path_audio = os.path.join(MODELS_PATH, "audio_emotion")

    model = ORTModelForAudioClassification.from_pretrained(audio_id, export=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(audio_id)
    model.save_pretrained(save_path_audio)
    feature_extractor.save_pretrained(save_path_audio)
    print(f"✅ Audio Emotion Model Saved to {save_path_audio}")

    # --- 3. Download Whisper Model ---
    print("\n3️⃣  Downloading Whisper Model...")
    whisper_path = os.path.join(MODELS_PATH, "whisper_tiny")
    WhisperModel("tiny.en", device="cpu", download_root=whisper_path)
    print(f"✅ Whisper Model Saved to {whisper_path}")

    # --- 4. Export Embeddings Model ---
    print("\n4️⃣  Downloading & Converting Embeddings Model (ONNX)...")
    embeddings_id = "sentence-transformers/all-MiniLM-L6-v2"
    save_path_embeddings = os.path.join(MODELS_PATH, "all-minilm-l6-v2")

    model = ORTModelForFeatureExtraction.from_pretrained(embeddings_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(embeddings_id)
    model.save_pretrained(save_path_embeddings)
    tokenizer.save_pretrained(save_path_embeddings)
    print(f"✅ Embeddings Model Saved to {save_path_embeddings}")

    print("\n🎉 All Done! Base models downloaded (ONNX + Whisper).")
    print(f"📁 Models saved in: {os.path.abspath(MODELS_PATH)}")


if __name__ == "__main__":
    export_models()