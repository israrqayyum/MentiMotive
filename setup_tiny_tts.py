import os
from huggingface_hub import hf_hub_download
import nltk

# Define where to save EVERYTHING
MODELS_PATH = "./onnx_models"
NLTK_DATA_PATH = os.path.join(MODELS_PATH, "nltk_data")

def setup_nltk():
    """Download required NLTK resources into the local project folder to make it fully portable."""
    print(f"\n📦 Setting up NLTK resources in {NLTK_DATA_PATH}...")

    os.makedirs(NLTK_DATA_PATH, exist_ok=True)
    
    # Add our local directory to NLTK's data path so it knows to look there
    if NLTK_DATA_PATH not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA_PATH)

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        nltk.data.find('corpora/cmudict')
        print("✅ NLTK already configured locally")
    except LookupError:
        print("⬇️ Downloading NLTK datasets offline...")
        nltk.download('averaged_perceptron_tagger', download_dir=NLTK_DATA_PATH)
        nltk.download('averaged_perceptron_tagger_eng', download_dir=NLTK_DATA_PATH)
        nltk.download('cmudict', download_dir=NLTK_DATA_PATH)
        print("✅ NLTK offline setup complete")

def setup_tiny_tts():
    print(f"🚀 Starting TinyTTS Setup. Saving to: {MODELS_PATH}")

    os.makedirs(MODELS_PATH, exist_ok=True)

    # --- 1. Download TinyTTS Full Model (PyTorch + Pipeline) ---
    print("\n1️⃣  Downloading TinyTTS Full Model (with pipeline)...")

    tinytts_repo = "backtracking/tiny-tts"
    save_path_tts = os.path.join(MODELS_PATH, "tiny_tts")
    os.makedirs(save_path_tts, exist_ok=True)

    hf_hub_download(
        repo_id=tinytts_repo,
        filename="G.pth",  # main model file
        local_dir=save_path_tts
    )

    print(f"✅ TinyTTS model downloaded to {save_path_tts}")

    # --- 2. Setup NLTK (IMPORTANT) ---
    setup_nltk()

    print(f"\n🎉 TinyTTS & NLTK Setup Done! Models saved in: {os.path.abspath(MODELS_PATH)}")

if __name__ == "__main__":
    setup_tiny_tts()
