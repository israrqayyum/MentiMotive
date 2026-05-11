import os
import nltk

# Point NLTK to our local offline data folder
MODELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "onnx_models"))
NLTK_DATA_PATH = os.path.join(MODELS_PATH, "nltk_data")
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_PATH)

from tiny_tts import TinyTTS
import time
import soundfile as sf
import numpy as np

# ================= CONFIG =================

# ⚡ Speed factor
# 1.0 = normal
# 1.25 = faster
# 1.5 = much faster

SPEED = 1.0
#But remember if we change the speed to 1.25 or 1.5 then audio quality is not good so 1.0 is best.
# SPEED = 1.25
# SPEED = 1.5
#Additionaly the tinytts only supports only one built-in model (one voice).

OUTPUT_FILE = "tiny_tts_output.wav"

# ==========================================


def change_speed(audio, speed):
    """Simple speed control (resampling)"""
    indices = np.round(np.arange(0, len(audio), speed))
    indices = indices[indices < len(audio)].astype(int)
    return audio[indices]


def generate_speech(text: str, output_path: str = OUTPUT_FILE):
    print(f"🎤 Generating speech with TinyTTS...")
    print(f"📝 Text: {text}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_output_path = os.path.join(script_dir, output_path)

    start_time = time.time()

    try:
        tts = TinyTTS()
        temp_path = full_output_path.replace(".wav", "_temp.wav")

        # Generate original audio
        tts.speak(text, output_path=temp_path)

        # Load audio
        audio, sr = sf.read(temp_path)

        # Apply speed change
        if SPEED != 1.0:
            audio = change_speed(audio, SPEED)

        # Save final output
        sf.write(full_output_path, audio, sr)

        # Remove temp file
        os.remove(temp_path)

        time_ms = (time.time() - start_time) * 1000

        print(f"✅ Speech generated successfully!")
        print(f"📁 Saved to: {full_output_path}")
        print(f"⚡ Speed: {SPEED}x")
        print(f"⏱️ Time taken: {time_ms:.2f} ms")

        return True, time_ms

    except Exception as e:
        print(f"❌ Error: {e}")
        return False, 0


if __name__ == "__main__":
    test_sentence = "I’ve been feeling overwhelmed lately, like everything is piling up faster than I can handle. Some days my mind just won’t slow down, and even small tasks feel exhausting. I keep wondering if I’m doing enough or if I’m falling behind."
    print("=" * 60)
    print("TinyTTS - Speed Control")
    print("=" * 60)

    success, time_ms = generate_speech(test_sentence)

    if success:
        print(f"\n🎉 Success! Audio generated in {time_ms:.2f} ms")
    else:
        print("\n❌ Failed to generate audio")