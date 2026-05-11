import edge_tts
import asyncio
import time
import os

# ================= CONFIG =================

# 🎤 Voice options (uncomment one)

VOICE = "en-IN-NeerjaExpressiveNeural"      # Female (India) ✅ current
# VOICE = "en-IN-PrabhatNeural"             # Male (India)
# VOICE = "en-US-JennyNeural"               # Female (US)
# VOICE = "en-US-GuyNeural"                 # Male (US)
# VOICE = "en-GB-SoniaNeural"               # Female (UK)
# VOICE = "en-GB-RyanNeural"                # Male (UK)

# ⚡ Speed control (rate)
# Format: "+25%" = 1.25x, "+50%" = 1.5x, "-10%" = slower

# RATE = "+0%"        # Normal speed
RATE = "+25%"     # 1.25x
# RATE = "+50%"     # 1.5x
# RATE = "-10%"     # slower

OUTPUT_FILE = "edge_tts_output.mp3"

# ==========================================


async def generate_speech(text: str, output_path: str = OUTPUT_FILE):
    print(f"🎤 Generating speech with Edge TTS...")
    print(f"📝 Text: {text}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_output_path = os.path.join(script_dir, output_path)

    start_time = time.time()

    try:
        communicate = edge_tts.Communicate(
            text=text,
            voice=VOICE,
            rate=RATE
        )
        await communicate.save(full_output_path)

        time_ms = (time.time() - start_time) * 1000

        print(f"✅ Speech generated successfully!")
        print(f"📁 Saved to: {full_output_path}")
        print(f"⏱️ Time taken: {time_ms:.2f} ms")

        return True, time_ms

    except Exception as e:
        print(f"❌ Error: {e}")
        return False, 0


if __name__ == "__main__":
    test_sentence = "I’ve been feeling overwhelmed lately, like everything is piling up faster than I can handle. Some days my mind just won’t slow down, and even small tasks feel exhausting. I keep wondering if I’m doing enough or if I’m falling behind."
    
    print("=" * 60)
    print("Edge TTS - Configurable Voice & Speed")
    print("=" * 60)

    success, time_ms = asyncio.run(generate_speech(test_sentence))

    if success:
        print(f"\n🎉 Success! Audio generated in {time_ms:.2f} ms")
    else:
        print("\n❌ Failed to generate audio")