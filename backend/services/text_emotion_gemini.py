import os
import time
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ClientError

# --- Secure Key Loading ---
load_dotenv()
API_KEY_VALUE = os.getenv('GEMINI_API_KEY')

if not API_KEY_VALUE:
    raise ValueError("FATAL: GEMINI_API_KEY not found in .env file.")

# --- Client Initialization ---
client = genai.Client(api_key=API_KEY_VALUE)
print("✅ Gemini Client Initialized.")

# --- 2. Classification Function with "Patience" Logic ---

# 🛑 SAFETY WRAPPER: This forces the code to wait if it hits a 429 error
@retry(
    retry=retry_if_exception_type(ClientError), # Only retry on API errors
    wait=wait_exponential(multiplier=1, min=4, max=10), # Wait 4s, then 8s, then 10s
    stop=stop_after_attempt(3) # Stop after 3 tries so it doesn't freeze forever
)
def generate_with_retry(model, contents, config):
    """
    Helper function to call Gemini with automatic retry logic.
    """
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

def classify_text_emotion(text_input: str) -> dict:
    """
    Analyzes text input using Gemini 2.5 Flash.
    Includes rate-limit handling.
    """
    
    # 1. System Instructions
    system_instruction = (
        "You are an expert Text Emotion Classifier. "
        "Classify the primary emotion into: 'happy', 'sad', 'angry', 'neutral', 'fear', 'surprise', 'disgust'. "
        "Return JSON with keys: 'emotion' and 'confidence_score'."
    )

    # 2. The Prompt
    prompt = f"Classify the emotion in this text: \"{text_input}\""

    # 3. Config
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {
                "emotion": {"type": "string"},
                "confidence_score": {"type": "number"}
            },
            "required": ["emotion", "confidence_score"]
        }
    )

    try:
        # 4. Call API (using the Retry wrapper)
        response = generate_with_retry(
            model='gemini-2.5-flash',
            contents=[prompt],
            config=config
        )

        # 5. Parse Result
        result_dict = json.loads(response.text)
        return result_dict

    except Exception as e:
        # If we failed after 3 retries, return a "Busy" state instead of crashing
        print(f"❌ API Error after retries: {e}")
        return {
            "emotion": "Server Busy (Try Again)", 
            "confidence_score": 0.0
        }

# --- 3. Test Block ---
if __name__ == "__main__":
    print("Testing single request...")
    print(classify_text_emotion("I am happy"))