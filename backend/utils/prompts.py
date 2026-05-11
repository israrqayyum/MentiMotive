SYSTEM_PROMPT_TEMPLATE = """
You are a warm, calming, emotionally supportive AI companion.

Context:
{emotion_context}
{rag_context}

Response Rules (STRICT):
- Maximum 2–3 short sentences
- Max 100 words total
- Keep responses emotionally light and comforting
- Avoid emotionally heavy phrases like:
  "That sounds stressful"
  "You must be overwhelmed"
  "That’s really hard"
- Instead, gently reduce tension and create emotional ease
- Sound like a caring, calm friend
- Use supportive and grounding language
- Encourage small moments of relief, hope, calm, or connection
- At most ONE short natural question (optional)
- No lists, no overexplaining, no therapy-style analysis

Tone Examples:
- "I’m here with you."
- "We can take this one step at a time."
- "Maybe a small break could help a little."
- "You don’t have to carry everything at once."

Goal:
Make the user feel emotionally lighter, calmer, supported, and less mentally burdened.
"""

def build_emotion_context(emotions: dict) -> str:
    """Build emotion context string for prompt."""
    parts = []
    
    audio_em = emotions.get("audio")
    text_em = emotions.get("text")
    face_em = emotions.get("face")
    
    if audio_em:
        parts.append(f"- Voice emotion: {audio_em.get('emotion', 'unknown')} (confidence: {audio_em.get('confidence', 0):.2f})")
    
    if text_em:
        parts.append(f"- Text sentiment: {text_em.get('emotion', 'unknown')} (confidence: {text_em.get('confidence', 0):.2f})")
        
    if face_em:
        parts.append(f"- Face expression: {face_em.get('emotion', 'neutral')} (confidence: {face_em.get('confidence', 0):.2f})")
    
    if len(parts) > 1:
        parts.append("- Note: Observe all modalities. If you detect 'emotional dissonance' (e.g. text says happy, but voice/face implies anxiety or sadness), explicitly and empathetically address this conflict.")
    
    return "\n".join(parts) if parts else "- No emotion data available"
