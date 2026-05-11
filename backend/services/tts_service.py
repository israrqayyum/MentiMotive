"""
TTS Service - Text-to-Speech with Edge TTS (primary) and TinyTTS (backup)
Handles audio generation for voice chat responses.
"""

import edge_tts
import asyncio
import time
import os
import tempfile
from typing import Tuple, Optional
import logging
import aiohttp
from config import settings

logger = logging.getLogger(__name__)

# ==================== CONFIG ====================

# Edge TTS Configuration
#EDGE_VOICE = "en-IN-NeerjaExpressiveNeural"  # Female (India)
EDGE_VOICE="en-IN-NeerjaNeural"
EDGE_RATE = "+25%"  # Speed: +0%, +25%, +50%

# TinyTTS Configuration (backup)
TINY_SPEED = 1.0  # 1.0 = normal

# ================================================


class TTSService:
    """Text-to-Speech service with Edge TTS (primary) and TinyTTS (backup)."""

    def __init__(self):
        self.tts_engine = settings.TTS_ENGINE
        self.tiny_tts = None
        self._session = None  # Reusable aiohttp session for connection pooling
        self._edge_warmed = False  # Track if Edge TTS connection is warmed up
        logger.info(f"TTS Service initialized with engine: {self.tts_engine}")

    async def _warmup_edge_tts(self):
        """Pre-warm Edge TTS connection with a dummy request."""
        if self._edge_warmed:
            return

        try:
            logger.info("Warming up Edge TTS connection...")
            communicate = edge_tts.Communicate(
                text="warmup",
                voice=EDGE_VOICE,
                rate=EDGE_RATE
            )
            async for _ in communicate.stream():
                break  # Just establish connection, don't need full audio
            self._edge_warmed = True
            logger.info("✅ Edge TTS connection warmed up")
        except Exception as e:
            logger.warning(f"Edge TTS warmup failed: {e}")

    async def _get_session(self):
        """Get or create aiohttp session for connection reuse."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def initialize_tiny_tts(self):
        """Preload TinyTTS model at server startup."""
        if self.tiny_tts is not None:
            logger.info("TinyTTS already loaded")
            return

        try:
            logger.info("Loading TinyTTS model...")            
            import nltk
            
            # Point NLTK to our local offline data folder before importing tiny_tts
            MODELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "onnx_models"))
            NLTK_DATA_PATH = os.path.join(MODELS_PATH, "nltk_data")
            if NLTK_DATA_PATH not in nltk.data.path:
                nltk.data.path.insert(0, NLTK_DATA_PATH)
            from tiny_tts import TinyTTS
            self.tiny_tts = TinyTTS()
            logger.info("✅ TinyTTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TinyTTS: {e}")
            self.tiny_tts = None

    async def generate_speech_edge(self, text: str) -> Tuple[bool, Optional[bytes], float]:
        """
        Generate speech using Edge TTS (primary, fast, cloud-based).
        Optimized: streams directly to memory, no disk I/O.

        Args:
            text: Text to convert to speech

        Returns:
            Tuple of (success, audio_bytes, time_ms)
        """
        start_time = time.time()

        try:
            # Warmup connection on first use
            if not self._edge_warmed:
                await self._warmup_edge_tts()

            logger.info(f"Generating speech with Edge TTS (voice: {EDGE_VOICE}, rate: {EDGE_RATE})")

            # Preprocess text: remove extra whitespace, limit length
            text = " ".join(text.split())
            if len(text) > 3000:
                text = text[:3000]

            # Stream directly to memory (no disk I/O)
            communicate = edge_tts.Communicate(
                text=text,
                voice=EDGE_VOICE,
                rate=EDGE_RATE
            )

            audio_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])

            audio_bytes = b"".join(audio_chunks)

            time_ms = (time.time() - start_time) * 1000
            logger.info(f"✅ Edge TTS generated audio in {time_ms:.2f} ms")

            return True, audio_bytes, time_ms

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Edge TTS failed: {e}")
            return False, None, time_ms

    def generate_speech_tiny(self, text: str) -> Tuple[bool, Optional[bytes], float]:
        """
        Generate speech using TinyTTS (backup, local, offline).

        Args:
            text: Text to convert to speech

        Returns:
            Tuple of (success, audio_bytes, time_ms)
        """
        start_time = time.time()

        try:
            if self.tiny_tts is None:
                logger.warning("TinyTTS not loaded, attempting to load now...")
                self.initialize_tiny_tts()

            if self.tiny_tts is None:
                logger.error("TinyTTS not available")
                return False, None, 0

            logger.info(f"Generating speech with TinyTTS (speed: {TINY_SPEED}x)")

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_path = temp_file.name

            # Generate speech
            self.tiny_tts.speak(text, output_path=temp_path)

            # Read audio bytes
            with open(temp_path, "rb") as f:
                audio_bytes = f.read()

            # Clean up temp file
            os.unlink(temp_path)

            time_ms = (time.time() - start_time) * 1000
            logger.info(f"✅ TinyTTS generated audio in {time_ms:.2f} ms")

            return True, audio_bytes, time_ms

        except Exception as e:
            time_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ TinyTTS failed: {e}")
            return False, None, time_ms

    async def generate_speech(self, text: str) -> Tuple[bool, Optional[bytes], float, str]:
        """
        Generate speech using configured TTS engine with automatic fallback.

        Args:
            text: Text to convert to speech

        Returns:
            Tuple of (success, audio_bytes, time_ms, engine_used)
        """
        # Try primary engine
        if self.tts_engine == "edge":
            success, audio_bytes, time_ms = await self.generate_speech_edge(text)
            if success:
                return True, audio_bytes, time_ms, "edge"

            # Fallback to TinyTTS
            logger.warning("Edge TTS failed, falling back to TinyTTS...")
            success, audio_bytes, time_ms = self.generate_speech_tiny(text)
            if success:
                return True, audio_bytes, time_ms, "tiny"

        elif self.tts_engine == "tiny":
            success, audio_bytes, time_ms = self.generate_speech_tiny(text)
            if success:
                return True, audio_bytes, time_ms, "tiny"

            # Fallback to Edge TTS
            logger.warning("TinyTTS failed, falling back to Edge TTS...")
            success, audio_bytes, time_ms = await self.generate_speech_edge(text)
            if success:
                return True, audio_bytes, time_ms, "edge"

        # Both failed
        logger.error("Both TTS engines failed")
        return False, None, 0, "none"

    def set_engine(self, engine: str):
        """
        Switch TTS engine.

        Args:
            engine: "edge" or "tiny"
        """
        if engine not in ["edge", "tiny"]:
            raise ValueError(f"Invalid TTS engine: {engine}. Must be 'edge' or 'tiny'")

        self.tts_engine = engine
        logger.info(f"TTS engine switched to: {engine}")


# Global instance
tts_service = TTSService()
