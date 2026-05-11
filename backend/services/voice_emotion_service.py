import asyncio
import os
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer, AutoFeatureExtractor
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForAudioClassification

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️ librosa not available - WebM audio may not work properly")

class VoiceEmotionAnalyzer:
    """Voice emotion analysis using parallel STT + Text Emotion + Audio Emotion models."""
    
    def __init__(self):
        print("⚡ Loading Voice Emotion Models...")
        load_start = time.time()
        
        # Define paths - use absolute paths and convert to strings with forward slashes
        base_dir = Path(__file__).resolve().parent.parent.parent
        base_path = base_dir / "onnx_models"
        whisper_root = str(base_path / "whisper_tiny").replace("\\", "/")
        text_path = str(base_path / "text_emotion").replace("\\", "/")
        audio_path = str(base_path / "audio_emotion").replace("\\", "/")

        # Load Whisper STT
        self.stt_model = WhisperModel(
            "tiny.en", 
            device="cpu", 
            compute_type="int8", 
            download_root=whisper_root, 
            local_files_only=True
        )
        
        # Load Text Emotion Model
        self.text_pipe = pipeline(
            "sentiment-analysis", 
            model=ORTModelForSequenceClassification.from_pretrained(text_path), 
            tokenizer=AutoTokenizer.from_pretrained(text_path)
        )
        
        # Load Audio Emotion Model
        self.audio_pipe = pipeline(
            "audio-classification", 
            model=ORTModelForAudioClassification.from_pretrained(audio_path), 
            feature_extractor=AutoFeatureExtractor.from_pretrained(audio_path)
        )
        
        self.load_time = time.time() - load_start
        print(f"✅ Voice Emotion Models Loaded in {self.load_time:.2f}s")

    def _load_audio(self, audio_path: str):
        """Load and preprocess audio file for model input with WebM support."""
        t_read_start = time.time()

        # Check if file is WebM format
        is_webm = audio_path.lower().endswith('.webm')

        try:
            # For WebM files, use librosa if available (better codec support)
            if is_webm and LIBROSA_AVAILABLE:
                print(f"   🎵 Detected WebM format, using librosa for better compatibility")
                data, samplerate = librosa.load(audio_path, sr=None, mono=False)
                t_read_end = time.time()

                # Convert to float32
                data = data.astype(np.float32)

                # Convert stereo to mono if needed
                if len(data.shape) > 1:
                    data = data.mean(axis=0)

                # Resample to 16kHz if needed
                t_resample_start = time.time()
                if samplerate != 16000:
                    data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
                    t_resample_end = time.time()
                    print(f"   🔄 Resampled {samplerate}Hz → 16000Hz in {(t_resample_end - t_resample_start)*1000:.2f}ms")
                else:
                    t_resample_end = t_resample_start
                    print(f"   ✅ Audio already at 16000Hz, skipping resample")

                print(f"   📁 Audio file read in {(t_read_end - t_read_start)*1000:.2f}ms")
                return data, 16000

            # For WAV/other formats, use soundfile (faster)
            data, samplerate = sf.read(audio_path)
            t_read_end = time.time()

            data = data.astype(np.float32)

            # Convert stereo to mono
            if len(data.shape) > 1:
                data = data.mean(axis=1)

            # Resample to 16kHz if needed (using scipy for 2-3x faster performance)
            t_resample_start = time.time()
            if samplerate != 16000:
                # Calculate resampling ratio
                gcd = np.gcd(samplerate, 16000)
                up = 16000 // gcd
                down = samplerate // gcd
                data = signal.resample_poly(data, up, down)
                t_resample_end = time.time()
                print(f"   🔄 Resampled {samplerate}Hz → 16000Hz in {(t_resample_end - t_resample_start)*1000:.2f}ms")
            else:
                t_resample_end = t_resample_start
                print(f"   ✅ Audio already at 16000Hz, skipping resample")

            print(f"   📁 Audio file read in {(t_read_end - t_read_start)*1000:.2f}ms")
            return data, 16000

        except Exception as e:
            # Final fallback: try librosa for any format
            if LIBROSA_AVAILABLE:
                print(f"   ⚠️ Primary load failed, using librosa fallback: {e}")
                data, sr = librosa.load(audio_path, sr=16000, mono=True)
                print(f"   🔄 Fallback loaded and resampled to 16000Hz")
                return data, 16000
            else:
                # Last resort: soundfile + scipy
                print(f"   ⚠️ Primary load failed, using soundfile fallback: {e}")
                data, sr = sf.read(audio_path)
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                if sr != 16000:
                    gcd = np.gcd(sr, 16000)
                    up = 16000 // gcd
                    down = sr // gcd
                    data = signal.resample_poly(data, up, down)
                    print(f"   🔄 Fallback resampled {sr}Hz → 16000Hz")
                return data, 16000

    def _transcribe(self, audio_path: str) -> str:
        """Transcribe audio to text using Whisper."""
        segments, _ = self.stt_model.transcribe(audio_path, beam_size=1)
        return " ".join([s.text for s in segments])

    def _analyze_text_emotion(self, text: str) -> dict:
        """Analyze emotion from transcribed text."""
        if not text.strip():
            return {"label": "neutral", "score": 0.0}
        return self.text_pipe(text)[0]

    def _analyze_audio_emotion(self, audio_data) -> list:
        """Analyze emotion from audio tone."""
        # Returns list: [{'score': 0.913..., 'label': 'sad'}, ...]
        results = self.audio_pipe(audio_data)
        return results

    async def analyze(self, audio_path: str) -> dict:
        """Main analysis function - processes audio file and returns emotion results."""
        try:
            # Load audio
            audio_data, _ = self._load_audio(audio_path)
            
            # Run STT and audio emotion in parallel
            stt_task = asyncio.to_thread(self._transcribe, audio_path)
            audio_task = asyncio.to_thread(self._analyze_audio_emotion, audio_data)
            
            text, audio_emotions = await asyncio.gather(stt_task, audio_task)
            
            # Analyze text emotion
            text_emotion = self._analyze_text_emotion(text)
            
            # Get top audio emotion
            top_audio = audio_emotions[0] if audio_emotions else {"label": "neutral", "score": 0.0}
            
            return {
                "success": True,
                "transcript": text,
                "text_emotion": {
                    "label": text_emotion["label"],
                    "confidence": round(text_emotion["score"], 2)
                },
                "voice_emotion": {
                    "label": top_audio["label"],
                    "confidence": round(top_audio["score"], 2)
                },
                "voice_emotion_all": [
                    {"label": e["label"], "confidence": round(e["score"], 2)}
                    for e in audio_emotions
                ]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# Global instance (initialized on first use)
_analyzer = None

def get_analyzer():
    """Get or create the global analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = VoiceEmotionAnalyzer()
    return _analyzer

async def analyze_voice_emotion(audio_path: str) -> dict:
    """Public API for voice emotion analysis."""
    analyzer = get_analyzer()
    return await analyzer.analyze(audio_path)
