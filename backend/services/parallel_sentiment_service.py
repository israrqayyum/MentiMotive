import asyncio
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer, AutoFeatureExtractor
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForAudioClassification
import logging

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not available - WebM audio may not work properly")

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
if not FFMPEG_AVAILABLE:
    logging.warning("ffmpeg not found in PATH — AMR/3GP/M4A audio will fail. Install with: sudo apt install ffmpeg")

class ParallelSentimentAnalyzer:
    """Parallel sentiment analysis: STT->Text Emotion + Audio Emotion (both run in parallel)."""
    
    def __init__(self):
        print("⚡ Loading Parallel Sentiment Models...")
        load_start = time.time()
        
        # Define paths - use absolute paths
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
        print(f"✅ Parallel Sentiment Models Loaded in {self.load_time:.2f}s")

    def _sniff_format(self, audio_path: str) -> str:
        """Read magic bytes to detect the real audio container format."""
        try:
            with open(audio_path, "rb") as f:
                header = f.read(12)
            if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
                return "wav"
            if header[:4] in (b"fLaC",):
                return "flac"
            if header[:3] == b"ID3" or header[:2] == b"\xff\xfb":
                return "mp3"
            if header[:4] == b"OggS":
                return "ogg"
            if header[4:8] in (b"ftyp", b"moov"):
                return "m4a"
            # 3GP / AMR-NB magic
            if header[4:8] == b"ftyp" or header[:6] in (b"#!AMR\n", b"#!AMR-"):
                return "amr"
            if header[:4] in (b"\x1a\x45\xdf\xa3",):
                return "webm"
            return "unknown"
        except Exception:
            return "unknown"

    def _convert_with_ffmpeg(self, audio_path: str) -> str:
        """
        Convert any audio file to 16kHz mono WAV using ffmpeg.
        Returns path to the converted file (caller must clean it up).
        Raises RuntimeError if ffmpeg is unavailable or conversion fails.
        """
        if not FFMPEG_AVAILABLE:
            raise RuntimeError(
                "ffmpeg is not installed. Run: sudo apt install ffmpeg"
            )
        converted_path = audio_path + "_ffmpeg.wav"
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            "-acodec", "pcm_s16le",
            converted_path,
        ]
        logging.info(f"   🔧 Running ffmpeg conversion: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg conversion failed (exit {result.returncode}):\n{result.stderr.strip()}"
            )
        size = os.path.getsize(converted_path)
        logging.info(f"   ✅ ffmpeg produced {size} bytes at {converted_path}")
        return converted_path

    def _fast_load_audio(self, audio_path: str):
        """
        Load and preprocess audio file.
        Strategy:
          1. Sniff real format from magic bytes and log it.
          2. If format is not natively readable (AMR/3GP/M4A/WebM/unknown),
             convert to PCM WAV with ffmpeg first.
          3. Try soundfile (fast path for true WAV/FLAC/OGG).
          4. Fallback to librosa if soundfile still fails.
        """
        t_read_start = time.time()
        detected_fmt = self._sniff_format(audio_path)
        logging.info(f"   🔍 Detected audio format from magic bytes: {detected_fmt} (path: {audio_path})")

        # Formats that soundfile cannot handle natively → pre-convert with ffmpeg
        non_native_formats = {"amr", "m4a", "mp3", "webm", "unknown"}
        ffmpeg_converted_path = None

        try:
            if detected_fmt in non_native_formats:
                logging.info(f"   ⚠️  Format '{detected_fmt}' not natively supported by soundfile — converting via ffmpeg")
                ffmpeg_converted_path = self._convert_with_ffmpeg(audio_path)
                load_path = ffmpeg_converted_path
            else:
                load_path = audio_path

            try:
                data, samplerate = sf.read(load_path)
                t_read_end = time.time()
                logging.info(f"   📁 soundfile read OK in {(t_read_end - t_read_start)*1000:.2f}ms")
            except Exception as sf_err:
                logging.warning(
                    f"   ⚠️ soundfile failed ({type(sf_err).__name__}: {sf_err}) — trying ffmpeg conversion"
                )
                # soundfile failed even on an apparently native format; force-convert
                if ffmpeg_converted_path is None:
                    ffmpeg_converted_path = self._convert_with_ffmpeg(audio_path)
                    load_path = ffmpeg_converted_path
                if LIBROSA_AVAILABLE:
                    logging.info(f"   🔄 Attempting librosa load after ffmpeg conversion")
                    data, samplerate = librosa.load(load_path, sr=None, mono=False)
                    t_read_end = time.time()
                    logging.info(f"   📁 librosa read OK in {(t_read_end - t_read_start)*1000:.2f}ms")
                else:
                    data, samplerate = sf.read(load_path)
                    t_read_end = time.time()

            data = data.astype(np.float32)

            # Stereo → mono
            if len(data.shape) > 1:
                data = data.mean(axis=-1)

            # Resample to 16 kHz
            t_resample_start = time.time()
            if samplerate != 16000:
                if LIBROSA_AVAILABLE:
                    data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
                else:
                    gcd = np.gcd(int(samplerate), 16000)
                    data = signal.resample_poly(data, 16000 // gcd, samplerate // gcd)
                t_resample_end = time.time()
                logging.info(f"   🔄 Resampled {samplerate}Hz → 16000Hz in {(t_resample_end - t_resample_start)*1000:.2f}ms")
            else:
                logging.info(f"   ✅ Audio already at 16000Hz, skipping resample")

            return data, 16000

        finally:
            if ffmpeg_converted_path and os.path.exists(ffmpeg_converted_path):
                try:
                    os.unlink(ffmpeg_converted_path)
                except Exception:
                    pass

    def _run_stt_task(self, audio_path: str):
        """Run STT transcription and return timing."""
        t_start = time.time()
        segments, _ = self.stt_model.transcribe(audio_path, beam_size=1)
        text = " ".join([s.text for s in segments])
        t_end = time.time()
        return text, t_start, t_end

    def _run_audio_emotion_task(self, audio_data):
        """Run audio emotion analysis and return timing."""
        t_start = time.time()
        results = self.audio_pipe(audio_data)
        t_end = time.time()
        return results, t_start, t_end

    def _run_text_emotion_task(self, text: str):
        """Run text emotion analysis and return timing."""
        t_start = time.time()
        if not text.strip():
            result = {"label": "neutral", "score": 0.0}
        else:
            result = self.text_pipe(text)[0]
        t_end = time.time()
        return result, t_start, t_end

    async def analyze_parallel(self, audio_path: str) -> dict:
        """
        Main analysis function with parallel processing:
        Path 1: Audio -> STT -> Text Emotion
        Path 2: Audio -> Audio Emotion
        Both paths run in parallel.
        """
        total_start = time.time()
        
        try:
            # Load audio
            t_load_start = time.time()
            audio_data, _ = self._fast_load_audio(audio_path)
            t_load_end = time.time()
            
            # === PARALLEL EXECUTION ===
            # Path 1: STT + Text Emotion (sequential within this path)
            # Path 2: Audio Emotion
            # Both paths run in parallel
            
            stt_task = asyncio.to_thread(self._run_stt_task, audio_path)
            audio_emotion_task = asyncio.to_thread(self._run_audio_emotion_task, audio_data)
            
            # Wait for both parallel tasks
            (text, stt_start, stt_end), (audio_emotions, audio_start, audio_end) = await asyncio.gather(
                stt_task,
                audio_emotion_task
            )
            
            # Now run text emotion analysis (depends on STT result)
            text_emotion, text_start, text_end = self._run_text_emotion_task(text)
            
            # Get top audio emotion
            top_audio = audio_emotions[0] if audio_emotions else {"label": "neutral", "score": 0.0}
            
            total_end = time.time()
            
            # === TIMELINE LOGGING ===
            logging.info("\n" + "="*60)
            logging.info("📊 PARALLEL SENTIMENT ANALYSIS - TIMELINE")
            logging.info("="*60)
            logging.info(f"\n📝 PATH 1: Audio -> STT -> Text Emotion")
            logging.info(f"   Transcript: {text[:100]}...")
            logging.info(f"   Text Emotion: {text_emotion}")
            logging.info(f"\n🎵 PATH 2: Audio -> Audio Emotion")
            logging.info(f"   Audio Emotions: {audio_emotions}")
            logging.info(f"   Top Audio Emotion: {top_audio}")
            logging.info(f"\n⏱️  DETAILED TIMELINE:")
            logging.info(f"   Audio Loading:      {round((t_load_end - t_load_start) * 1000, 2)} ms")
            logging.info(f"   STT (Whisper):      {round((stt_end - stt_start) * 1000, 2)} ms")
            logging.info(f"   Audio Emotion:      {round((audio_end - audio_start) * 1000, 2)} ms")
            logging.info(f"   Text Emotion:       {round((text_end - text_start) * 1000, 2)} ms")
            logging.info(f"   Total Processing:   {round((total_end - total_start) * 1000, 2)} ms")
            logging.info(f"   Parallel Check:     {'✅ Success' if abs(stt_start - audio_start) < 0.1 else '❌ Sequential'}")
            logging.info("="*60 + "\n")
            
            return {
                "success": True,
                "transcript": text,
                # Path 1: STT -> Text Emotion
                "path1_text_emotion": {
                    "label": text_emotion["label"],
                    "confidence": round(text_emotion["score"], 2)
                },
                # Path 2: Audio Emotion
                "path2_audio_emotion": {
                    "label": top_audio["label"],
                    "confidence": round(top_audio["score"], 2)
                },
                "path2_audio_emotion_all": [
                    {"label": e["label"], "confidence": round(e["score"], 2)}
                    for e in audio_emotions
                ],
                # Timing stats
                "stats": {
                    "total_processing_ms": round((total_end - total_start) * 1000, 2),
                    "breakdown_ms": {
                        "audio_loading": round((t_load_end - t_load_start) * 1000, 2),
                        "stt_whisper": round((stt_end - stt_start) * 1000, 2),
                        "audio_emotion": round((audio_end - audio_start) * 1000, 2),
                        "text_emotion": round((text_end - text_start) * 1000, 2)
                    },
                    "parallel_check": "✅ Success" if abs(stt_start - audio_start) < 0.1 else "❌ Sequential"
                }
            }
            
        except Exception as e:
            tb = traceback.format_exc()
            logging.error(
                f"❌ Error in parallel sentiment analysis: {type(e).__name__}: {e}\n{tb}"
            )
            return {
                "success": False,
                "error": f"{type(e).__name__}: {e}" or tb.strip().splitlines()[-1],
            }

# Global instance
_analyzer = None

def get_analyzer():
    """Get or create the global analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = ParallelSentimentAnalyzer()
    return _analyzer

async def analyze_parallel_sentiment(audio_path: str) -> dict:
    """Public API for parallel sentiment analysis."""
    analyzer = get_analyzer()
    return await analyzer.analyze_parallel(audio_path)
