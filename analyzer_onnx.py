import asyncio
import os
import json
import librosa
import soundfile as sf
import warnings
import logging
import time
import numpy as np
from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer, AutoFeatureExtractor
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForAudioClassification

# --- CONFIGURATION ---
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
# logging.getLogger("transformers").setLevel(logging.ERROR)

class FastVoiceSystem:
    def __init__(self):
        print("⚡ Loading Models from LOCAL ./onnx_models...")
        load_start = time.time()
        
        # DEFINING PATHS
        base_path = "./onnx_models"
        whisper_root = os.path.join(base_path, "whisper_tiny")
        text_path = os.path.join(base_path, "text_emotion")
        audio_path = os.path.join(base_path, "audio_emotion")

        # 1. LOAD WHISPER
        self.stt_model = WhisperModel("tiny.en", device="cpu", compute_type="int8", download_root=whisper_root, local_files_only=True)
        
        # 2. LOAD TEXT MODEL
        self.text_pipe = pipeline(
            "sentiment-analysis", 
            model=ORTModelForSequenceClassification.from_pretrained(text_path), 
            tokenizer=AutoTokenizer.from_pretrained(text_path)
        )
        
        # 3. LOAD AUDIO MODEL
        self.audio_pipe = pipeline(
            "audio-classification", 
            model=ORTModelForAudioClassification.from_pretrained(audio_path), 
            feature_extractor=AutoFeatureExtractor.from_pretrained(audio_path)
        )
        
        self.load_time = time.time() - load_start
        print(f"✅ Models Loaded in {self.load_time:.2f} seconds")

    def _run_stt_task(self, audio_path):
        t_start = time.time()
        segments, _ = self.stt_model.transcribe(audio_path, beam_size=1)
        text = " ".join([s.text for s in segments])
        t_end = time.time()
        return text, t_start, t_end

    def _run_audio_task(self, audio_data):
        t_start = time.time()
        res = self.audio_pipe(audio_data)
        t_end = time.time()
        return res, t_start, t_end

    def _fast_load_audio(self, audio_path):
        """Fixed: Resamples in memory to avoid double-loading."""
        try:
            # 1. Fast Read
            data, samplerate = sf.read(audio_path)
            
            # 2. Convert to float32 (Standard for AI)
            data = data.astype(np.float32)
            
            # 3. Convert Stereo to Mono if needed
            if len(data.shape) > 1:
                data = data.mean(axis=1)
                
            # 4. Resample In-Memory (Don't reload file!)
            if samplerate != 16000:
                # This is much faster than reloading the file
                data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
                
            return data, 16000
        except Exception as e:
            # Only use slow fallback if soundfile fails completely
            # print(f"DEBUG: Fast load failed ({e}), using librosa...")
            return librosa.load(audio_path, sr=16000, mono=True)

    async def process(self, audio_path):
        total_start = time.time()

        # --- 1. AUDIO LOADING ---
        t_load_start = time.time()
        try:
            audio, sr = self._fast_load_audio(audio_path)
        except Exception as e:
            return {"error": f"Audio Load Error: {e}"}
        t_load_end = time.time()

        # --- 2. PARALLEL EXECUTION ---
        stt_task = asyncio.to_thread(self._run_stt_task, audio_path)
        audio_task = asyncio.to_thread(self._run_audio_task, audio)
        
        (text_raw, stt_start, stt_end), (audio_res_raw, audio_start, audio_end) = await asyncio.gather(stt_task, audio_task)

        # --- 3. TEXT ANALYSIS ---
        sent_start = time.time()
        text_res = self.text_pipe(text_raw)[0] if text_raw.strip() else {"label": "neutral", "score": 0.0}
        sent_end = time.time()
        
        top_audio = audio_res_raw[0]
        total_end = time.time()

        # --- LOGGING ---
        print("\n" + "="*60)
        print("📊 MODEL RESPONSES & TIMELINE")
        print("="*60)
        print(f"\n🔤 TEXT EMOTION MODEL (Full Response):")
        print(f"   Raw Output: {text_res}")
        print(f"\n🎵 AUDIO EMOTION MODEL (Full Response):")
        print(f"   Raw Output: {audio_res_raw}")
        print(f"   Top Result: {top_audio}")
        print(f"\n⏱️  TIMELINE:")
        print(f"   Audio Loading:    {round((t_load_end - t_load_start) * 1000, 2)} ms")
        print(f"   STT (Whisper):    {round((stt_end - stt_start) * 1000, 2)} ms")
        print(f"   Audio Emotion:    {round((audio_end - audio_start) * 1000, 2)} ms")
        print(f"   Text Emotion:     {round((sent_end - sent_start) * 1000, 2)} ms")
        print(f"   Total Processing: {round((total_end - total_start) * 1000, 2)} ms")
        print(f"   Parallel Check:   {'✅ Success' if abs(stt_start - audio_start) < 0.1 else '❌ Sequential'}")
        print("="*60 + "\n")

        return {
            "result": {
                "text": text_raw,
                "text_emotion": text_res['label'],
                "text_confidence": round(text_res['score'], 2),
                "voice_emotion": top_audio['label'],
                "voice_confidence": round(top_audio['score'], 2)
            },
            "stats": {
                "loading_time_sec": round(self.load_time, 2),
                "total_processing_ms": round((total_end - total_start) * 1000, 2),
                "breakdown_ms": {
                    "audio_loading": round((t_load_end - t_load_start) * 1000, 2),
                    "stt_whisper": round((stt_end - stt_start) * 1000, 2),
                    "audio_tone": round((audio_end - audio_start) * 1000, 2),
                    "text_sentiment": round((sent_end - sent_start) * 1000, 2),
                },
                "timeline": {
                    "parallel_check": "✅ Success" if abs(stt_start - audio_start) < 0.1 else "❌ Sequential"
                }
            }
        }

if __name__ == "__main__":
    system = FastVoiceSystem()
    if os.path.exists("test.wav"):
        print("\nrunning pipeline...")
        res = asyncio.run(system.process("test.wav"))
        print(json.dumps(res, indent=2))
    else:
        print("❌ File 'test.wav' not found.")