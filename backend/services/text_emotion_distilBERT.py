import os
import time
from pathlib import Path
from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)

class TextEmotionAnalyzer:
    """Local text emotion classification using DistilBERT model."""
    
    def __init__(self):
        print("⚡ Loading Text Emotion Model...")
        load_start = time.time()
        
        # Define path - use absolute path and convert to string with forward slashes
        base_dir = Path(__file__).resolve().parent.parent.parent
        text_model_path = str(base_dir / "onnx_models" / "text_emotion").replace("\\", "/")
        
        # Load Text Emotion Model
        self.text_pipe = pipeline(
            "sentiment-analysis", 
            model=ORTModelForSequenceClassification.from_pretrained(text_model_path), 
            tokenizer=AutoTokenizer.from_pretrained(text_model_path)
        )
        
        load_time = time.time() - load_start
        print(f"✅ Text Emotion Model Loaded in {load_time:.2f}s")
    
    def classify(self, text: str) -> dict:
        """Classify emotion from text."""
        t_start = time.time()
        
        if not text.strip():
            return {"label": "neutral", "score": 0.0}
        
        # Model returns: {'label': 'joy', 'score': 0.998...}
        result = self.text_pipe(text)[0]
        
        t_end = time.time()
        logger.info(f"   📝 Text Emotion Analysis: {round((t_end - t_start) * 1000, 2)} ms - Result: {result['label']} ({result['score']:.2f})")
        
        return {
            "label": result["label"],
            "score": round(result["score"], 2)
        }

# Global instance (initialized on first use)
_analyzer = None

def get_analyzer():
    """Get or create the global analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = TextEmotionAnalyzer()
    return _analyzer

def classify_text_emotion(text: str) -> dict:
    """Public API for text emotion classification."""
    analyzer = get_analyzer()
    return analyzer.classify(text)
