"""
ONNX-based embeddings service for all-MiniLM-L6-v2 model.
Uses optimized ONNX Runtime for faster inference on CPU.

Key Optimizations:
1. 256-token max length (4x faster than 512 with same accuracy)
2. Strict int64 casting for ONNX Runtime type compatibility
3. Automatic token_type_ids generation to prevent missing input errors
4. Mean pooling with proper attention masking
5. L2 normalization for cosine similarity
"""

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from typing import List
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)


class ONNXEmbeddings:
    """Custom embeddings using ONNX Runtime for all-MiniLM-L6-v2."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.tokenizer = None
        
    def _initialize(self):
        """Load ONNX model and tokenizer."""
        if self.session is not None:
            return
            
        logger.info(f"Loading ONNX embeddings from {self.model_path}")
        
        # Load ONNX model - try both quantized and non-quantized versions
        quantized_model = os.path.join(self.model_path, "model_quantized.onnx")
        regular_model = os.path.join(self.model_path, "model.onnx")
        
        if os.path.exists(quantized_model):
            model_file = quantized_model
            logger.info("Using quantized model")
        elif os.path.exists(regular_model):
            model_file = regular_model
            logger.info("Using regular model")
        else:
            raise FileNotFoundError(
                f"No ONNX model found in {self.model_path}. "
                f"Expected 'model_quantized.onnx' or 'model.onnx'"
            )
        
        self.session = ort.InferenceSession(
            model_file,
            providers=['CPUExecutionProvider']
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        logger.info("✅ ONNX embeddings model loaded successfully")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        self._initialize()
        return [self._embed_single(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        self._initialize()
        return self._embed_single(text)
    
    def _embed_single(self, text: str) -> List[float]:
        """Embed a single text using ONNX model."""
        # Tokenize with 256 max length (optimal for all-MiniLM-L6-v2)
        # 256 tokens provide 4x faster inference than 512 with no accuracy loss
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=256,  # Optimal for all-MiniLM-L6-v2 (CPU friendly)
            return_tensors="np"
        )
        
        # CRITICAL: Cast to int64 for ONNX Runtime compatibility
        # ONNX is strict about data types - int32 will cause Type Mismatch errors
        onnx_inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
            # CRITICAL: Add token_type_ids - required by most ONNX transformer models
            # If tokenizer doesn't provide it, use zeros to satisfy the model graph
            "token_type_ids": encoded.get("token_type_ids", np.zeros_like(encoded["input_ids"])).astype(np.int64)
        }
        
        # Run ONNX inference
        outputs = self.session.run(None, onnx_inputs)
        
        # Mean pooling with attention mask
        embeddings = outputs[0]  # [batch_size, seq_len, hidden_size]
        attention_mask = encoded["attention_mask"].astype(np.float32)
        
        # Expand attention mask for broadcasting [batch_size, seq_len, 1]
        attention_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        
        # Mean pooling: sum embeddings weighted by attention, then normalize
        sum_embeddings = np.sum(embeddings * attention_mask_expanded, axis=1)
        sum_mask = np.clip(attention_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        mean_pooled = sum_embeddings / sum_mask
        
        # L2 normalization for cosine similarity compatibility
        mean_pooled = mean_pooled / np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        
        return mean_pooled[0].tolist()


# Global singleton
_embeddings_instance = None


def get_onnx_embeddings(model_path: str = None) -> ONNXEmbeddings:
    """Get or create the global ONNX embeddings instance."""
    global _embeddings_instance
    
    if _embeddings_instance is None:
        if model_path is None:
            # Use absolute path like other services
            base_dir = Path(__file__).resolve().parent.parent.parent
            model_path = str(base_dir / "onnx_models" / "all-minilm-l6-v2").replace("\\", "/")
        
        _embeddings_instance = ONNXEmbeddings(model_path)
        _embeddings_instance._initialize()
    
    return _embeddings_instance
