"""
Test script for ONNX embeddings service.
Tests embedding generation and similarity calculation.

Run from project root:
    python backend/test_embeddings.py
"""

import sys
import os
import time
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from services.onnx_embeddings_service import get_onnx_embeddings


def cosine_similarity(vec1, vec2):
    # Since your service already normalizes, this is equivalent
    return np.dot(vec1, vec2)


def test_embeddings():
    """Test ONNX embeddings with sample sentences."""
    
    # Start total time tracking
    total_start = time.time()
    
    print("=" * 70)
    print("🧪 Testing ONNX Embeddings Service")
    print("=" * 70)
    print()
    
    # Test sentences
    sentences = [
        "I'm feeling anxious about my upcoming presentation",
        "I have severe anxiety about public speaking",
        "The weather is nice today",
        "I can't sleep at night due to stress",
        "Insomnia is affecting my daily life"
    ]
    
    print("📝 Test Sentences:")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent}")
    print()
    
    # Load embeddings model
    print("🔄 Loading ONNX embeddings model...")
    load_start = time.time()
    try:
        embeddings_service = get_onnx_embeddings()
        load_time = time.time() - load_start
        print(f"✅ Model loaded successfully!")
        print(f"⏱️  Load time: {load_time:.3f}s")
        print()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Generate embeddings
    print("🔄 Generating embeddings...")
    embed_start = time.time()
    try:
        # Use embed_documents for batch processing
        embeddings = embeddings_service.embed_documents(sentences)
        embed_time = time.time() - embed_start
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"📊 Embedding dimension: {len(embeddings[0])}")
        print(f"⏱️  Embedding time: {embed_time:.3f}s ({embed_time/len(sentences):.3f}s per sentence)")
        print()
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Calculate similarities
    print("🔍 Similarity Analysis:")
    print("-" * 70)
    
    compare_start = time.time()
    
    # Compare related sentences
    comparisons = [
        (0, 1, "Anxiety sentences (should be HIGH)"),
        (0, 2, "Anxiety vs Weather (should be LOW)"),
        (3, 4, "Sleep/Insomnia sentences (should be HIGH)"),
        (0, 3, "Anxiety vs Sleep (moderate)"),
        (1, 2, "Anxiety vs Weather (should be LOW)")
    ]
    
    for idx1, idx2, description in comparisons:
        sim = cosine_similarity(embeddings[idx1], embeddings[idx2])
        
        # Color coding for similarity
        if sim > 0.7:
            indicator = "🟢 HIGH"
        elif sim > 0.4:
            indicator = "🟡 MEDIUM"
        else:
            indicator = "🔴 LOW"
        
        print(f"\n{description}")
        print(f"  Sentence {idx1+1}: '{sentences[idx1][:50]}...'")
        print(f"  Sentence {idx2+1}: '{sentences[idx2][:50]}...'")
        print(f"  Similarity: {sim:.4f} {indicator}")
    
    compare_time = time.time() - compare_start
    
    print()
    print("-" * 70)
    print(f"⏱️  Comparison time: {compare_time:.3f}s ({compare_time/len(comparisons):.4f}s per comparison)")
    
    # Test single query embedding
    print("\n🔍 Testing single query embedding...")
    query = "I need help with my mental health"
    query_start = time.time()
    query_embedding = embeddings_service.embed_query(query)
    query_time = time.time() - query_start
    print(f"✅ Query: '{query}'")
    print(f"📊 Embedding dimension: {len(query_embedding)}")
    print(f"⏱️  Query embedding time: {query_time:.3f}s")
    
    # Find most similar sentence
    print("\n🎯 Finding most similar sentence to query:")
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    max_idx = np.argmax(similarities)
    
    print(f"  Most similar: Sentence {max_idx+1}")
    print(f"  '{sentences[max_idx]}'")
    print(f"  Similarity: {similarities[max_idx]:.4f}")
    
    # Calculate total time
    total_time = time.time() - total_start
    
    print()
    print("=" * 70)
    print("✅ All tests completed successfully!")
    print()
    print("📊 Performance Summary:")
    print(f"  • Model loading:     {load_time:.3f}s")
    print(f"  • Batch embedding:   {embed_time:.3f}s ({embed_time/len(sentences):.3f}s per sentence)")
    print(f"  • Single query:      {query_time:.3f}s")
    print(f"  • Comparisons:       {compare_time:.3f}s")
    print(f"  • Total time:        {total_time:.3f}s")
    print("=" * 70)


if __name__ == "__main__":
    test_embeddings()

if __name__ == "__main__":
    test_embeddings()
