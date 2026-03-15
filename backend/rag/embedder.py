import os
from sentence_transformers import SentenceTransformer
import numpy as np

# ── Load model once at import time ────────────────────────────────────────
# Downloads ~80MB on first run, then cached locally forever
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model ready ✅")


def embed_text(text: str) -> list:
    """Embed a single string — returns 384-dimensional vector"""
    vector = embedding_model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: list) -> list:
    """
    Embed a list of strings efficiently in one batch.
    Much faster than calling embed_text() in a loop.
    Returns list of 384-dimensional vectors.
    """
    vectors = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64
    )
    return vectors.tolist()