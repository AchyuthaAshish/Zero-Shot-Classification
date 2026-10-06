"""Pretrained multilingual Sentence Transformer encoder wrapper.

Loads 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' locally.
Supports CPU and GPU inference without external API calls.
"""

from typing import List, Optional
import numpy as np

from ml.config import PRETRAINED_MODEL_NAME

_MODEL_INSTANCE = None


def get_embedding_model(model_name: str = PRETRAINED_MODEL_NAME):
    """
    Returns cached SentenceTransformer instance.
    Uses Hugging Face local cache; works offline once downloaded.
    CPU-compatible by default.
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL_INSTANCE = SentenceTransformer(model_name, device=device)
    return _MODEL_INSTANCE


def encode_texts(texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
    """Encodes a list of texts into dense embedding vectors."""
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings
