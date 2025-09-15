from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np

import fasttext
import fasttext.util


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:  # Convert list of strings into an array of vectors.
        pass


class FastTextEmbedder(EmbeddingModel):
    """FastText embedder (English pre-trained)."""

    def __init__(self, dim: int = 300):
        """
        Load English FastText vectors.
        Default: 300 dimensions (cc.en.300.bin).
        """
        fasttext.util.download_model('en', if_exists='ignore')  # downloads if not present
        self.model = fasttext.load_model('cc.en.300.bin')

        # If user requests smaller dimension, reduce model
        if dim != 300:
            fasttext.util.reduce_model(self.model, dim)

        self.dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalized embeddings for a list of texts."""
        vecs = np.vstack([self.model.get_sentence_vector(str(t)) for t in texts]).astype("float32")

        # L2 normalization (important for cosine distance)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs / norms
