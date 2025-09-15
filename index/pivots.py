from __future__ import annotations
import numpy as np
from typing import Tuple
from sklearn.decomposition import PCA


class PivotSelector:
    """
    Selects pivots from a matrix of embeddings using PCA.
    """

    def __init__(self, k: int = 3, seed: int = 42):
        self.k = k
        self.seed = seed
        self.pivots: np.ndarray | None = None

    def fit(self, vectors: np.ndarray) -> np.ndarray:
        """
        Select k pivots using PCA principal components.

        Args:
            vectors: (n, d) matrix of embeddings.
        Returns:
            pivots: (k, d) matrix of selected pivots (principal directions).
        """
        n, d = vectors.shape
        n_components = min(self.k, d)

        pca = PCA(n_components=n_components, random_state=self.seed)
        pca.fit(vectors)

        # pivots are the top principal component directions
        self.pivots = pca.components_
        return self.pivots

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """
        Map vectors into pivot space (distances to pivots).

        Args:
            vectors: (n, d) input vectors.
        Returns:
            distances: (n, k) distances to each pivot.
        """
        if self.pivots is None:
            raise RuntimeError("PivotSelector has not been fitted yet.")

        # compute L2 distances from each vector to each pivot direction
        dists = np.linalg.norm(vectors[:, None, :] - self.pivots[None, :, :], axis=2)
        return dists

    def fit_transform(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit pivots and return both pivots and transformed distances.
        """
        pivots = self.fit(vectors)
        dists = self.transform(vectors)
        return pivots, dists
