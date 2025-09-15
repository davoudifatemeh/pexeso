from __future__ import annotations
import numpy as np
from typing import List, Tuple


class HierarchicalGrid:
    """
    Hierarchical grid partitioning of pivot-space distances.
    Each vector is assigned a sequence of cell IDs across grid levels.
    """

    def __init__(self, levels: int = 2):
        """
        Args:
            levels: number of levels in the hierarchy (>=1).
        """
        self.levels = levels
        self.bin_edges: List[np.ndarray] | None = None

    def fit(self, distances: np.ndarray):
        """
        Fit grid bin edges based on observed distances.
        
        Args:
            distances: (n, k) matrix of pivot distances.
        """
        n, k = distances.shape
        self.bin_edges = []

        for dim in range(k):
            col = distances[:, dim]
            edges_per_level = []
            # Create increasingly finer partitions at each level
            for l in range(1, self.levels + 1):
                bins = np.linspace(col.min(), col.max(), 2**l + 1)  # uniform bins
                edges_per_level.append(bins)
            self.bin_edges.append(edges_per_level)

    def transform(self, distances: np.ndarray) -> List[List[Tuple[int, ...]]]:
        """
        Assign each row to hierarchical grid cells.

        Args:
            distances: (n, k) matrix of pivot distances.
        Returns:
            cells: list of length n, each entry is a list of cell IDs across levels.
                   Each cell ID is a tuple of bin indices (per pivot).
        """
        if self.bin_edges is None:
            raise RuntimeError("Grid has not been fitted yet.")

        n, k = distances.shape
        all_cells: List[List[Tuple[int, ...]]] = []

        for i in range(n):
            row = distances[i]
            row_cells = []
            # At each level, assign bin indices
            for l in range(self.levels):
                bin_indices = []
                for dim in range(k):
                    edges = self.bin_edges[dim][l]
                    idx = np.digitize(row[dim], edges) - 1  # bin index
                    bin_indices.append(idx)
                row_cells.append(tuple(bin_indices))
            all_cells.append(row_cells)

        return all_cells
