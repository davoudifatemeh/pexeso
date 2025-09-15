# pexeso/index/index.py
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

from index.pivots import PivotSelector
from index.grid import HierarchicalGrid
from index.inverted_index import InvertedIndex
from utils.types import TableId, ColumnId


class PEXESOIndex:
    """
    Main index structure that glues:
    - PivotSelector (PCA pivots)
    - HierarchicalGrid (multi-resolution blocking)
    - InvertedIndex (cell → postings)
    """

    def __init__(self, pivots_k: int = 5, grid_levels: int = 3):
        self.selector = PivotSelector(k=pivots_k)
        self.grid = HierarchicalGrid(levels=grid_levels)
        self.inv_index = InvertedIndex()
        self.fitted = False

    def fit(self, embeddings: np.ndarray):
        """Fit pivots and hierarchical grid globally."""
        pivots = self.selector.fit(embeddings)
        self.grid.fit(embeddings)
        self.fitted = True
        return pivots

    def add_column(
        self,
        vectors: np.ndarray,
        table: TableId,
        column: ColumnId,
    ):
        """
        Transform a column into distances → grid cells → insert into inverted index.
        """
        if not self.fitted:
            raise RuntimeError("PEXESOIndex must be fitted with fit() first.")

        dists = self.selector.transform(vectors)
        cells = self.grid.transform(dists)
        self.inv_index.add(cells, table, column)

    def lookup(self, vector: np.ndarray) -> Dict[Tuple[int, ...], list]:
        """
        Given a single embedding vector, return candidate postings
        from its grid cell.
        """
        dists = self.selector.transform(vector.reshape(1, -1))
        cells = self.grid.transform(dists)[0]
        postings = {cell: self.inv_index.query(cell) for cell in cells}
        return postings
