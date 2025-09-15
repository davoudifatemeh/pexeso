# pexeso/index/inverted_index.py
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from utils.types import TableId, ColumnId


class InvertedIndex:
    """
    Inverted index mapping grid cell IDs → postings list.
    Each posting is (table_id, column_id, row_id).
    """

    def __init__(self):
        # keys = cell_id (tuple of ints)
        # values = list of postings
        self.index: Dict[Tuple[int, ...], List[Tuple[TableId, ColumnId, int]]] = defaultdict(list)

    def add(self, cells: List[Tuple[int, ...]], table: TableId, column: ColumnId):
        """
        Add all rows from one column into the inverted index.
        Args:
            cells: list of grid cell IDs for each row.
            table: table identifier
            column: column identifier
        """
        for row_id, cell in enumerate(cells):
            self.index[cell].append((table, column, row_id))

    def query(self, cell: Tuple[int, ...]) -> List[Tuple[TableId, ColumnId, int]]:
        """Return postings for a given cell ID."""
        return self.index.get(cell, [])

    def __len__(self):
        return sum(len(v) for v in self.index.values())
