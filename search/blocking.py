from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from utils.types import TableId, ColumnId
from utils.config import Config


class Blocker:
    """
    Blocking step:
    - Find candidate columns in the same grid cell as query.
    - Apply τ-based distance filtering to prune dissimilar vectors.
    """

    def __init__(self, config: Config):
        self.cfg = config

    def block(
        self,
        query_vecs: np.ndarray,              # (n_q, d)
        candidate_postings: Dict[Tuple[int, ...], List[Tuple[TableId, ColumnId, int]]],
        cand_vecs_map: Dict[Tuple[str, str], np.ndarray],  # (table, col) -> embeddings
    ) -> Dict[Tuple[TableId, ColumnId], List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Given query embeddings and postings from inverted index,
        return candidate column matches after τ filtering.

        Returns:
            Dict[(table, col)] -> list of (q_vec, c_vec) pairs
        """
        candidates: Dict[Tuple[TableId, ColumnId], List[Tuple[np.ndarray, np.ndarray]]] = {}

        # Precompute max distance in normalized L2 space (≈ 2.0)
        tau_threshold = self.cfg.tau_ratio * 2.0

        for cell, postings in candidate_postings.items():
            for (table, col, row_id) in postings:
                key = (table, col)
                cand_vecs = cand_vecs_map[(table.name, col.name)]

                # Ensure row index is valid
                if row_id >= len(cand_vecs):
                    continue

                c_vec = cand_vecs[row_id]
                for q_vec in query_vecs:
                    dist = np.linalg.norm(q_vec - c_vec)

                    # τ-based pruning
                    if dist <= tau_threshold:
                        if key not in candidates:
                            candidates[key] = []
                        candidates[key].append(row_id)

        return candidates
