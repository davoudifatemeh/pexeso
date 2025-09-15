from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# distance types
DistanceType = Literal["euclidean"]

@dataclass
class Config:
    """Global configuration for PEXESO (v1: in-memory only)."""
    # index parameters
    pivots_k: int = 3              # |P| (recommended 3..7)
    grid_levels: int = 3           # m (recommended 3..8)

    # distance / thresholds
    distance: DistanceType = "euclidean"  # on L2-normalized vectors
    tau_ratio: float = 0.06        # τ as % of max distance (2.0 for L2-norm)
    T_ratio: float = 0.60          # joinability threshold (% of |Q|)

    # data heuristics
    min_col_len: int = 5          # ignore very small columns
    distinct_ratio_min: float = 0.30  # heuristic used in adapters.detect_key_columns

    # misc
    seed: int = 42
    dtype: str = "float32"
