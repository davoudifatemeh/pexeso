import os
import pandas as pd
from typing import Iterator, Tuple, List

from utils.types import TableId, ColumnId

class DataLakeAdapter:
    """Base class for data lake adapters."""
    def iter_tables(self) -> Iterator[Tuple[TableId, pd.DataFrame]]:
        raise NotImplementedError

    def detect_key_columns(self, df: pd.DataFrame) -> List[ColumnId]:
        """Heuristic: pick columns that are good join candidates."""
        candidates = []
        n_rows = len(df)

        for col in df.columns:
            series = df[col].dropna()
            # distinct_ratio = series.nunique() / max(1, n_rows)

            # # skip repetitive columns
            # if distinct_ratio < 0.3:
            #     continue

            # Check "__EMPTY__" ratio (after normalization)
            empty_ratio = (series == "__EMPTY__").sum() / max(1, n_rows)
            if empty_ratio > 0.9:   # skip columns with >90% empties
                continue


            # accept strings, numerics, datetimes
            if pd.api.types.is_string_dtype(series):
                candidates.append(ColumnId(name=col))
            elif pd.api.types.is_numeric_dtype(series):
                candidates.append(ColumnId(name=col))
            elif pd.api.types.is_datetime64_any_dtype(series):
                candidates.append(ColumnId(name=col))

        return candidates


class CSVFolderAdapter(DataLakeAdapter):
    """Adapter for a folder of CSV/TSV files."""

    def __init__(self, folder: str, sep: str = ","):
        self.folder = folder
        self.sep = sep

    def iter_tables(self) -> Iterator[Tuple[TableId, pd.DataFrame]]:
        for fname in os.listdir(self.folder):
            if not (fname.endswith(".csv") or fname.endswith(".tsv")):
                continue
            sep = "\t" if fname.endswith(".tsv") else self.sep
            path = os.path.join(self.folder, fname)
            try:
                df = pd.read_csv(path, sep=sep)
                yield TableId(name=fname), df
            except Exception as e:
                print(f"⚠ Could not load {fname}: {e}")
