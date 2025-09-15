import re
import pandas as pd
from typing import List

_punct_re = re.compile(r"[^\w\s]", re.UNICODE)

def clean_text(values: List[str]) -> List[str]:
    """Lowercase, strip, remove punctuation."""
    cleaned = []
    for v in values:
        if not isinstance(v, str):
            v = str(v)
        v = v.lower().strip()
        v = _punct_re.sub("", v)
        cleaned.append(v if v != "" else "__EMPTY__")
    return cleaned


def normalize_column(series: pd.Series) -> List[str]:
    """Normalize a column to strings ready for embedding."""
    if pd.api.types.is_numeric_dtype(series):
        values = [str(v) if pd.notna(v) else "__EMPTY__" for v in series]
    elif pd.api.types.is_datetime64_any_dtype(series):
        values = [
            str(pd.to_datetime(v, errors="coerce").date()) if pd.notna(v) else "__EMPTY__"
            for v in series
        ]
    else:
        values = clean_text(series.fillna("__EMPTY__").astype(str).tolist())

    # final cleanup
    values = [v if v.strip() != "" else "__EMPTY__" for v in values]
    return values
