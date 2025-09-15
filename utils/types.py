from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TableId:
    """Identifier for a table in the data lake."""
    name: str  # e.g., "people.csv"

@dataclass(frozen=True)
class ColumnId:
    """Identifier for a column in a table."""
    name: str  # e.g., "name"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
