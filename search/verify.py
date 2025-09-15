from __future__ import annotations
from typing import Dict, Tuple, List
from utils.result import JoinableResult
from utils.types import TableId, ColumnId
from utils.config import Config


class Verifier:
    """
    Verifies candidate joinability by computing overlap ratio.
    """

    def __init__(self, config: Config):
        self.cfg = config

    def verify(
        self,
        query_table: TableId,
        query_column: ColumnId,
        query_size: int,
        candidates: Dict[Tuple[str, str], List[int]]
    ) -> List[JoinableResult]:
        """
        Check joinability of query column with candidates.

        Args:
            query_table: identifier of the query table
            query_column: identifier of the query column
            query_size: number of rows in the query column
            candidates: dict mapping (table, column) → list of matched row_ids

        Returns:
            results: list of JoinableResult
        """
        results: List[JoinableResult] = []

        for (table, col), matched_rows in candidates.items():
            matches = len(set(matched_rows))
            joinability = matches / max(1, query_size)
            is_joinable = joinability >= self.cfg.T_ratio

            res = JoinableResult(
                query_table=query_table,
                candidate_table=TableId(table),
                query_column=query_column,
                candidate_column=ColumnId(col),
                joinability=joinability,
                is_joinable=is_joinable,
                matches=matches,
                query_size=query_size,
            )
            results.append(res)

        return results
