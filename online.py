import os
import json
import csv
import numpy as np
import pandas as pd

from data.preprocess import normalize_column
from embedding.embedder import FastTextEmbedder
from index.pivots import PivotSelector
from index.grid import HierarchicalGrid
from index.inverted_index import InvertedIndex
from search.blocking import Blocker
from search.verify import Verifier
from utils.config import Config
from utils.types import TableId, ColumnId, GREEN, RED, YELLOW, RESET


# ONLINE PHASE
def run_online(query_dir: str, out_dir: str, cfg: Config):
    print(f"\nStarting ONLINE PHASE...")
    embedder = FastTextEmbedder(dim=300)

    # load artifacts
    pivots = np.load(os.path.join(out_dir, "pivots.npy"))
    with open(os.path.join(out_dir, "grid.json")) as f:
        grid_cfg = json.load(f)
    with open(os.path.join(out_dir, "inverted_index.json")) as f:
        inv_data = json.load(f)
    with open(os.path.join(out_dir, "col_meta.json")) as f:
        col_meta = json.load(f)
    emb_data = np.load(os.path.join(out_dir, "embeddings.npz"))

    # rebuild structures
    selector = PivotSelector(k=cfg.pivots_k, seed=cfg.seed)
    selector.pivots = pivots
    grid = HierarchicalGrid(levels=grid_cfg["levels"])
    grid.bin_edges = grid_cfg["bin_edges"]
    inv_index = InvertedIndex()
    for entry in inv_data:
        cell = tuple(entry["cell"])
        for p in entry["postings"]:
            inv_index.index[cell].append((TableId(p["table"]), ColumnId(p["column"]), p["row_id"]))

    # rebuild cand_vecs_map
    cand_vecs_map = {}
    for cid_str, meta in col_meta.items():
        vecs = emb_data[cid_str]
        key = (meta["table"], meta["column"])
        cand_vecs_map[key] = vecs

    # query
    query_files = [f for f in os.listdir(query_dir) if f.endswith(".csv") or f.endswith(".tsv")]
    results_out = []
    for qf in query_files:
        qpath = os.path.join(query_dir, qf)
        df = pd.read_csv(qpath)
        for qcol in df.columns:
            print(f"{YELLOW}Querying column: {qcol} in {qf}{RESET}")
            values = normalize_column(df[qcol])
            q_vecs = embedder.embed(values)

            q_dists = selector.transform(q_vecs)
            q_cells = grid.transform(q_dists)
            leaf_cells = [row[-1] for row in q_cells]

            postings = {}
            for cell in leaf_cells:
                postings[cell] = inv_index.query(cell)

            blocker = Blocker(cfg)
            verifier = Verifier(cfg)
            candidates = blocker.block(q_vecs, postings, cand_vecs_map)
            verified = verifier.verify(TableId(qf), ColumnId(qcol), len(values), candidates)

            for res in verified:
                if res.is_joinable:
                    results_out.append([res.candidate_table.name, res.candidate_column.name, res.joinability])

    out_csv = os.path.join(out_dir, "joinable.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "column_name", "joinability_score"])
        writer.writerows(results_out)
    print(f"{GREEN}Results saved to {out_csv}{RESET}")