import os
import sys
import json
import numpy as np

# project imports
from data.adapters import CSVFolderAdapter
from data.preprocess import normalize_column
from embedding.embedder import FastTextEmbedder
from index.pivots import PivotSelector
from index.grid import HierarchicalGrid
from index.inverted_index import InvertedIndex
from utils.config import Config
from utils.types import TableId, ColumnId, GREEN, RED, YELLOW, RESET


def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    return obj

# OFFLINE PHASE
def run_offline(dataset_dir: str, out_dir: str, cfg: Config):
    os.makedirs(out_dir, exist_ok=True)
    adapter = CSVFolderAdapter(dataset_dir)
    embedder = FastTextEmbedder(dim=300)

    all_embeddings = []
    col_embeddings = {}
    col_meta = {}
    col_id_counter = 0

    print(f"\nScanning dataset for offline phase...")
    for table_id, df in adapter.iter_tables():
        print(f"Processing table: {table_id.name}")
        for c in df.columns:
            df[c] = normalize_column(df[c])

        key_cols = adapter.detect_key_columns(df)
        for cid in key_cols:
            col_name = cid.name
            values = df[col_name].tolist()
            if len(values) < cfg.min_col_len:
                continue
            vecs = embedder.embed(values)
            col_embeddings[col_id_counter] = vecs
            col_meta[col_id_counter] = {"table": table_id.name, "column": col_name}
            all_embeddings.append(vecs)
            col_id_counter += 1

    if not all_embeddings:
        print(f"{RED}No embeddings found!{RESET}")
        sys.exit(1)

    all_embeddings = np.vstack(all_embeddings)
    print(f"Collected embeddings: {all_embeddings.shape}")

    # pivots
    selector = PivotSelector(k=cfg.pivots_k, seed=cfg.seed)
    pivots = selector.fit(all_embeddings)
    np.save(os.path.join(out_dir, "pivots.npy"), pivots)
    print(f"{GREEN}Saved pivots.npy{RESET}")

    # grid
    grid = HierarchicalGrid(levels=cfg.grid_levels)
    grid.fit(selector.transform(all_embeddings))
    grid_config = {"levels": grid.levels, "bin_edges": _to_serializable(grid.bin_edges)}
    with open(os.path.join(out_dir, "grid.json"), "w") as f:
        json.dump(grid_config, f)
    print(f"{GREEN}Saved grid.json{RESET}")

    # inverted index
    inv = InvertedIndex()
    for cid, vecs in col_embeddings.items():
        dists = selector.transform(vecs)
        cells_per_row = grid.transform(dists)
        leaf_cells = [row[-1] for row in cells_per_row]
        t = TableId(col_meta[cid]["table"])
        c = ColumnId(col_meta[cid]["column"])
        inv.add(leaf_cells, t, c)

    inv_serializable = []
    for cell, postings in inv.index.items():
        inv_serializable.append({
            "cell": [int(x) for x in cell],
            "postings": [
                {"table": p[0].name, "column": p[1].name, "row_id": int(p[2])}
                for p in postings
            ]
        })
    with open(os.path.join(out_dir, "inverted_index.json"), "w") as f:
        json.dump(inv_serializable, f)
    print(f"{GREEN}Saved inverted_index.json{RESET}")

    # save embeddings for lookup in online phase
    np.savez_compressed(
        os.path.join(out_dir, "embeddings.npz"),
        **{str(cid): vecs for cid, vecs in col_embeddings.items()}
    )
    with open(os.path.join(out_dir, "col_meta.json"), "w") as f:
        json.dump(col_meta, f)

    print(f"{GREEN}OFFLINE PHASE COMPLETED{RESET}")