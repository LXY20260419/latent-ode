#!/usr/bin/env python3
"""Convert GSE226824 filtered h5ad.gz to publication_bundle leworld_input format."""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT / "data" / "external_geo" / "GSE226824" / "GSE226824_HSPC-all_filtered.h5ad.gz"
DEFAULT_OUT = ROOT / "data" / "leworld_input_gse226824"

TIME_MAP = {"control": 0, "3h": 3, "24h": 24, "72h": 72}


def _read_h5ad(path: Path) -> sc.AnnData:
    if path.suffix == ".gz":
        tmp = path.with_suffix("")
        if not tmp.exists() or tmp.stat().st_size < path.stat().st_size * 0.9:
            with gzip.open(path, "rb") as fi, open(tmp, "wb") as fo:
                shutil.copyfileobj(fi, fo)
        path = tmp
    return sc.read_h5ad(path)


def convert(src: Path, out_dir: Path, n_hvg: int = 2000) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    adata = _read_h5ad(src)
    if "time" not in adata.obs.columns:
        raise KeyError(f"Expected obs['time'], got columns: {list(adata.obs.columns)}")

    adata.obs["timepoint_numeric"] = adata.obs["time"].astype(str).map(TIME_MAP)
    missing = adata.obs["timepoint_numeric"].isna()
    if missing.any():
        bad = sorted(adata.obs.loc[missing, "time"].astype(str).unique().tolist())
        raise ValueError(f"Unmapped time labels: {bad}")

    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3")
    except Exception:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)

    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    np.save(out_dir / "expression_matrix.npy", X.astype(np.float32))
    adata.obs.to_csv(out_dir / "metadata.csv")
    with open(out_dir / "gene_names.txt", "w", encoding="utf-8") as f:
        for g in adata.var_names:
            f.write(f"{g}\n")

    summary = {
        "source": str(src),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "timepoints": sorted(adata.obs["timepoint_numeric"].unique().tolist()),
        "cells_per_timepoint": adata.obs["timepoint_numeric"].value_counts().sort_index().to_dict(),
    }
    with open(out_dir / "data_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-hvg", type=int, default=2000)
    args = parser.parse_args()
    convert(args.src, args.out, n_hvg=args.n_hvg)


if __name__ == "__main__":
    main()
