#!/usr/bin/env python3
"""Convert GSE224518 RAW CSV + metadata to publication_bundle leworld_input format.

Focus: mouse macrophage LPS time course (M0 polarization by default), aligned with
Sheu et al. Molecular Cell 2024 front-loaded stimulation design.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_META = ROOT / "data" / "external_geo" / "GSE224518" / "GSE224518_samptag.all_cellannotations_metadata.txt.gz"
DEFAULT_RAW = ROOT / "data" / "external_geo" / "GSE224518" / "GSE224518_RAW.tar"
DEFAULT_OUT = ROOT / "data" / "leworld_input_gse224518"


def _parse_timept_hours(label: str) -> int | None:
    label = str(label).strip()
    if not label or label.upper() == "NA":
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)\s*hr$", label, flags=re.I)
    if not m:
        return None
    val = float(m.group(1))
    if abs(val - round(val)) > 1e-6:
        return None
    return int(round(val))


def _cell_index_keys(raw: str) -> list[str]:
    s = str(raw).strip()
    keys = [s]
    if "_" in s:
        keys.append(s.split("_", 1)[1])
    return keys


def _normalize_cell_index(raw: str) -> str:
    return str(raw).strip()


def _load_metadata(meta_path: Path, cell_type: str | None, stimulus: str) -> pd.DataFrame:
    meta = pd.read_csv(meta_path, sep="\t")
    meta["Cell_Index"] = meta["Cell_Index"].astype(str).map(_normalize_cell_index)

    keep = pd.Series(False, index=meta.index)
    if stimulus.upper() == "LPS":
        keep |= meta["stimulus"].astype(str).eq("LPS")
        keep |= (meta["stimulus"].astype(str).eq("Unstim") & meta["timept"].astype(str).eq("0hr"))
    else:
        keep |= meta["stimulus"].astype(str).eq(stimulus)

    meta = meta.loc[keep].copy()
    if cell_type:
        meta = meta.loc[meta["type"].astype(str).eq(cell_type)].copy()

    meta["timepoint_numeric"] = meta["timept"].map(_parse_timept_hours)
    meta = meta.loc[meta["timepoint_numeric"].notna()].copy()
    meta["timepoint_numeric"] = meta["timepoint_numeric"].astype(int)
    meta = meta.drop_duplicates(subset=["Cell_Index"], keep="first")
    return meta.reset_index(drop=True)


def _iter_mol_tables(raw_tar: Path, extract_dir: Path):
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(raw_tar, "r") as tf:
        members = [
            m
            for m in tf.getmembers()
            if m.isfile() and m.name.endswith("_DBEC_MolsPerCell.csv.gz")
        ]
        if not members:
            raise FileNotFoundError(f"No *_DBEC_MolsPerCell.csv.gz files found in {raw_tar}")
        for member in members:
            tf.extract(member, path=extract_dir)
            yield extract_dir / member.name


def _read_mol_table(path: Path) -> pd.DataFrame:
    read_kw = {"comment": "#"}
    if path.suffix == ".gz":
        df = pd.read_csv(path, compression="gzip", **read_kw)
    else:
        df = pd.read_csv(path, **read_kw)
    if "Cell_Index" not in df.columns:
        raise KeyError(f"Missing Cell_Index in {path}")
    gene_cols = [c for c in df.columns if c not in {"Cell_Index", "Sample_Tag", "Sample_Name"}]
    out = df[gene_cols].copy()
    out.index = df["Cell_Index"].astype(str).map(_normalize_cell_index)
    return out


def _build_expression(
    meta: pd.DataFrame, raw_tar: Path, extract_dir: Path
) -> tuple[np.ndarray, list[str], list[str]]:
    wanted_ids = meta["Cell_Index"].tolist()
    wanted = set(wanted_ids)
    alias_to_canonical: dict[str, str] = {}
    for cid in wanted_ids:
        for key in _cell_index_keys(cid):
            alias_to_canonical[key] = cid

    chunks: list[pd.DataFrame] = []
    found: set[str] = set()

    for csv_path in _iter_mol_tables(raw_tar, extract_dir):
        table = _read_mol_table(csv_path)
        hit_keys = [k for k in table.index if k in alias_to_canonical]
        if not hit_keys:
            continue
        sub = table.loc[hit_keys].copy()
        sub.index = [alias_to_canonical[k] for k in sub.index]
        chunks.append(sub)
        found.update(sub.index.tolist())

    missing = wanted - found
    if missing:
        print(
            f"Warning: dropping {len(missing)} / {len(wanted)} cells without expression in RAW tar "
            f"(example: {sorted(missing)[:3]})"
        )
        wanted_ids = [cid for cid in wanted_ids if cid in found]

    expr = pd.concat(chunks, axis=0)
    expr = expr[~expr.index.duplicated(keep="first")]
    expr = expr.loc[wanted_ids]
    genes = expr.columns.tolist()
    return expr.to_numpy(dtype=np.float32), genes, wanted_ids


def convert(
    meta_path: Path,
    raw_tar: Path,
    out_dir: Path,
    n_hvg: int = 2000,
    cell_type: str | None = "M0",
    stimulus: str = "LPS",
    extract_dir: Path | None = None,
) -> dict:
    if not raw_tar.exists() or raw_tar.stat().st_size < 10_000_000:
        raise FileNotFoundError(
            f"Expected valid RAW tar at {raw_tar} (~60MB). "
            "Download via FTP: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE224nnn/GSE224518/suppl/GSE224518_RAW.tar"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = extract_dir or (raw_tar.parent / "extracted_raw")

    meta = _load_metadata(meta_path, cell_type=cell_type, stimulus=stimulus)
    X_raw, gene_names, kept_ids = _build_expression(meta, raw_tar, extract_dir)
    meta = meta.loc[meta["Cell_Index"].isin(kept_ids)].copy()
    meta = meta.set_index("Cell_Index").loc[kept_ids].reset_index()

    adata = sc.AnnData(X=X_raw, obs=meta.copy(), var=pd.DataFrame(index=gene_names))
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
        "source_meta": str(meta_path),
        "source_raw": str(raw_tar),
        "cell_type_filter": cell_type,
        "stimulus_filter": stimulus,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "timepoints": sorted(adata.obs["timepoint_numeric"].unique().tolist()),
        "cells_per_timepoint": adata.obs["timepoint_numeric"].value_counts().sort_index().to_dict(),
        "recommended_protocol": {
            "LEWORLD_TRAIN_TIMEPOINTS": "0,1,3",
            "LEWORLD_HOLDOUT_TIMEPOINTS": "8",
            "LEWORLD_MAX_TIME": "8",
            "endpoint": "3h->8h population extrapolation (LPS M0)",
        },
    }
    with open(out_dir / "data_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--raw-tar", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--extract-dir", type=Path, default=None)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--cell-type", type=str, default="M0", help="Use empty string for all types")
    parser.add_argument("--stimulus", type=str, default="LPS")
    args = parser.parse_args()
    cell_type = args.cell_type.strip() or None
    convert(
        args.meta,
        args.raw_tar,
        args.out,
        n_hvg=args.n_hvg,
        cell_type=cell_type,
        stimulus=args.stimulus,
        extract_dir=args.extract_dir,
    )


if __name__ == "__main__":
    main()
