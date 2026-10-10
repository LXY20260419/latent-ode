#!/usr/bin/env python3
"""Build a leakage-audited bundle for the GSE305979 vehicle-control trajectory.

Inputs (under ``data/external_geo/GSE305979``):

* ``GSE305979_day0_raw_counts.h5ad`` — untreated, unhashed day-0 cells.
* ``GSE305979_day1-7_demuxed_counts.h5ad`` — demultiplexed day 1–7 singlets
  with per-well treatment assignments.

The converter keeps only the vehicle/untreated trajectory, derives the sampling
day from the plate/sample labels (see the label audit below), and fits
normalization, feature selection and scaling on training timepoints only
(0/1/2 days).  Output contract matches ``data/leworld_input_gse148794`` etc.:
``expression_matrix.npy`` (cells × 2991 genes), ``metadata.csv``,
``gene_names.txt``.

Label audit (recorded in the QC report as ``source_timepoint_audit``): the
demuxed object's ``TIMEPOINT_HOURS`` column mislabels every Day 7 cell as
``24.0`` h.  ``PLATE_NAME`` (``Day 7``), ``sample_name`` (``Day 7_Lib 1..4``)
and the series description ("Days 1, 2, 5, and 7") all agree that the fourth
plate is day 7 = 168 h.  The converter therefore derives the day from the
plate/sample label and records the row-level mismatch count instead of
silently trusting the broken column.

``--inspect`` prints the obs columns and treatment-value counts of the
demuxed file, then exits.  The vehicle filter must be locked from that output
before the real conversion; there is no silent fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_external_candidate_bundle import (  # noqa: E402
    fit_select_scale,
    make_unique,
    normalize_log_sparse,
)

SOURCE_DIR = ROOT / "data" / "external_geo" / "GSE305979"
OUTPUT_DIR = ROOT / "data" / "leworld_input_gse305979"
REPORT_DIR = ROOT / "artifacts" / "reports"

TRAIN_TIMES = [0, 1, 2]
HOLDOUT_TIMES = [5]
SECONDARY_TIMES = [7]
RESULT_TAG = "ext_gse305979_control_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def categorical_values(frame: pd.DataFrame, column: str):
    series = frame[column]
    if isinstance(series.dtype, pd.CategoricalDtype):
        values = series.astype(str)
    else:
        values = series.astype(str)
    return values


def inspect_demuxed(path: Path) -> None:
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    print(f"shape={adata.shape}")
    print("obs columns:")
    for column in adata.obs.columns:
        print(f"  - {column}: {adata.obs[column].dtype}")
    for column in adata.obs.columns:
        lowered = column.lower()
        if any(key in lowered for key in ("treat", "compound", "vehicle", "well", "cmo", "sample", "typ", "name", "id")):
            try:
                counts = adata.obs[column].astype(str).value_counts()
            except Exception as exc:  # pragma: no cover - diagnostics only
                print(f"  {column}: unreadable ({exc})")
                continue
            print(f"  {column}: {counts.head(40).to_dict()}")
    if "TIMEPOINT_HOURS" in adata.obs.columns:
        print(
            "TIMEPOINT_HOURS:",
            adata.obs["TIMEPOINT_HOURS"].astype(str).value_counts().to_dict(),
        )
    adata.file.close()


def load_day0(path: Path):
    import anndata as ad

    adata = ad.read_h5ad(path)
    counts = adata.X.tocsr().astype(np.float32)
    genes, duplicated = make_unique(list(map(str, adata.var_names)))
    obs = pd.DataFrame(
        {
            "cell_id": [f"d0_{i}" for i in range(adata.n_obs)],
            "sample_id": adata.obs.get("POOL_NAME", pd.Series(["Day 0"] * adata.n_obs))
            .astype(str)
            .values,
            "library_id": adata.obs.get("LIBRARY_ID", pd.Series(["unknown"] * adata.n_obs))
            .astype(str)
            .values,
            "timepoint_numeric": 0,
        }
    )
    if "TIMEPOINT_HOURS" in adata.obs.columns:
        hours = adata.obs["TIMEPOINT_HOURS"].astype(int).to_numpy()
        if not np.all(hours == 0):
            raise ValueError(f"day0 file contains non-zero timepoints: {sorted(set(hours))}")
    return counts, genes, obs, duplicated


DAY_LABEL_COLUMNS = ("PLATE_NAME", "sample_name", "POOL_NAME")


def derive_days_from_labels(obs):
    """Derive whole days from the plate/sample labels (Day N)."""
    for column in DAY_LABEL_COLUMNS:
        if column not in obs.columns:
            continue
        extracted = obs[column].astype(str).str.extract(
            r"[Dd]ay[ _]?(\d+)", expand=False
        )
        if extracted.notna().all():
            return extracted.astype(int).to_numpy(), column
    raise ValueError(
        "no plate/sample label column exposes a Day N pattern; "
        f"tried {DAY_LABEL_COLUMNS}"
    )


def load_demuxed(path: Path, vehicle_pattern: str):
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    obs_all = adata.obs
    days, label_column = derive_days_from_labels(obs_all)

    audit = {"derived_from_column": label_column}
    if "TIMEPOINT_HOURS" in obs_all.columns:
        recorded_hours = obs_all["TIMEPOINT_HOURS"].astype(float).to_numpy()
        derived_hours = days * 24.0
        mismatch = ~np.isclose(recorded_hours, derived_hours)
        audit["recorded_column"] = "TIMEPOINT_HOURS"
        audit["mismatch_rows"] = int(mismatch.sum())
        breakdown = Counter(
            (int(d), int(h))
            for d, h in zip(days[mismatch], recorded_hours[mismatch])
        )
        audit["mismatch_breakdown"] = {
            f"derived_day_{d}_recorded_{h}h": int(n)
            for (d, h), n in sorted(breakdown.items())
        }
        audit["resolution"] = (
            "days derived from plate/sample labels; the recorded column is "
            "kept in the audit but is not trusted for the split"
        )

    treatment_columns = [
        column
        for column in obs_all.columns
        if re.search(r"(treat|compound|vehicle|cmo|well)", column, re.IGNORECASE)
    ]
    if not treatment_columns:
        raise ValueError("no treatment-like column found in the demuxed obs")

    regex = re.compile(vehicle_pattern, re.IGNORECASE)
    mask = np.zeros(adata.n_obs, dtype=bool)
    matched_column = None
    for column in treatment_columns:
        values = obs_all[column].astype(str)
        current = values.str.contains(regex, regex=True).to_numpy()
        if current.any():
            mask |= current
            matched_column = column
            break
    if not mask.any() or matched_column is None:
        options = {
            column: sorted(obs_all[column].astype(str).unique().tolist())[:50]
            for column in treatment_columns
        }
        raise ValueError(
            f"vehicle pattern {vehicle_pattern!r} matched no cells; columns={options}"
        )
    audit["vehicle_column"] = matched_column
    audit["vehicle_pattern"] = vehicle_pattern
    audit["vehicle_cells"] = int(mask.sum())
    audit["vehicle_cells_total_demuxed"] = int(adata.n_obs)
    audit["compound_classes"] = int(
        obs_all[matched_column].astype(str).nunique()
    )

    sub = adata[mask].to_memory()
    adata.file.close()
    counts = sub.X.tocsr().astype(np.float32)
    genes, duplicated = make_unique(list(map(str, sub.var_names)))
    obs = pd.DataFrame(
        {
            "cell_id": [f"dv_{i}" for i in range(sub.n_obs)],
            "sample_id": sub.obs.get(
                "POOL_NAME", pd.Series(["unknown"] * sub.n_obs)
            ).astype(str).values,
            "library_id": sub.obs.get(
                "LIBRARY_ID", pd.Series(["unknown"] * sub.n_obs)
            ).astype(str).values,
            "timepoint_numeric": days[mask],
            "treatment_value": sub.obs[matched_column].astype(str).values,
        }
    )
    return counts, genes, obs, duplicated, matched_column, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--vehicle-pattern", default=None)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-genes", type=int, default=2991)
    args = parser.parse_args()

    day0_path = args.source_dir / "GSE305979_day0_raw_counts.h5ad"
    demuxed_path = args.source_dir / "GSE305979_day1-7_demuxed_counts.h5ad"
    for path in (day0_path, demuxed_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.inspect:
        inspect_demuxed(demuxed_path)
        return 0

    if not args.vehicle_pattern:
        raise SystemExit("--vehicle-pattern is required (run --inspect first)")

    import scipy.sparse as sparse

    d0_counts, d0_genes, d0_obs, d0_dup = load_day0(day0_path)
    dv_counts, dv_genes, dv_obs, dv_dup, matched_column, source_audit = load_demuxed(
        demuxed_path, args.vehicle_pattern
    )
    print(f"day0 cells={d0_counts.shape[0]}, vehicle cells={dv_counts.shape[0]}")
    print(
        "source timepoint audit:",
        json.dumps(source_audit, ensure_ascii=False),
    )

    common_genes = [g for g in d0_genes if g in set(dv_genes)]
    if len(common_genes) < 20000:
        raise SystemExit(
            "gene panels overlap by only "
            f"{len(common_genes)} symbols — reconcile before conversion"
        )
    if d0_genes != dv_genes:
        d0_index = [d0_genes.index(g) for g in common_genes]
        dv_index = [dv_genes.index(g) for g in common_genes]
        d0_counts = d0_counts[:, d0_index]
        dv_counts = dv_counts[:, dv_index]
        print(
            f"gene panels differ; intersected to {len(common_genes)} symbols "
            f"(day0 {len(d0_genes)}, demuxed {len(dv_genes)})"
        )
        d0_genes = common_genes

    counts = sparse.vstack([d0_counts, dv_counts], format="csr")
    metadata = pd.concat([d0_obs, dv_obs], ignore_index=True).set_index("cell_id")
    metadata["timepoint_numeric"] = metadata["timepoint_numeric"].astype(int)

    log_counts = normalize_log_sparse(counts)
    matrix, selected_genes, fit_record = fit_select_scale(
        log_counts, metadata, d0_genes, TRAIN_TIMES, args.n_genes
    )

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "expression_matrix.npy", matrix)
    metadata.to_csv(out_dir / "metadata.csv")
    (out_dir / "gene_names.txt").write_text(
        "\n".join(selected_genes) + "\n", encoding="utf-8"
    )

    counts_by_time = Counter(metadata["timepoint_numeric"].tolist())
    cells_by_sample = Counter(metadata["sample_id"].astype(str).tolist())
    input_hashes = {
        name: sha256(args.source_dir / name)
        for name in (
            "GSE305979_day0_raw_counts.h5ad",
            "GSE305979_day1-7_demuxed_counts.h5ad",
        )
    }
    qc = {
        "record_type": "external_candidate_bundle_qc",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "accession": "GSE305979",
        "result_tag": RESULT_TAG,
        "source_dir": str(args.source_dir),
        "input_sha256": input_hashes,
        "vehicle_pattern": args.vehicle_pattern,
        "vehicle_column": matched_column,
        "source_timepoint_audit": source_audit,
        "input_shape_gene_by_cell_or_cell_by_gene": {
            "n_cells": int(matrix.shape[0]),
            "n_genes_after_selection": int(matrix.shape[1]),
            "raw_nonzero_entries": int(counts.nnz),
        },
        "timepoints": sorted(int(t) for t in counts_by_time),
        "cells_per_timepoint": {
            str(k): int(v) for k, v in sorted(counts_by_time.items())
        },
        "cells_per_sample": {
            str(k): int(v) for k, v in sorted(cells_by_sample.items())
        },
        "train_timepoints": TRAIN_TIMES,
        "holdout_timepoints": HOLDOUT_TIMES,
        "secondary_holdout_timepoints": SECONDARY_TIMES,
        "fit_record": fit_record,
        "gene_names_sha256": sha256(out_dir / "gene_names.txt"),
        "metadata_sha256": sha256(out_dir / "metadata.csv"),
        "expression_matrix_sha256": sha256(out_dir / "expression_matrix.npy"),
        "status": (
            "PASS_FOR_SMOKE"
            if set(TRAIN_TIMES + HOLDOUT_TIMES).issubset(set(counts_by_time))
            else "FAIL_MISSING_TIMEPOINT"
        ),
        "replicate_note": (
            "single donor; cells nested within CMO well and library. "
            "No donor-level inference; protocol/seed-level uncertainty only."
        ),
        "leakage_control": (
            "per-cell library normalization; HVG selection and scaling fit on "
            f"training timepoints only ({TRAIN_TIMES})"
        ),
    }
    report_path = REPORT_DIR / "external_gse305979_qc.json"
    report_path.write_text(
        json.dumps(qc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(qc, indent=2, ensure_ascii=False))
    return 0 if qc["status"] == "PASS_FOR_SMOKE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
