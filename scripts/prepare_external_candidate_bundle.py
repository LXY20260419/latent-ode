#!/usr/bin/env python3
"""Prepare a leakage-audited bundle for the first external candidates.

The existing benchmark consumes a dense, log-normalized, scaled matrix.  This
converter fits feature selection and scaling statistics on *training
timepoints only* and applies them to holdout cells.  It therefore avoids the
all-timepoint HVG/scaling leakage present in the legacy external converters.

Raw/processed GEO inputs are ignored by git.  The generated bundle contains
metadata and a QC/provenance report; the large ``expression_matrix.npy`` is
also ignored by repository policy.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "artifacts" / "reports"

DEFAULTS = {
    "GSE148794": {
        "train_times": [0, 3, 6, 9],
        "holdout_times": [12],
        "secondary_times": [15],
        "result_tag": "ext_gse148794_v1",
    },
    "GSE149622": {
        "train_times": [0, 1, 2, 3],
        "holdout_times": [4],
        "secondary_times": [],
        "result_tag": "ext_gse149622_v1",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_unique(names: Iterable[str]) -> tuple[list[str], int]:
    seen: Counter[str] = Counter()
    result = []
    duplicated = 0
    for raw in names:
        name = str(raw).strip() or "unnamed_gene"
        count = seen[name]
        if count:
            duplicated += 1
            result.append(f"{name}__dup{count}")
        else:
            result.append(name)
        seen[name] += 1
    return result, duplicated


def normalize_log_dense(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float32)
    library = counts.sum(axis=1, dtype=np.float64).astype(np.float32)
    safe_library = np.maximum(library, 1.0)
    normalized = counts / safe_library[:, None] * np.float32(1e4)
    np.log1p(normalized, out=normalized)
    return normalized


def normalize_log_sparse(counts: sparse.spmatrix) -> sparse.csr_matrix:
    matrix = counts.tocsr().astype(np.float32)
    library = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    safe_library = np.maximum(library, 1.0)
    matrix = sparse.diags(1.0 / safe_library).dot(matrix).tocsr()
    matrix = matrix.multiply(np.float32(1e4)).tocsr()
    matrix.data = np.log1p(matrix.data).astype(np.float32)
    return matrix


def fit_select_scale(
    log_counts: np.ndarray | sparse.csr_matrix,
    metadata: pd.DataFrame,
    gene_names: list[str],
    train_times: list[int],
    n_genes: int,
) -> tuple[np.ndarray, list[str], dict]:
    train_mask = metadata["timepoint_numeric"].isin(train_times).to_numpy()
    if not train_mask.any():
        raise ValueError(f"No training cells for times {train_times}")

    if sparse.issparse(log_counts):
        train = log_counts[train_mask]
        mean = np.asarray(train.mean(axis=0)).ravel().astype(np.float32)
        mean_sq = np.asarray(train.multiply(train).mean(axis=0)).ravel().astype(np.float32)
        variance = np.maximum(mean_sq - mean * mean, 0.0)
    else:
        train = np.asarray(log_counts[train_mask], dtype=np.float32)
        mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
        variance = train.var(axis=0, dtype=np.float64).astype(np.float32)

    keep_n = min(int(n_genes), len(gene_names))
    # Stable tie-break: descending variance, then original gene order.
    order = np.lexsort((np.arange(len(gene_names)), -variance))
    selected = np.sort(order[:keep_n])

    if sparse.issparse(log_counts):
        selected_matrix = log_counts[:, selected].toarray().astype(np.float32)
    else:
        selected_matrix = np.asarray(log_counts[:, selected], dtype=np.float32)
    train_selected = selected_matrix[train_mask]
    train_mean = train_selected.mean(axis=0, dtype=np.float64).astype(np.float32)
    train_std = train_selected.std(axis=0, dtype=np.float64).astype(np.float32)
    train_std[train_std < 1e-6] = 1.0
    selected_matrix = (selected_matrix - train_mean[None, :]) / train_std[None, :]
    np.clip(selected_matrix, -10.0, 10.0, out=selected_matrix)

    fit_record = {
        "fit_timepoints": [int(t) for t in train_times],
        "fit_cells": int(train_mask.sum()),
        "selected_gene_count": int(keep_n),
        "selected_gene_indices_sha256": hashlib.sha256(
            selected.astype(np.int64).tobytes()
        ).hexdigest(),
        "normalization": "per-cell library size 1e4 followed by log1p",
        "scaling": "mean/std fitted on training timepoints only; clip [-10, 10]",
    }
    return selected_matrix.astype(np.float32), [gene_names[i] for i in selected], fit_record


def parse_gse149622(source_dir: Path) -> tuple[np.ndarray | sparse.csr_matrix, list[str], pd.DataFrame]:
    matrix_path = source_dir / "GSE149622_RNA_timecourse_unadjusted_matrix.mtx.gz"
    barcodes_path = source_dir / "GSE149622_RNA_timecourse_unadjusted_barcodes.tsv.gz"
    genes_path = source_dir / "GSE149622_RNA_timecourse_unadjusted_genes.tsv.gz"
    required = [matrix_path, barcodes_path, genes_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing GSE149622 inputs: {missing}")

    with gzip.open(genes_path, "rt", encoding="utf-8") as handle:
        genes = [line.rstrip("\n").split("\t")[0] for line in handle if line.strip()]
    with gzip.open(barcodes_path, "rt", encoding="utf-8") as handle:
        barcodes = [line.rstrip("\n").split("\t")[0] for line in handle if line.strip()]
    gene_names, duplicated_genes = make_unique(genes)
    if duplicated_genes:
        print(f"GSE149622: made {duplicated_genes} duplicate gene names unique")

    matrix = mmread(str(matrix_path))
    if matrix.shape != (len(gene_names), len(barcodes)):
        raise ValueError(
            f"GSE149622 matrix shape {matrix.shape} does not match "
            f"genes/cells {(len(gene_names), len(barcodes))}"
        )
    counts = matrix.T.tocsr().astype(np.float32)

    rows = []
    for barcode in barcodes:
        match = re.match(r"^Timept(?P<time>\d+)\.", barcode)
        if match is None:
            raise ValueError(f"Unmapped GSE149622 barcode: {barcode}")
        timepoint = int(match.group("time"))
        sample_id = barcode.split(".", 1)[0]
        rows.append(
            {
                "cell_id": barcode,
                "sample_id": sample_id,
                "replicate_level": "single_pooled_sample_per_timepoint",
                "condition": "IL23_skin_ILC_timecourse",
                "timepoint_numeric": timepoint,
            }
        )
    metadata = pd.DataFrame(rows).set_index("cell_id")
    metadata["timepoint_numeric"] = metadata["timepoint_numeric"].astype(int)
    metadata.attrs["duplicated_genes"] = duplicated_genes
    return counts, gene_names, metadata


def load_gse148794_sample_times(meta_path: Path) -> dict[str, int]:
    metadata = pd.read_csv(meta_path, sep="\t", compression="gzip")
    mapping: dict[str, int] = {}
    for title in metadata["Title"].astype(str):
        match = re.search(
            r"Time Course IBD\.(CONTROL|DAY(?P<day>\d+))\.(SCI\d+_\d+)_S\d+",
            title,
        )
        if match is None:
            continue
        timepoint = 0 if match.group(1) == "CONTROL" else int(match.group("day"))
        key = re.search(r"(SCI\d+_\d+)", title)
        if key is None:
            raise ValueError(f"Missing sample key in title: {title}")
        mapping[key.group(1)] = timepoint
    return mapping


def parse_gse148794(
    source_dir: Path,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    count_path = source_dir / "GSE148794_tc_ibd.count_table.tsv.gz"
    meta_path = source_dir / "GSE148794_tc_ibd.metadata.tsv.gz"
    required = [count_path, meta_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing GSE148794 inputs: {missing}")

    sample_times = load_gse148794_sample_times(meta_path)
    table = pd.read_csv(count_path, sep="\t", compression="gzip", index_col=0)
    genes, duplicated_genes = make_unique(table.index.tolist())
    table.index = genes

    keep_columns = []
    rows = []
    for column in table.columns.astype(str):
        match = re.search(r"(SCI\d+_\d+)", column)
        if match is None:
            continue
        sample_id = match.group(1)
        if sample_id not in sample_times:
            continue
        keep_columns.append(column)
        rows.append(
            {
                "cell_id": column,
                "sample_id": sample_id,
                "replicate_level": "cell_within_sample_library; pooled_animals",
                "condition": "DSS_colitis_timecourse_IBD",
                "timepoint_numeric": int(sample_times[sample_id]),
            }
        )
    if not keep_columns:
        raise ValueError("No Time Course IBD columns matched GSE148794 metadata")

    counts = table.loc[:, keep_columns].to_numpy(dtype=np.float32).T
    metadata = pd.DataFrame(rows).set_index("cell_id")
    metadata["timepoint_numeric"] = metadata["timepoint_numeric"].astype(int)
    metadata.attrs["duplicated_genes"] = duplicated_genes
    return counts, genes, metadata


def prepare(accession: str, source_dir: Path, output_dir: Path, n_genes: int) -> dict:
    defaults = DEFAULTS[accession]
    if accession == "GSE148794":
        counts, gene_names, metadata = parse_gse148794(source_dir)
    elif accession == "GSE149622":
        counts, gene_names, metadata = parse_gse149622(source_dir)
    else:
        raise ValueError(f"Unsupported accession: {accession}")

    if sparse.issparse(counts):
        log_counts = normalize_log_sparse(counts)
        raw_nonzero = int(counts.nnz)
    else:
        log_counts = normalize_log_dense(counts)
        raw_nonzero = int(np.count_nonzero(counts))
    matrix, selected_genes, fit_record = fit_select_scale(
        log_counts,
        metadata,
        gene_names,
        defaults["train_times"],
        n_genes,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "expression_matrix.npy", matrix)
    metadata.to_csv(output_dir / "metadata.csv")
    (output_dir / "gene_names.txt").write_text(
        "\n".join(selected_genes) + "\n", encoding="utf-8"
    )

    counts_by_time = Counter(metadata["timepoint_numeric"].astype(int).tolist())
    cells_by_sample = Counter(metadata["sample_id"].astype(str).tolist())
    input_hashes = {
        path.name: sha256(path)
        for path in sorted(source_dir.iterdir())
        if path.is_file() and not path.name.endswith(".status")
    }
    qc = {
        "record_type": "external_candidate_bundle_qc",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "accession": accession,
        "result_tag": defaults["result_tag"],
        "source_dir": str(source_dir),
        "input_sha256": input_hashes,
        "input_shape_gene_by_cell_or_cell_by_gene": {
            "n_cells": int(matrix.shape[0]),
            "n_genes_after_selection": int(matrix.shape[1]),
            "raw_nonzero_entries": raw_nonzero,
        },
        "timepoints": sorted(int(t) for t in counts_by_time),
        "cells_per_timepoint": {
            str(k): int(v) for k, v in sorted(counts_by_time.items())
        },
        "cells_per_sample": {
            str(k): int(v) for k, v in sorted(cells_by_sample.items())
        },
        "train_timepoints": defaults["train_times"],
        "holdout_timepoints": defaults["holdout_times"],
        "secondary_holdout_timepoints": defaults["secondary_times"],
        "fit_record": fit_record,
        "gene_names_sha256": sha256(output_dir / "gene_names.txt"),
        "metadata_sha256": sha256(output_dir / "metadata.csv"),
        "expression_matrix_sha256": sha256(output_dir / "expression_matrix.npy"),
        "status": (
            "PASS_FOR_SMOKE"
            if set(defaults["train_times"] + defaults["holdout_times"]).issubset(
                set(counts_by_time)
            )
            else "FAIL_MISSING_TIMEPOINT"
        ),
        "replicate_note": (
            "GEO sample/library groups are not independent animal replicates; "
            "the source study pooled biological material before profiling. "
            "Do not report cell-level or library-level intervals as animal-level "
            "biological uncertainty."
            if accession == "GSE148794"
            else "one pooled sample per timepoint; no biological replicate inference"
        ),
        "leakage_control": (
            "library normalization per cell; HVG selection and scaling fit on "
            f"training timepoints only ({defaults['train_times']})"
        ),
    }
    report_path = REPORT_DIR / f"external_{accession.lower()}_qc.json"
    report_path.write_text(
        json.dumps(qc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return qc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accession", choices=sorted(DEFAULTS), required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Defaults to data/external_geo/<accession>",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Defaults to data/leworld_input_<accession lower>",
    )
    parser.add_argument("--n-genes", type=int, default=2991)
    args = parser.parse_args()
    source_dir = args.source_dir or ROOT / "data" / "external_geo" / args.accession
    output_dir = args.out or ROOT / "data" / f"leworld_input_{args.accession.lower()}"
    qc = prepare(args.accession, source_dir, output_dir, args.n_genes)
    print(json.dumps(qc, indent=2, ensure_ascii=False))
    return 0 if qc["status"] == "PASS_FOR_SMOKE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
