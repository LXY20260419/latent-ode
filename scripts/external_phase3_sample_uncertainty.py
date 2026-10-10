#!/usr/bin/env python3
"""Sample-aware uncertainty for the external population-mean endpoints.

The production endpoint is ``population_mean_across_gene_pearson_r``: the
across-gene Pearson correlation between the population mean of the predicted
held-out profile and the population mean of the true held-out profile. For a
fixed checkpoint that comparison is deterministic, so the estimable
uncertainty is on the data side. This tool resamples the data:

* cell-level bootstrap: resample cells with replacement within the source and
  target timepoints, recompute both population means, recompute the endpoint.
* sample-cluster bootstrap: resample GEO sample/library groups with
  replacement within each timepoint, then resample cells within the chosen
  groups. For GSE148794 this is the replicate-aware interval (10 library
  groups per timepoint; pooled animals, so still not independent animal
  replicates). GSE149622 has one pooled sample per timepoint, so this
  bootstrap is recorded as ``not_defined_single_pooled_sample`` instead of
  fabricating an interval.

The no-training reference here is persistence (source-time population mean
carried forward); model arms share the same target-side sampling term. The
report states this rather than pretending the interval covers training
variance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "artifacts" / "reports" / "external_phase3_sample_uncertainty.json"

ARMS = {
    "GSE148794": {
        "data_dir": "data/leworld_input_gse148794",
        "source_time": 9,
        "targets": [12, 15],
        "replicate_mode": "sample_library_groups",
    },
    "GSE149622": {
        "data_dir": "data/leworld_input_gse149622",
        "source_time": 3,
        "targets": [4],
        "replicate_mode": "single_pooled_sample",
    },
    "GSE305979": {
        "data_dir": "data/leworld_input_gse305979",
        "source_time": 2,
        "targets": [5, 7],
        "replicate_mode": "cmo_well_and_library",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def across_gene_pearson(left, right) -> float:
    import numpy as np

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    if denominator == 0.0:
        return float("nan")
    return float(left @ right) / denominator


def percentile(sorted_values, probability: float) -> float:
    if probability <= 0:
        return sorted_values[0]
    if probability >= 1:
        return sorted_values[-1]
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def load_matrix(data_dir: Path):
    import numpy as np

    matrix = np.load(data_dir / "expression_matrix.npy")
    times, samples = [], []
    with (data_dir / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            times.append(int(float(row["timepoint_numeric"])))
            samples.append(row["sample_id"])
    if len(times) != matrix.shape[0]:
        raise ValueError("metadata rows do not match the expression matrix")
    return matrix, np.asarray(times), np.asarray(samples)


def sample_mean(matrix, indices):
    import numpy as np

    return matrix[np.asarray(indices)].mean(axis=0)


def bootstrap_interval(
    matrix,
    times,
    samples,
    source_time: int,
    target_time: int,
    replicates: int,
    rng_seed: int,
    cluster: bool,
) -> dict:
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    source_idx = np.flatnonzero(times == source_time)
    target_idx = np.flatnonzero(times == target_time)
    source_groups: dict = {}
    target_groups: dict = {}
    for idx in source_idx:
        source_groups.setdefault(samples[idx], []).append(int(idx))
    for idx in target_idx:
        target_groups.setdefault(samples[idx], []).append(int(idx))
    source_keys = sorted(source_groups)
    target_keys = sorted(target_groups)

    values = []
    for _ in range(replicates):
        if cluster:
            src_chosen = rng.choice(len(source_keys), size=len(source_keys), replace=True)
            tgt_chosen = rng.choice(len(target_keys), size=len(target_keys), replace=True)
            src_pool = [idx for k in src_chosen for idx in source_groups[source_keys[k]]]
            tgt_pool = [idx for k in tgt_chosen for idx in target_groups[target_keys[k]]]
            src_take = rng.choice(src_pool, size=len(src_pool), replace=True)
            tgt_take = rng.choice(tgt_pool, size=len(tgt_pool), replace=True)
        else:
            src_take = rng.choice(source_idx, size=len(source_idx), replace=True)
            tgt_take = rng.choice(target_idx, size=len(target_idx), replace=True)
        sources = sample_mean(matrix, src_take)
        targets = sample_mean(matrix, tgt_take)
        values.append(across_gene_pearson(targets, sources))
    values = [v for v in values if not math.isnan(v)]
    values.sort()
    point = {
        "corr": across_gene_pearson(
            sample_mean(matrix, target_idx), sample_mean(matrix, source_idx)
        ),
        "n_source_cells": int(len(source_idx)),
        "n_target_cells": int(len(target_idx)),
        "n_source_groups": int(len(source_keys)),
        "n_target_groups": int(len(target_keys)),
    }
    mean = float(sum(values) / len(values)) if values else None
    sd = (
        float(
            math.sqrt(
                sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
            )
        )
        if values and len(values) > 1
        else None
    )
    return {
        **point,
        "replicates": len(values),
        "bootstrap_mean": mean,
        "bootstrap_sd": sd,
        "ci95_percentile": (
            [percentile(values, 0.025), percentile(values, 0.975)] if values else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = {
        "record_type": "external_phase3_sample_uncertainty",
        "method": "cell_level_and_sample_cluster_bootstrap_of_population_mean_endpoint",
        "metric": "population_mean_across_gene_pearson_r",
        "replicates": args.replicates,
        "note": (
            "The endpoint compares population means, so for a fixed checkpoint the "
            "resampling uncertainty is on the data side. The no-training reference "
            "used here is persistence (source-time population mean carried forward); "
            "the model arms share the same target-side sampling term, which this "
            "report states instead of pretending the interval covers training variance."
        ),
        "datasets": {},
    }

    for accession, spec in ARMS.items():
        data_dir = ROOT / spec["data_dir"]
        matrix_path = data_dir / "expression_matrix.npy"
        metadata_path = data_dir / "metadata.csv"
        if not matrix_path.is_file() or not metadata_path.is_file():
            output["datasets"][accession] = {
                "data_dir": spec["data_dir"],
                "replicate_mode": spec["replicate_mode"],
                "status": "pending_data",
                "reason": "converted bundle not present yet; rerun after conversion",
                "horizons": {},
            }
            continue
        matrix, times, samples = load_matrix(data_dir)
        block = {
            "data_dir": spec["data_dir"],
            "expression_matrix_sha256": sha256(matrix_path),
            "metadata_sha256": sha256(metadata_path),
            "replicate_mode": spec["replicate_mode"],
            "horizons": {},
        }
        for offset, target in enumerate(spec["targets"]):
            entry = {
                "from_time": spec["source_time"],
                "target_time": target,
                "cell_level": bootstrap_interval(
                    matrix,
                    times,
                    samples,
                    spec["source_time"],
                    target,
                    args.replicates,
                    rng_seed=3000 + offset,
                    cluster=False,
                ),
            }
            if spec["replicate_mode"] == "single_pooled_sample":
                entry["sample_cluster"] = {
                    "status": "not_defined_single_pooled_sample",
                    "reason": (
                        "GSE149622 has exactly one pooled RNA sample per timepoint; "
                        "resampling sample groups would either be a no-op or fabricate "
                        "biological replication that does not exist."
                    ),
                }
            else:
                entry["sample_cluster"] = bootstrap_interval(
                    matrix,
                    times,
                    samples,
                    spec["source_time"],
                    target,
                    args.replicates,
                    rng_seed=4000 + offset,
                    cluster=True,
                )
                entry["sample_cluster"]["replicate_level"] = {
                    "sample_library_groups": (
                        "GEO sample/library groups (pooled animals; not independent "
                        "animal replicates)"
                    ),
                    "cmo_well_and_library": (
                        "CMO well and library groups within a single donor; wells are "
                        "technical/library multiplexing units, not biological replicates"
                    ),
                }.get(
                    spec["replicate_mode"],
                    "GEO sample/library groups",
                )
            block["horizons"][str(target)] = entry
        output["datasets"][accession] = block

    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
