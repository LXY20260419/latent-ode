#!/usr/bin/env python3
"""Cell-bootstrap uncertainty for the population-level extrapolation endpoint.

Reviewers objected that three seeds give almost no statistical power. That
objection is about *training* variability. A separate and cheaper question is
how much of the endpoint value is *sampling* noise in the estimated population
means, which can be quantified directly on the frozen matrix without training.

For each hold-out horizon this tool:

1. resamples cells within the 27 h source timepoint and within the target
   timepoint with replacement;
2. recomputes the population means and the across-gene Pearson correlation;
3. reports a percentile confidence interval.

It also reports the same interval for a deterministic reference profile, to
show that the interval width is dominated by the number of cells, not by the
predictor. This is a read-only analysis of ``data/leworld_input``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data" / "leworld_input"
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "reports" / "population_endpoint_bootstrap.json"
)
RECOVERED_DIR = ROOT / "artifacts" / "results" / "primary_internal_recovered"
TRAIN_TIMES = (0, 1, 4, 16, 27)
HORIZONS = (36, 48)
SOURCE_TIME = 27


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("empty sample")
    if probability <= 0:
        return sorted_values[0]
    if probability >= 1:
        return sorted_values[-1]
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


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


def load_matrix(data_dir: Path):
    import numpy as np

    matrix = np.load(data_dir / "expression_matrix.npy")
    times = []
    with (data_dir / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            times.append(int(float(row["timepoint_numeric"])))
    if len(times) != matrix.shape[0]:
        raise ValueError("metadata rows do not match the expression matrix")
    return matrix, np.asarray(times)


def bootstrap_horizon(
    matrix,
    times,
    *,
    target: int,
    replicates: int,
    seed: int,
) -> dict:
    import numpy as np

    source_index = np.flatnonzero(times == SOURCE_TIME)
    target_index = np.flatnonzero(times == target)
    if source_index.size == 0 or target_index.size == 0:
        raise ValueError(f"missing cells for {SOURCE_TIME}h -> {target}h")

    rng = np.random.default_rng(seed)
    source_matrix = matrix[source_index]
    target_matrix = matrix[target_index]
    observed = across_gene_pearson(
        source_matrix.mean(axis=0), target_matrix.mean(axis=0)
    )

    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        source_sample = source_matrix[
            rng.integers(0, source_matrix.shape[0], source_matrix.shape[0])
        ].mean(axis=0)
        target_sample = target_matrix[
            rng.integers(0, target_matrix.shape[0], target_matrix.shape[0])
        ].mean(axis=0)
        values[index] = across_gene_pearson(source_sample, target_sample)

    ordered = sorted(values.tolist())
    return {
        "from_time": SOURCE_TIME,
        "target_time": target,
        "observed_corr": observed,
        "n_source_cells": int(source_index.size),
        "n_target_cells": int(target_index.size),
        "replicates": replicates,
        "seed": seed,
        "bootstrap_mean": float(np.mean(values)),
        "bootstrap_sd": float(np.std(values, ddof=1)),
        "ci95_percentile": [
            percentile(ordered, 0.025),
            percentile(ordered, 0.975),
        ],
        "ci95_width": percentile(ordered, 0.975) - percentile(ordered, 0.025),
    }


def primary_context(result_dir: Path, tag: str) -> dict | None:
    """Descriptive comparison of the recovered model estimates to the interval.

    This is **not** a test. The recovered validation files store only
    population-level correlations, not per-cell predictions, so a paired
    cell bootstrap of model-versus-persistence is not possible from them.
    What can be said is whether the model's point estimate sits inside or
    outside the resampling interval of the control.
    """
    values = {}
    for horizon in HORIZONS:
        per_seed = []
        for seed in (42, 123, 2026):
            path = (
                result_dir
                / f"validation_results_ode_{tag}_seed{seed}.json"
            )
            if not path.is_file():
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            pairs = payload["time_extrapolation"]["pairs"]
            per_seed.append(float(pairs[str(horizon)]["corr"]))
        values[str(horizon)] = per_seed
    return values


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--result-dir", type=Path, default=RECOVERED_DIR)
    parser.add_argument("--result-tag", default="primary_internal_recovered")
    args = parser.parse_args(argv)

    try:
        matrix, times = load_matrix(args.data_dir)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    horizons = {}
    for target in HORIZONS:
        try:
            horizons[str(target)] = bootstrap_horizon(
                matrix,
                times,
                target=target,
                replicates=args.replicates,
                seed=args.seed + target,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 2

    context = primary_context(args.result_dir, args.result_tag)
    if context is not None:
        for target in HORIZONS:
            block = horizons[str(target)]
            per_seed = context[str(target)]
            low, high = block["ci95_percentile"]
            mean = sum(per_seed) / len(per_seed)
            block["primary_arm_per_seed"] = per_seed
            block["primary_arm_mean"] = mean
            block["primary_arm_inside_control_ci"] = low <= mean <= high
            block["primary_arm_margin_vs_control"] = (
                mean - block["observed_corr"]
            )

    payload = {
        "status": "completed",
        "method": "cell_level_bootstrap_population_mean_across_gene_pearson",
        "metric": "population_mean_across_gene_pearson_r",
        "source_time": SOURCE_TIME,
        "train_times": list(TRAIN_TIMES),
        "replicates": args.replicates,
        "rng_seed": args.seed,
        "horizons": horizons,
        "matrix_shape": list(matrix.shape),
        "source_files": {
            "expression_matrix": str(
                (args.data_dir / "expression_matrix.npy").relative_to(ROOT)
            )
            if (args.data_dir / "expression_matrix.npy").is_relative_to(ROOT)
            else str(args.data_dir / "expression_matrix.npy"),
            "metadata_sha256": file_sha256(args.data_dir / "metadata.csv"),
            "gene_manifest_sha256": file_sha256(
                args.data_dir / "gene_names.txt"
            ),
        },
        "notes": [
            "This quantifies sampling noise in the population means, not "
            "training variability across seeds.",
            "It is not a substitute for the tagged-run uncertainty artifact "
            "required by the submission gate.",
            "primary_arm_* fields are descriptive only. The recovered "
            "validation files store population-level correlations, not "
            "per-cell predictions, so no paired cell bootstrap of model "
            "versus control can be computed from them.",
            "Comparing the model's margin against this interval is a sanity "
            "check on how much of the reported difference could be specific "
            "to this cell sample, not a hypothesis test.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"saved: {args.output}")
    for key, block in horizons.items():
        print(
            f"  {SOURCE_TIME}h -> {key}h: r={block['observed_corr']:.4f} "
            f"95% CI [{block['ci95_percentile'][0]:.4f}, "
            f"{block['ci95_percentile'][1]:.4f}] "
            f"width={block['ci95_width']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
