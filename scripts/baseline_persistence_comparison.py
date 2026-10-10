#!/usr/bin/env python3
"""Compare recovered primary results against the population persistence baseline.

The reviewer-critical question for this manuscript is not "does the model beat
zero?" but "does the model beat copying the 27 h population mean forward?".
That comparison has to be made on one metric, one gene panel and one set of
time pairs:

    population_mean_across_gene_pearson_r
    train times 0,1,4,16,27 -> hold-out 36 h and 48 h, source 27 h
    2,991 frozen genes

This tool is read-only with respect to primary artifacts. It never writes into
the primary result namespace; it consumes

    artifacts/results/baselines/persistence_population.json
    artifacts/results/<result-tag>/validation_results_{ode,gru}_<tag>_seed*.json

and writes a single JSON report under ``artifacts/reports/``.

Unlike the legacy ``extrapolation_significance.json`` (which silently paired
baseline predictions with the ``pathway_delta_weight=0`` ablation instead of
the primary configuration, and which scored baselines on the first 2,000
genes), this tool records the exact source files and their SHA-256 digests so
the pairing can be re-verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULT_DIR = ROOT / "artifacts" / "results"
DEFAULT_PERSISTENCE = (
    ROOT / "artifacts" / "results" / "baselines" / "persistence_population.json"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "reports" / "persistence_vs_primary_comparison.json"
)
METRIC = "population_mean_across_gene_pearson_r"
DEFAULT_SEEDS = [42, 123, 2026]
DEFAULT_HORIZONS = ["36", "48"]
DEFAULT_SOURCE_TIME = 27


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Serialize paths relative to the bundle when possible."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except (ValueError, OSError):
        return str(path)


def finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def tagged_validation_path(
    result_dir: Path, model: str, result_tag: str, seed: int
) -> Path:
    if not result_tag:
        raise ValueError("a result tag is required")
    return (
        result_dir
        / result_tag
        / f"validation_results_{model}_{result_tag}_seed{seed}.json"
    )


def load_primary_seed(path: Path, horizons: list[str], source_time: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("time_extrapolation", {}).get("pairs")
    if not isinstance(pairs, dict):
        raise ValueError(f"{path.name}: missing time_extrapolation.pairs")
    per_horizon = {}
    for horizon in horizons:
        row = pairs.get(horizon, pairs.get(int(horizon)))
        if not isinstance(row, dict) or row.get("from_time") != source_time:
            raise ValueError(
                f"{path.name}: {horizon}h pair must extrapolate from {source_time}h"
            )
        corr = row.get("corr")
        if not finite_number(corr):
            raise ValueError(f"{path.name}: {horizon}h correlation is not finite")
        per_horizon[horizon] = float(corr)
    return {"per_horizon": per_horizon}


def load_persistence(
    path: Path, horizons: list[str], source_time: int
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("metric") != METRIC:
        raise ValueError(f"{path.name}: metric is not {METRIC!r}")
    if payload.get("target_times") != [int(h) for h in horizons]:
        raise ValueError(f"{path.name}: target_times do not match {horizons}")
    pairs = payload.get("pairs")
    if not isinstance(pairs, dict):
        raise ValueError(f"{path.name}: missing pairs")
    per_horizon = {}
    for horizon in horizons:
        row = pairs.get(horizon, pairs.get(int(horizon)))
        if not isinstance(row, dict) or row.get("from_time") != source_time:
            raise ValueError(
                f"{path.name}: {horizon}h pair must extrapolate from {source_time}h"
            )
        corr = row.get("corr")
        if not finite_number(corr):
            raise ValueError(f"{path.name}: {horizon}h correlation is not finite")
        per_horizon[horizon] = float(corr)
    return {"per_horizon": per_horizon}


def t2_sf(t_stat: float) -> float:
    """Exact survival function of Student's t with df = 2: P(T > t).

    For df = 2 the CDF is ``0.5 + 0.5 * t / sqrt(2 + t**2)``, so the
    one-sided upper-tail probability is ``0.5 - 0.5 * t / sqrt(2 + t**2)``.
    This is exact, not an approximation, and is only valid for the frozen
    three-seed policy (n - 1 = 2).
    """
    return 0.5 - 0.5 * t_stat / math.sqrt(2.0 + t_stat * t_stat)


def paired_stats(differences: list[float]) -> dict:
    """Seed-level paired summary plus an exact t-based p-value.

    ``p_one_sided_greater`` is exact for the frozen three-seed policy
    (df = 2). It is reported as ``None`` for any other seed count so that a
    non-frozen design cannot silently reuse the closed form.
    """
    n = len(differences)
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences) if n > 1 else 0.0
    if sd == 0.0:
        t_stat = math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0)
    else:
        t_stat = mean / (sd / math.sqrt(n))
    if n == 3 and finite_number(t_stat):
        p_greater = min(max(t2_sf(t_stat), 0.0), 1.0)
    elif n == 3 and t_stat == math.inf:
        p_greater = 0.0
    elif n == 3 and t_stat == -math.inf:
        p_greater = 1.0
    else:
        p_greater = None
    half = 0.0
    if n > 1:
        # t(0.975, df=2) = 4.302653 for the frozen three-seed policy.
        half = 4.302653 * (sd / math.sqrt(n)) if n == 3 else 1.96 * (sd / math.sqrt(n))
    return {
        "n": n,
        "per_seed_difference": differences,
        "mean_difference": mean,
        "sd_difference": sd,
        "ci95_difference": [mean - half, mean + half],
        "t_statistic": t_stat,
        "p_one_sided_greater": p_greater,
    }


def build_report(
    *,
    result_dir: Path,
    persistence_path: Path,
    result_tag: str,
    model: str,
    seeds: list[int],
    horizons: list[str],
    source_time: int,
) -> dict:
    persistence = load_persistence(persistence_path, horizons, source_time)
    per_seed_primary = {}
    source_files = {
        "persistence": {
            "path": portable_path(persistence_path),
            "sha256": file_sha256(persistence_path),
        }
    }
    for seed in seeds:
        path = tagged_validation_path(result_dir, model, result_tag, seed)
        if not path.exists():
            raise FileNotFoundError(path)
        per_seed_primary[str(seed)] = load_primary_seed(path, horizons, source_time)
        source_files[f"{model}_{seed}"] = {
            "path": portable_path(path),
            "sha256": file_sha256(path),
        }

    horizons_report = {}
    for horizon in horizons:
        base = persistence["per_horizon"][horizon]
        values = [per_seed_primary[str(seed)]["per_horizon"][horizon] for seed in seeds]
        differences = [value - base for value in values]
        horizons_report[horizon] = {
            "from_time": source_time,
            "baseline_corr": base,
            "primary_per_seed": {str(seed): value for seed, value in zip(seeds, values)},
            "primary_mean": statistics.mean(values),
            "paired_vs_baseline": paired_stats(differences),
        }

    # Composite is the within-seed arithmetic mean of the horizon
    # correlations, matching the frozen composite_rule in the manifest.
    composite_per_seed = [
        statistics.mean(
            per_seed_primary[str(seed)]["per_horizon"][horizon]
            for horizon in horizons
        )
        for seed in seeds
    ]
    composite_baseline = statistics.mean(
        persistence["per_horizon"][horizon] for horizon in horizons
    )
    composite_values = composite_per_seed
    composite_differences = [value - composite_baseline for value in composite_values]
    return {
        "status": "completed",
        "metric": METRIC,
        "model": model,
        "result_tag": result_tag,
        "seeds": list(seeds),
        "horizons": horizons_report,
        "composite": {
            "baseline_corr": composite_baseline,
            "primary_per_seed": {
                str(seed): value for seed, value in zip(seeds, composite_values)
            },
            "primary_mean": statistics.mean(composite_values),
            "paired_vs_baseline": paired_stats(composite_differences),
        },
        "source_files": source_files,
        "notes": [
            "Baseline copies the 27 h population mean forward unchanged.",
            "Both arms use population_mean_across_gene_pearson_r on the same "
            "2,991-gene frozen panel.",
            "Seed spread is stochastic training variability, not biological "
            "replication; the baseline has no seed spread.",
            "This file supersedes the legacy extrapolation_significance.json "
            "pairing for the persistence comparison.",
        ],
    }


def write_json(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--persistence", type=Path, default=DEFAULT_PERSISTENCE)
    parser.add_argument("--result-tag", default="primary_internal_recovered")
    parser.add_argument("--model", default="ode", choices=["ode", "gru"])
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--horizons", default=",".join(DEFAULT_HORIZONS))
    parser.add_argument("--source-time", type=int, default=DEFAULT_SOURCE_TIME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    horizons = [item.strip() for item in args.horizons.split(",") if item.strip()]
    try:
        report = build_report(
            result_dir=args.result_dir,
            persistence_path=args.persistence,
            result_tag=args.result_tag,
            model=args.model,
            seeds=seeds,
            horizons=horizons,
            source_time=args.source_time,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    write_json(report, args.output)
    print(f"saved: {args.output}")
    composite = report["composite"]
    print(
        "composite: primary={:.4f} baseline={:.4f} diff={:+.4f} p(one-sided)={}".format(
            composite["primary_mean"],
            composite["baseline_corr"],
            composite["paired_vs_baseline"]["mean_difference"],
            composite["paired_vs_baseline"]["p_one_sided_greater"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
