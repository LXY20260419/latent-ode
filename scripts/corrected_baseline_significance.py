#!/usr/bin/env python3
"""Recompute the endpoint-aligned baseline tests against the correct ODE arm.

Problem being corrected
-----------------------
``artifacts/results/baselines/extrapolation_significance.json`` stores its
baseline predictions correctly, but its ``ode_seed_results`` block holds the
``pathway_delta_weight = 0`` ablation (mean 0.4956), not the primary ODE
configuration (mean 0.4833). Every p-value in that file therefore describes a
different model than the one the manuscript claims, and the manuscript's
significance table inherits the error.

This tool re-pairs the stored baseline arrays with an explicitly chosen ODE
result namespace and reports, for every baseline:

* the p-value that the legacy file implies (ablation arm), and
* the p-value for the primary arm,

so the difference is visible instead of being silently swapped.

Known caveat, reported rather than hidden: the stored baseline arrays were
evaluated on the first 2,000 genes, while the primary endpoint uses the frozen
2,991-gene panel. This tool records that mismatch in ``caveats``; it does not
silently truncate the primary arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LEGACY_SIGNIFICANCE = (
    ROOT / "artifacts" / "results" / "baselines" / "extrapolation_significance.json"
)
RECOVERED_DIR = ROOT / "artifacts" / "results" / "primary_internal_recovered"
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "reports" / "corrected_baseline_significance.json"
)
SEEDS = (42, 123, 2026)
BASELINE_GENES = 2000
PRIMARY_GENES = 2991


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


def t2_sf(t_stat: float) -> float:
    """Exact one-sided upper-tail probability for Student's t with df = 2."""
    return 0.5 - 0.5 * t_stat / math.sqrt(2.0 + t_stat * t_stat)


def paired_one_sided(differences: list[float]) -> dict:
    n = len(differences)
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    t_stat = mean / (sd / math.sqrt(n)) if sd else (
        math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0)
    )
    if math.isinf(t_stat):
        p_value = 0.0 if t_stat > 0 else 1.0
    else:
        p_value = min(max(t2_sf(t_stat), 0.0), 1.0)
    half = 4.302653 * (sd / math.sqrt(n))
    return {
        "n": n,
        "differences": differences,
        "mean_difference": mean,
        "sd_difference": sd,
        "ci95_difference": [mean - half, mean + half],
        "t_statistic": t_stat,
        "p_one_sided_greater": p_value,
    }


def load_primary_composite(result_dir: Path, result_tag: str) -> dict:
    values = []
    for seed in SEEDS:
        path = (
            result_dir
            / f"validation_results_ode_{result_tag}_seed{seed}.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        values.append(
            float(payload["time_extrapolation"]["mean_correlation"])
        )
    return {str(seed): value for seed, value in zip(SEEDS, values)}


def build_report(
    *, legacy_path: Path, result_dir: Path, result_tag: str
) -> dict:
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    ablation_arm = {
        seed: legacy["ode_seed_results"][str(seed)]["mean_extrapolation_corr"]
        for seed in SEEDS
    }
    primary_arm = load_primary_composite(result_dir, result_tag)
    baseline_names = list(legacy["baseline_seed_results"]["42"].keys())
    baselines = {}
    for name in baseline_names:
        per_seed = {
            seed: legacy["baseline_seed_results"][str(seed)][name][
                "mean_extrapolation_corr"
            ]
            for seed in SEEDS
        }
        baselines[name] = {
            "per_seed": {str(seed): per_seed[seed] for seed in SEEDS},
            "mean": statistics.mean(per_seed.values()),
            "vs_legacy_ablation_arm": paired_one_sided(
                [ablation_arm[seed] - per_seed[seed] for seed in SEEDS]
            ),
            "vs_primary_arm": paired_one_sided(
                [primary_arm[str(seed)] - per_seed[seed] for seed in SEEDS]
            ),
        }
    return {
        "status": "completed",
        "metric": "population_mean_across_gene_pearson_r",
        "primary_result_tag": result_tag,
        "ablation_arm": {str(seed): ablation_arm[seed] for seed in SEEDS},
        "ablation_arm_mean": statistics.mean(ablation_arm.values()),
        "primary_arm": primary_arm,
        "primary_arm_mean": statistics.mean(primary_arm.values()),
        "baselines": baselines,
        "caveats": [
            "Legacy baseline arrays were scored on the first "
            f"{BASELINE_GENES} genes; the primary endpoint uses the frozen "
            f"{PRIMARY_GENES}-gene panel. Direction is unaffected but the "
            "exact p-values are not panel-matched.",
            "Three seeds are stochastic training repeats, not biological "
            "replicates; df=2 makes these tests exploratory.",
            "The disagreement between the two columns is the correction: the "
            "legacy file paired baselines with the pathway_delta_weight=0 "
            "ablation.",
        ],
        "sources": {
            "legacy_significance": {
                "path": portable_path(legacy_path),
                "sha256": file_sha256(legacy_path),
            }
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, default=LEGACY_SIGNIFICANCE)
    parser.add_argument("--result-dir", type=Path, default=RECOVERED_DIR)
    parser.add_argument("--result-tag", default="primary_internal_recovered")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    try:
        report = build_report(
            legacy_path=args.legacy,
            result_dir=args.result_dir,
            result_tag=args.result_tag,
        )
    except (OSError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"saved: {args.output}")
    print(
        f"legacy ablation arm mean {report['ablation_arm_mean']:.4f} "
        f"vs primary arm mean {report['primary_arm_mean']:.4f}"
    )
    for name, block in report["baselines"].items():
        print(
            f"  {name:22s} mean={block['mean']:.4f} "
            f"p_legacy={block['vs_legacy_ablation_arm']['p_one_sided_greater']:.5f} "
            f"p_corrected={block['vs_primary_arm']['p_one_sided_greater']:.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
