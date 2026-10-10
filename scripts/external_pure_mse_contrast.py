#!/usr/bin/env python3
"""Contrast the external pure-MSE arm against the external full arm.

Implements the analysis plan preregistered in
``artifacts/reports/external_pure_mse_preregistration.json``: for each
accession and endpoint, the paired per-seed difference ``pure_mse - full``
computed from tagged artifacts whose recorded provenance (result_tag + seed)
matches the files they are read from. Statistics use the same exploratory
df=2 conventions as the internal ``pure_mse_contrast.py``.

Guarantees:

* both arms must exist and carry matching provenance, otherwise the tool
  aborts instead of producing a number from mismatched runs;
* every endpoint is reported regardless of sign; a nominally-better
  pure-MSE arm is surfaced as such, never hidden;
* the GRU control arm of the same runs is recorded as a secondary contrast;
* read-only with respect to training artifacts.

Output: ``artifacts/reports/external_pure_mse_contrast.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "artifacts" / "results" / "ode_benchmark"
DEFAULT_OUTPUT = ROOT / "artifacts" / "reports" / "external_pure_mse_contrast.json"

SEEDS = (42, 123, 2026)

ARMS = {
    "GSE148794": {
        "full_tag": "ext_gse148794_v2",
        "pure_mse_tag": "ext_gse148794_puremse_v1",
        "endpoints": [12, 15],
        "primary_endpoint": 12,
        "secondary_endpoint": 15,
    },
    "GSE149622": {
        "full_tag": "ext_gse149622_v2",
        "pure_mse_tag": "ext_gse149622_puremse_v1",
        "endpoints": [4],
        "primary_endpoint": 4,
        "secondary_endpoint": None,
    },
}


def _t_ci95(values: list[float]) -> list[float]:
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    # t_{0.975, df=2}; this tool only ever handles three seeds.
    t_crit = 4.302652729911275
    half = t_crit * sd / math.sqrt(n)
    return [mean - half, mean + half]


def t2_sf(t_stat: float) -> float:
    """Exact one-sided upper-tail probability for Student's t with df = 2."""
    return 0.5 - 0.5 * t_stat / math.sqrt(2.0 + t_stat * t_stat)


def summarize(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values),
        "ci95": _t_ci95(values),
    }


def paired(left: list[float], right: list[float]) -> dict:
    diffs = [a - b for a, b in zip(left, right)]
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    t_stat = mean / (sd / math.sqrt(len(diffs))) if sd else 0.0
    p_greater = min(max(t2_sf(t_stat), 0.0), 1.0)
    return {
        "n": len(diffs),
        "per_seed_difference": diffs,
        "diff_mean": mean,
        "diff_sd": sd,
        "ci95_difference": _t_ci95(diffs),
        "t_statistic": t_stat,
        "p_one_sided_left_greater": p_greater,
        "p_two_sided": min(1.0, 2 * p_greater),
        "note": "one-sided tests are exploratory: three stochastic seeds, df = 2",
    }


def load_arm(result_tag: str, seed: int, model: str = "ode") -> dict:
    path = RESULT_DIR / f"validation_results_{model}_{result_tag}_seed{seed}.json"
    if not path.is_file():
        raise SystemExit(f"missing tagged result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = payload.get("run_metadata", {})
    protocol = meta.get("protocol", {})
    config = meta.get("config", {})
    if protocol.get("seed") != seed or protocol.get("result_tag") != result_tag:
        raise SystemExit(f"{path.name}: provenance mismatch")
    pairs = payload["time_extrapolation"]["pairs"]
    return {
        "composite": float(payload["time_extrapolation"]["mean_correlation"]),
        "horizons": {str(k): float(v["corr"]) for k, v in pairs.items()},
        "holdout_reconstruction": float(payload["time_holdout"]["mean_correlation"]),
        "pathway_weight": config.get("pathway_weight"),
        "pathway_delta_weight": config.get("pathway_delta_weight"),
        "n_genes": meta.get("dataset", {}).get("n_genes"),
    }


def build() -> dict:
    phase3_path = ROOT / "artifacts" / "reports" / "external_phase3_aggregate_v2.json"
    phase3 = (
        json.loads(phase3_path.read_text(encoding="utf-8"))
        if phase3_path.is_file()
        else None
    )
    datasets: dict = {}
    source_files: list[str] = []
    for accession, spec in ARMS.items():
        full = {seed: load_arm(spec["full_tag"], seed) for seed in SEEDS}
        pure = {seed: load_arm(spec["pure_mse_tag"], seed) for seed in SEEDS}
        full_gru = {seed: load_arm(spec["full_tag"], seed, "gru") for seed in SEEDS}
        pure_gru = {
            seed: load_arm(spec["pure_mse_tag"], seed, "gru") for seed in SEEDS
        }
        for tag in (spec["full_tag"], spec["pure_mse_tag"]):
            for model in ("ode", "gru"):
                for seed in SEEDS:
                    source_files.append(
                        "artifacts/results/ode_benchmark/"
                        f"validation_results_{model}_{tag}_seed{seed}.json"
                    )

        composite_full = [full[s]["composite"] for s in SEEDS]
        composite_pure = [pure[s]["composite"] for s in SEEDS]
        endpoints = {}
        for horizon in spec["endpoints"]:
            key = str(horizon)
            left = [full[s]["horizons"][key] for s in SEEDS]
            right = [pure[s]["horizons"][key] for s in SEEDS]
            endpoints[key] = {
                "full": summarize(left),
                "pure_mse": summarize(right),
                "paired_pure_minus_full": paired(right, left),
            }

        datasets[accession] = {
            "endpoints": endpoints,
            "composite_context": {
                "full_5seed_means": (
                    {
                        key: block["ode"]["summary"]["mean"]
                        for key, block in phase3["datasets"][accession][
                            "horizons"
                        ].items()
                    }
                    if phase3
                    else None
                ),
                "note": (
                    "context only; the paired tests use the three shared "
                    "seeds, not the five-seed means"
                ),
                "full_3seed": summarize(composite_full),
                "pure_mse_3seed": summarize(composite_pure),
                "paired_pure_minus_full": paired(composite_pure, composite_full),
            },
            "holdout_reconstruction": {
                "full": summarize([full[s]["holdout_reconstruction"] for s in SEEDS]),
                "pure_mse": summarize(
                    [pure[s]["holdout_reconstruction"] for s in SEEDS]
                ),
                "paired_pure_minus_full": paired(
                    [pure[s]["holdout_reconstruction"] for s in SEEDS],
                    [full[s]["holdout_reconstruction"] for s in SEEDS],
                ),
            },
            "gru_control": {
                "full": summarize([full_gru[s]["composite"] for s in SEEDS]),
                "pure_mse": summarize([pure_gru[s]["composite"] for s in SEEDS]),
                "paired_pure_minus_full": paired(
                    [pure_gru[s]["composite"] for s in SEEDS],
                    [full_gru[s]["composite"] for s in SEEDS],
                ),
            },
            "config_contrast": {
                "full": {
                    "result_tag": spec["full_tag"],
                    "pathway_weight": full[SEEDS[0]]["pathway_weight"],
                    "pathway_delta_weight": full[SEEDS[0]]["pathway_delta_weight"],
                },
                "pure_mse": {
                    "result_tag": spec["pure_mse_tag"],
                    "pathway_weight": pure[SEEDS[0]]["pathway_weight"],
                    "pathway_delta_weight": pure[SEEDS[0]]["pathway_delta_weight"],
                },
            },
        }

    payload = {
        "record_type": "external_pure_mse_contrast",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric": "population_mean_across_gene_pearson_r",
        "model": "ode",
        "seeds": list(SEEDS),
        "preregistration": "artifacts/reports/external_pure_mse_preregistration.json",
        "datasets": datasets,
        "source_files": sorted(set(source_files)),
        "interpretation_notes": [
            "Both arms are tagged runs of the identical frozen protocol on the "
            "same 2,991-gene panel and the same train/holdout timepoints; the "
            "only difference is the pathway loss weights (0.2/0.1 vs 0.0/0.0).",
            "Three stochastic seeds give df = 2; these tests are exploratory "
            "and the intervals are wide.",
            "Every endpoint is reported regardless of sign; a nominally better "
            "pure-MSE arm is surfaced, not hidden.",
            "This contrast bounds the pathway terms' contribution to the "
            "external population endpoints only.",
        ],
    }

    # Headline per accession: the primary endpoint's paired verdict.
    headline = {}
    for accession, spec in ARMS.items():
        primary_key = str(spec["primary_endpoint"])
        d = datasets[accession]["endpoints"][primary_key][
            "paired_pure_minus_full"
        ]
        headline[accession] = {
            "primary_endpoint": primary_key,
            "pure_mse_minus_full": d["diff_mean"],
            "ci95": d["ci95_difference"],
            "p_two_sided": d["p_two_sided"],
            "ci95_includes_zero": d["ci95_difference"][0]
            <= 0.0
            <= d["ci95_difference"][1],
            "verdict": (
                "indistinguishable"
                if d["ci95_difference"][0] <= 0.0 <= d["ci95_difference"][1]
                else ("pure_mse_better" if d["diff_mean"] > 0 else "full_better")
            ),
        }
    payload["headline"] = headline
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build()
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"saved: {args.output}")
    for accession, h in payload["headline"].items():
        print(
            f"  {accession} {h['primary_endpoint']}: "
            f"{h['pure_mse_minus_full']:+.4f} "
            f"CI95 [{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}] -> {h['verdict']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
