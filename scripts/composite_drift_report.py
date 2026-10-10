#!/usr/bin/env python3
"""Report the numeric drift between locked claims and the tagged rerun.

Read-only. Compares the locked composite claims in the freeze manifest with the
independently rerun, uniquely tagged per-seed artifacts, and computes the
endpoint-aligned contrasts (ODE vs persistence, ODE vs GRU) on the rerun.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "artifacts" / "results" / "ode_benchmark"
MANIFEST = ROOT / "artifacts" / "reports" / "primary_internal_freeze_manifest.json"
PERSISTENCE = ROOT / "artifacts" / "results" / "baselines" / "persistence_population.json"
OUTPUT = ROOT / "artifacts" / "reports" / "composite_drift_report.json"
TAG = "primary_internal_freeze"
SEEDS = (42, 123, 2026)


def composite(model: str, seed: int) -> dict:
    payload = json.loads(
        (RESULT_DIR / f"validation_results_{model}_{TAG}_seed{seed}.json").read_text(
            encoding="utf-8"
        )
    )
    pairs = {
        key: float(row["corr"])
        for key, row in payload["time_extrapolation"]["pairs"].items()
    }
    return {"pairs": pairs, "composite": statistics.mean(pairs.values())}


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    locked = manifest["locked_claims"]
    persistence = json.loads(PERSISTENCE.read_text(encoding="utf-8"))

    rerun = {model: [composite(model, seed) for seed in SEEDS] for model in ("ode", "gru")}
    report: dict = {
        "status": "completed",
        "result_tag": TAG,
        "seeds": list(SEEDS),
        "locked_claims": locked,
        "rerun": {},
        "drift": {},
        "endpoint_aligned_contrasts": {},
        "interpretation_notes": [
            "The rerun is an independent retraining on the same frozen protocol, not a "
            "bit-identical replay: cross-host and library-version nondeterminism shifts "
            "the composite.",
            "Persisted artifacts carry no seed-level persistence array, so the "
            "ODE-vs-persistence contrast is computed against the deterministic "
            "population-persistence value.",
            "Only three stochastic seeds are available; these tests are exploratory "
            "and are not independent biological replicates.",
        ],
    }

    for model in ("ode", "gru"):
        values = [row["composite"] for row in rerun[model]]
        report["rerun"][model] = {
            "per_seed": {str(seed): row["composite"] for seed, row in zip(SEEDS, rerun[model])},
            "mean": statistics.mean(values),
            "sd": statistics.stdev(values),
            "sd_ddof": 1,
            "per_horizon_mean": {
                horizon: statistics.mean(row["pairs"][horizon] for row in rerun[model])
                for horizon in ("36", "48")
            },
        }
        locked_mean = locked[f"{model}_composite_mean"]
        report["drift"][model] = {
            "locked_mean": locked_mean,
            "rerun_mean": statistics.mean(values),
            "absolute": statistics.mean(values) - locked_mean,
            "relative": (statistics.mean(values) - locked_mean) / locked_mean,
        }

    p36 = float(persistence["pairs"]["36"]["corr"])
    p48 = float(persistence["pairs"]["48"]["corr"])
    p_composite = statistics.mean([p36, p48])
    ode_values = [row["composite"] for row in rerun["ode"]]
    gru_values = [row["composite"] for row in rerun["gru"]]

    def paired(left: list[float], right: list[float]) -> dict:
        test = ttest_rel(left, right, alternative="greater")
        differences = [a - b for a, b in zip(left, right)]
        return {
            "left_mean": statistics.mean(left),
            "right_mean": statistics.mean(right),
            "diff_mean": statistics.mean(differences),
            "diff_sd": statistics.stdev(differences),
            "t_stat": float(test.statistic),
            "p_value": float(test.pvalue),
            "note": "one-sided paired t-test (left > right), df=2, exploratory",
        }

    contrasts = {
        "ode_vs_persistence": {
            "composite": paired(ode_values, [p_composite] * len(SEEDS)),
            "persistence": {
                "36": p36,
                "48": p48,
                "composite": p_composite,
            },
            "horizons": {
                horizon: paired(
                    [row["pairs"][horizon] for row in rerun["ode"]],
                    [float(persistence["pairs"][horizon]["corr"])] * len(SEEDS),
                )
                for horizon in ("36", "48")
            },
        },
        "ode_vs_gru": {"composite": paired(ode_values, gru_values)},
    }
    report["endpoint_aligned_contrasts"] = contrasts
    report["headline_assessment"] = {
        "ode_above_persistence_composite": ode_values and statistics.mean(ode_values) > p_composite,
        "ode_vs_persistence_p": contrasts["ode_vs_persistence"]["composite"]["p_value"],
        "ode_vs_persistence_significant_at_0.05": (
            contrasts["ode_vs_persistence"]["composite"]["p_value"] < 0.05
        ),
        "ode_vs_gru_significant_at_0.05": (
            contrasts["ode_vs_gru"]["composite"]["p_value"] < 0.05
        ),
        "summary": (
            "Direction is reproduced (ODE positive, GRU negative, ODE far above GRU), "
            "but on this rerun the ODE advantage over the trivial persistence control "
            "is not significant at the composite endpoint. The manuscript narrative "
            "must not claim a significant or large persistence margin."
        ),
    }

    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)
    print(
        f"ODE rerun mean={report['rerun']['ode']['mean']:.4f} "
        f"(locked {locked['ode_composite_mean']}, drift {report['drift']['ode']['absolute']:+.4f})"
    )
    print(
        f"ODE vs persistence: diff={contrasts['ode_vs_persistence']['composite']['diff_mean']:+.4f} "
        f"p={contrasts['ode_vs_persistence']['composite']['p_value']:.4f}"
    )
    print(
        f"ODE vs GRU: diff={contrasts['ode_vs_gru']['composite']['diff_mean']:+.4f} "
        f"p={contrasts['ode_vs_gru']['composite']['p_value']:.5f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
