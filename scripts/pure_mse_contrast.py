#!/usr/bin/env python3
"""Contrast the pure-MSE ablation arm against the primary arm.

Both arms are read from their *tagged* three-seed validation results, so this
contrast is panel-matched (2,991 genes) and protocol-matched (same train/hold-out
hours, same entry point, same checkpoint-selection rule). The only difference is
the objective: the primary arm adds the pathway-score term and the pathway-delta
term, the pure-MSE arm sets both weights to zero.

Why this file exists: the legacy ablation matrix compared the primary
configuration only against variants *within* that matrix, at a different
absolute level, and it never contained a true "no pathway losses at all" arm
under the frozen protocol. That gap is what the reviewer asks for, and closing
it changes the conclusion, so the result is recorded explicitly rather than
left implicit.

Read-only with respect to training artifacts.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "artifacts" / "results" / "ode_benchmark"
OUTPUT = ROOT / "artifacts" / "reports" / "pure_mse_vs_primary_comparison.json"

SEEDS = (42, 123, 2026)
ARMS = {
    "primary": "primary_internal_freeze",
    "pure_mse": "pure_mse",
}


def load_arm(result_tag: str, model: str = "ode") -> dict:
    per_seed = {}
    for seed in SEEDS:
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
        per_seed[str(seed)] = {
            "composite": float(payload["time_extrapolation"]["mean_correlation"]),
            "horizons": {str(k): float(v["corr"]) for k, v in pairs.items()},
            "holdout_reconstruction": float(payload["time_holdout"]["mean_correlation"]),
            "pathway_weight": config.get("pathway_weight"),
            "pathway_delta_weight": config.get("pathway_delta_weight"),
            "n_genes": meta.get("dataset", {}).get("n_genes"),
        }
    return per_seed


def summarize(values: list[float]) -> dict:
    ci = _t_ci95(values)
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values),
        "ci95": ci,
    }


def _t_ci95(values: list[float]) -> list[float]:
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    # t_{0.975, df=2}; the client only ever uses three seeds.
    t_crit = 4.302652729911275
    half = t_crit * sd / math.sqrt(n)
    return [mean - half, mean + half]


def t2_sf(t_stat: float) -> float:
    """Exact one-sided upper-tail probability for Student's t with df = 2."""
    return 0.5 - 0.5 * t_stat / math.sqrt(2.0 + t_stat * t_stat)


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


def main() -> None:
    primary = load_arm(ARMS["primary"])
    pure = load_arm(ARMS["pure_mse"])
    primary_gru = load_arm(ARMS["primary"], model="gru")
    pure_gru = load_arm(ARMS["pure_mse"], model="gru")

    composite_primary = [primary[str(s)]["composite"] for s in SEEDS]
    composite_pure = [pure[str(s)]["composite"] for s in SEEDS]
    recon_primary = [primary[str(s)]["holdout_reconstruction"] for s in SEEDS]
    recon_pure = [pure[str(s)]["holdout_reconstruction"] for s in SEEDS]

    horizons = {}
    for horizon in ("36", "48"):
        left = [primary[str(s)]["horizons"][horizon] for s in SEEDS]
        right = [pure[str(s)]["horizons"][horizon] for s in SEEDS]
        horizons[horizon] = {
            "primary": summarize(left),
            "pure_mse": summarize(right),
            "paired_pure_minus_primary": paired(right, left),
        }

    payload = {
        "status": "completed",
        "metric": "population_mean_across_gene_pearson_r",
        "model": "ode",
        "panel_matched": True,
        "n_genes": primary["42"]["n_genes"],
        "seeds": list(SEEDS),
        "config_contrast": {
            "primary": {
                "result_tag": ARMS["primary"],
                "pathway_weight": primary["42"]["pathway_weight"],
                "pathway_delta_weight": primary["42"]["pathway_delta_weight"],
            },
            "pure_mse": {
                "result_tag": ARMS["pure_mse"],
                "pathway_weight": pure["42"]["pathway_weight"],
                "pathway_delta_weight": pure["42"]["pathway_delta_weight"],
            },
        },
        "composite": {
            "primary": summarize(composite_primary),
            "pure_mse": summarize(composite_pure),
            "paired_pure_minus_primary": paired(composite_pure, composite_primary),
        },
        "holdout_reconstruction": {
            "primary": summarize(recon_primary),
            "pure_mse": summarize(recon_pure),
            "paired_pure_minus_primary": paired(recon_pure, recon_primary),
        },
        "horizons": horizons,
        "gru_control": {
            "primary": summarize([primary_gru[str(s)]["composite"] for s in SEEDS]),
            "pure_mse": summarize([pure_gru[str(s)]["composite"] for s in SEEDS]),
            "paired_pure_minus_primary": paired(
                [pure_gru[str(s)]["composite"] for s in SEEDS],
                [primary_gru[str(s)]["composite"] for s in SEEDS],
            ),
        },
        "source_files": [
            f"artifacts/results/ode_benchmark/validation_results_{model}_{tag}_seed{seed}.json"
            for tag in (ARMS["primary"], ARMS["pure_mse"])
            for model in ("ode", "gru")
            for seed in SEEDS
        ],
        "headline_assessment": {},
        "interpretation_notes": [
            "Both arms are tagged runs of the identical frozen protocol on the frozen "
            "2,991-gene panel, so this contrast is panel-matched and protocol-matched; "
            "unlike the classical-baseline tests it carries no 2,000-vs-2,991 caveat.",
            "Three stochastic training seeds give df = 2, so these tests are exploratory "
            "and their confidence intervals are wide.",
            "The pathway-prior terms are a training-time soft constraint, not an "
            "architectural component; a null here bounds their contribution to the "
            "frozen population endpoint and does not speak to other endpoints, datasets, "
            "or mechanistic readouts.",
        ],
    }

    diff = payload["composite"]["paired_pure_minus_primary"]
    significant = diff["p_two_sided"] < 0.05
    payload["headline_assessment"] = {
        "pure_mse_minus_primary_composite": diff["diff_mean"],
        "p_two_sided": diff["p_two_sided"],
        "ci95_includes_zero": diff["ci95_difference"][0] <= 0.0 <= diff["ci95_difference"][1],
        "pathway_priors_improve_primary_endpoint": bool(
            significant and diff["diff_mean"] > 0
        ),
        "summary": (
            "The pure-MSE ablation is statistically indistinguishable from the primary "
            "arm at the composite endpoint and sits nominally above it, so the "
            "pathway-prior terms show no measurable benefit on the frozen population "
            "endpoint. The manuscript must not attribute extrapolation level or "
            "stability to the pathway losses."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")

    print("\nComposite (population extrapolation, 2,991 genes):")
    for arm in ("primary", "pure_mse"):
        block = payload["composite"][arm]
        print(
            f"  {arm:9s} mean={block['mean']:+.4f} sd={block['sd']:.4f} "
            f"ci95=[{block['ci95'][0]:+.4f}, {block['ci95'][1]:+.4f}]"
        )
    print(
        f"  pure-MSE minus primary: {diff['diff_mean']:+.4f} "
        f"(CI95 [{diff['ci95_difference'][0]:+.4f}, {diff['ci95_difference'][1]:+.4f}], "
        f"p_two_sided={diff['p_two_sided']:.3f})"
    )
    for horizon, block in payload["horizons"].items():
        d = block["paired_pure_minus_primary"]
        print(
            f"  27->{horizon}h: primary {block['primary']['mean']:+.4f} "
            f"vs pure {block['pure_mse']['mean']:+.4f} "
            f"(diff {d['diff_mean']:+.4f}, p_two_sided={d['p_two_sided']:.3f})"
        )
    gru = payload["gru_control"]
    print(
        f"  GRU control: primary {gru['primary']['mean']:+.4f} "
        f"vs pure {gru['pure_mse']['mean']:+.4f}"
    )


if __name__ == "__main__":
    main()
