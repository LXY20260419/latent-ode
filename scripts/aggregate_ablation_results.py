#!/usr/bin/env python3
"""Aggregate 3-seed ODE ablation JSON results into reports."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "artifacts" / "results" / "ode_benchmark"
REPORT_DIR = ROOT / "artifacts" / "reports"

METRICS = [
    "time_holdout_mean_corr",
    "time_extrapolation_mean_corr",
    "time_extrapolation_pair_27_to_36_corr",
    "time_extrapolation_pair_27_to_48_corr",
    "tf_direction_consistency_overall",
    "pathway_activation_order_overall",
    "marker_rank_spearman_overall",
]

SEEDS = [42, 123, 2026]


def _load_metric(path: Path, key: str) -> float | None:
    data = json.loads(path.read_text())
    if key == "time_holdout_mean_corr":
        return data.get("time_holdout", {}).get("mean_correlation")
    if key == "time_extrapolation_mean_corr":
        return data.get("time_extrapolation", {}).get("mean_correlation")
    if key.startswith("time_extrapolation_pair_"):
        tp = key.split("_")[-2]  # 27_to_36 -> 36
        target = int(tp)
        pairs = data.get("time_extrapolation", {}).get("pairs", {})
        entry = pairs.get(str(target)) or pairs.get(target)
        if entry:
            return entry.get("corr")
        return None
    if key == "tf_direction_consistency_overall":
        return data.get("tf_direction_consistency", {}).get("overall")
    if key == "pathway_activation_order_overall":
        return data.get("pathway_activation_order", {}).get("overall")
    if key == "marker_rank_spearman_overall":
        return data.get("marker_rank_spearman", {}).get("overall")
    return None


def aggregate(group: str, pattern: str) -> dict:
    per_seed = {}
    for seed in SEEDS:
        path = RESULT_DIR / pattern.format(seed=seed)
        if not path.exists():
            raise FileNotFoundError(path)
        per_seed[str(seed)] = {m: _load_metric(path, m) for m in METRICS}
    agg3 = {}
    for m in METRICS:
        vals = [per_seed[str(s)][m] for s in SEEDS if per_seed[str(s)][m] is not None]
        agg3[m] = {"mean": statistics.mean(vals), "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0}
    payload = {"group": group, "seeds": SEEDS, "per_seed": per_seed, "agg3": agg3}
    out = REPORT_DIR / f"stage3_ablation_{group}_agg.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    axes = [
        ("pathway0", "validation_results_ode_ablation_pathway0_seed{seed}.json"),
        ("pathwaydelta0", "validation_results_ode_ablation_pathwaydelta0_seed{seed}.json"),
        ("nopriors", "validation_results_ode_ablation_nopriors_seed{seed}.json"),
    ]
    summary = {}
    for group, pattern in axes:
        try:
            summary[group] = aggregate(group, pattern)
            print(f"ok {group}")
        except FileNotFoundError as exc:
            print(f"skip {group}: {exc}")

    ranking_path = REPORT_DIR / "stage3_ablation_contribution_ranking.json"
    full = aggregate("full_reference", "validation_results_ode_seed{seed}.json") if all(
        (RESULT_DIR / f"validation_results_ode_seed{s}.json").exists() for s in SEEDS
    ) else None
    ranking = {
        "full_ode_extrapolation_mean": full["agg3"]["time_extrapolation_mean_corr"] if full else None,
        "axes": {
            k: v["agg3"]["time_extrapolation_mean_corr"]
            for k, v in summary.items()
        },
    }
    ranking_path.write_text(json.dumps(ranking, indent=2), encoding="utf-8")
    print(json.dumps(ranking, indent=2))


if __name__ == "__main__":
    main()
