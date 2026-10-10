#!/usr/bin/env python3
"""Plot the cross-dataset horizon-decay aggregate with all seed panels.

The manuscript's gap-decay figure is generated per training run (single
seed). This tool renders the aggregated view from the committed numeric table
(``horizon_decay_table.json``): ODE, GRU and deterministic persistence per
dataset, with per-seed ODE/GRU points, so the figure matches the numbers a
reviewer can re-derive.

All horizons are drawn, including ties and losses; nothing is filtered. The
figure is descriptive and is not a pooled fit.

Output: ``artifacts/reports/horizon_decay_aggregate.png``.
Read-only with respect to all result artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "artifacts" / "reports"
RESULT_DIR = ROOT / "artifacts" / "results" / "ode_benchmark"

LAYER_SPECS = {
    "internal_frozen": {"tag": "primary_internal_freeze", "color": "#444444"},
    "gse148794": {"tag": "ext_gse148794_v2", "color": "#A23B72"},
    "gse149622": {"tag": "ext_gse149622_v2", "color": "#2E86AB"},
    "gse305979": {
        "tag": "ext_gse305979_control_v2",
        "color": "#F18F01",
    },
}


def load_per_seed(tag: str, model: str, seed: int) -> dict:
    path = RESULT_DIR / f"validation_results_{model}_{tag}_seed{seed}.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("time_extrapolation", {}).get("pairs") or {}
    return {str(h): float(row["corr"]) for h, row in pairs.items()}


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        type=Path,
        default=REPORT_DIR / "horizon_decay_table.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "horizon_decay_aggregate.png",
    )
    args = parser.parse_args()

    table = json.loads(args.table.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.6), sharey=True)
    labels = {
        "internal_frozen": "GSE151658 (internal, frozen)",
        "gse148794": "GSE148794 (external)",
        "gse149622": "GSE149622 (external)",
        "gse305979": "GSE305979 (human, external)",
    }

    for ax, (layer_key, spec) in zip(axes, LAYER_SPECS.items()):
        rows = [r for r in table["rows"] if r["layer"] == layer_key]
        rows.sort(key=lambda r: r["target_time"])
        xs = [r["gap"] for r in rows]
        ax.plot(
            xs,
            [r["ode_mean"] for r in rows],
            "o-",
            color="#A23B72",
            label="ODE",
            linewidth=2,
            markersize=7,
        )
        ax.plot(
            xs,
            [r["gru_mean"] for r in rows],
            "s-",
            color="#2E86AB",
            label="GRU",
            linewidth=2,
            markersize=7,
        )
        ax.plot(
            xs,
            [r["persistence"] for r in rows],
            "^--",
            color="#666666",
            label="persistence",
            linewidth=1.5,
            markersize=7,
        )
        # per-seed ODE points
        for horizon_key, r in zip(
            [str(int(r["target_time"])) for r in rows], rows
        ):
            for seed in (42, 123, 2026, 7, 13):
                pair = load_per_seed(spec["tag"], "ode", seed)
                if horizon_key in pair:
                    ax.scatter(
                        [r["gap"]],
                        [pair[horizon_key]],
                        s=14,
                        color="#A23B72",
                        alpha=0.35,
                        zorder=1,
                    )
        ax.set_title(labels[layer_key], fontsize=11)
        ax.set_xlabel(f"extrapolation gap ({rows[0]['time_unit']})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("population-mean Pearson r")
    axes[0].legend(fontsize=9, loc="best")
    fig.suptitle(
        "Extrapolation decay by dataset (all arms, all horizons retained)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
