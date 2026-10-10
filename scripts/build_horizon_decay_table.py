#!/usr/bin/env python3
"""Build the cross-dataset horizon-decay table (plan §12.8, numeric form).

The manuscript's gap-decay figure is a per-run PNG; there is no aggregated
numeric artifact that shows how each arm decays with the extrapolation gap on
every dataset at once. This tool reads the committed validation artifacts and
emits one row per (dataset, source time, horizon) with the ODE, GRU and
persistence population-mean correlations, the gap in native time units and in
training-grid steps, plus the paired ODE-minus-GRU difference where seed
pairs exist.

Rules:

* internal layer: the frozen rerun artifacts backing the 0.4516 composite
  (seeds 42/123/2026) plus the drift report's persistence values;
* external layers: the tagged phase-2/3 artifacts (five seeds for the two
  mouse candidates and three for GSE305979) plus the deterministic external
  persistence report;
* GRU is reported next to ODE even where it wins or ties; nothing is dropped;
* the "steps" column is ``predict_steps`` when the artifact records it
  (number of ODE integration steps / rollout steps to the target), otherwise
  the gap in time units.

Outputs ``artifacts/reports/horizon_decay_table.json`` and
``horizon_decay_table.md``. Read-only with respect to all result artifacts.
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "artifacts" / "results" / "ode_benchmark"
REPORT_DIR = ROOT / "artifacts" / "reports"

LAYERS = [
    {
        "key": "internal_frozen",
        "label": "GSE151658 (frozen rerun)",
        "tag": "primary_internal_freeze",
        "seeds": [42, 123, 2026],
        "source_time": 27,
        "time_unit": "h",
    },
    {
        "key": "gse148794",
        "label": "GSE148794",
        "tag": "ext_gse148794_v2",
        "seeds": [42, 123, 2026, 7, 13],
        "source_time": 9,
        "time_unit": "d",
    },
    {
        "key": "gse149622",
        "label": "GSE149622",
        "tag": "ext_gse149622_v2",
        "seeds": [42, 123, 2026, 7, 13],
        "source_time": 3,
        "time_unit": "d",
    },
    {
        "key": "gse305979",
        "label": "GSE305979 (human CD34+ HSPC, DMSO)",
        "tag": "ext_gse305979_control_v2",
        "seeds": [42, 123, 2026],
        "source_time": 2,
        "time_unit": "d",
    },
]

PERSISTENCE = {
    "internal_frozen": {
        "path": "artifacts/reports/composite_drift_report.json",
        "kind": "drift",
    },
    "gse148794": {
        "path": "artifacts/reports/external_persistence_v1.json",
        "kind": "external",
        "accession": "GSE148794",
    },
    "gse149622": {
        "path": "artifacts/reports/external_persistence_v1.json",
        "kind": "external",
        "accession": "GSE149622",
    },
    "gse305979": {
        "path": "artifacts/reports/external_persistence_v1.json",
        "kind": "external",
        "accession": "GSE305979",
    },
}


def load_seed_pairs(tag: str, model: str, seed: int) -> dict:
    path = RESULT_DIR / f"validation_results_{model}_{tag}_seed{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("time_extrapolation", {}).get("pairs") or {}
    return {
        str(horizon): {
            "corr": float(row["corr"]),
            "from_time": int(row["from_time"]),
            "predict_steps": int(row.get("predict_steps", 1)),
        }
        for horizon, row in pairs.items()
    }


def persistence_values(layer_key: str) -> dict:
    spec = PERSISTENCE[layer_key]
    payload = json.loads((ROOT / spec["path"]).read_text(encoding="utf-8"))
    if spec["kind"] == "drift":
        horizons = payload["endpoint_aligned_contrasts"]["ode_vs_persistence"][
            "horizons"
        ]
        return {
            str(h): {"corr": float(v["right_mean"])} for h, v in horizons.items()
        }
    block = payload["datasets"][spec["accession"]]["horizons"]
    return {
        str(h): {"corr": float(v["corr"])} for h, v in block.items()
    }


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for layer in LAYERS:
        tag = layer["tag"]
        seeds = layer["seeds"]
        source_time = layer["source_time"]
        persistence = persistence_values(layer["key"])

        ode_by_horizon: dict[str, dict] = {}
        gru_by_horizon: dict[str, dict] = {}
        for horizon in sorted(
            set(
                h
                for seed in seeds
                for h in load_seed_pairs(tag, "ode", seed).keys()
            ),
            key=lambda x: float(x),
        ):
            ode_seed_values: dict[str, float] = {}
            gru_seed_values: dict[str, float] = {}
            from_time = None
            predict_steps = None
            for seed in seeds:
                ode = load_seed_pairs(tag, "ode", seed).get(horizon)
                gru = load_seed_pairs(tag, "gru", seed).get(horizon)
                if ode is not None:
                    ode_seed_values[str(seed)] = ode["corr"]
                    from_time = ode["from_time"]
                    predict_steps = ode["predict_steps"]
                if gru is not None:
                    gru_seed_values[str(seed)] = gru["corr"]
            ode_mean = statistics.fmean(ode_seed_values.values())
            gru_mean = statistics.fmean(gru_seed_values.values())
            diffs = [
                ode_seed_values[s] - gru_seed_values[s]
                for s in ode_seed_values
                if s in gru_seed_values
            ]
            ode_by_horizon[horizon] = {
                "mean": ode_mean,
                "sd": (
                    statistics.stdev(ode_seed_values.values())
                    if len(ode_seed_values) > 1
                    else 0.0
                ),
                "per_seed": ode_seed_values,
                "from_time": from_time,
                "predict_steps": predict_steps,
            }
            gru_by_horizon[horizon] = {
                "mean": gru_mean,
                "sd": (
                    statistics.stdev(gru_seed_values.values())
                    if len(gru_seed_values) > 1
                    else 0.0
                ),
                "per_seed": gru_seed_values,
                "diff_mean_ode_minus_gru": statistics.fmean(diffs) if diffs else None,
            }

        for horizon, ode in ode_by_horizon.items():
            gru = gru_by_horizon[horizon]
            target = float(horizon)
            gap = target - float(ode["from_time"]) if ode["from_time"] is not None else None
            persistence_row = persistence.get(horizon)
            rows.append(
                {
                    "layer": layer["key"],
                    "label": layer["label"],
                    "source_time": ode["from_time"],
                    "target_time": target,
                    "gap": gap,
                    "predict_steps": ode["predict_steps"],
                    "time_unit": layer["time_unit"],
                    "n_seeds": len(ode["per_seed"]),
                    "ode_mean": ode["mean"],
                    "ode_sd": ode["sd"],
                    "gru_mean": gru["mean"],
                    "gru_sd": gru["sd"],
                    "ode_minus_gru": gru["diff_mean_ode_minus_gru"],
                    "persistence": (
                        persistence_row["corr"] if persistence_row else None
                    ),
                    "ode_minus_persistence": (
                        ode["mean"] - persistence_row["corr"]
                        if persistence_row
                        else None
                    ),
                }
            )
    return rows


def build_markdown(rows: list[dict], created_at: str) -> str:
    lines = [
        "# Cross-dataset horizon-decay table",
        "",
        f"Generated: {created_at}",
        "",
        "Population-mean across-gene Pearson r at each extrapolation horizon,",
        "grouped by dataset. `gap` is the extrapolation distance in the",
        "dataset's native time unit; `steps` is the recorded predict_steps",
        "(ODE integration / rollout steps). Negative GRU values and horizon",
        "reversals are reported as-is: no arm or horizon is filtered out.",
        "`persistence` is the deterministic no-training source-time carry-",
        "forward baseline on the same endpoint.",
        "",
        "| dataset | source->target | gap | steps | n seeds | ODE mean ± SD | GRU mean ± SD | ODE−GRU | persistence | ODE−persistence |",
        "|---|---|---:|---:|---:|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['source_time']}->{row['target_time']:g} | "
            f"{row['gap']:g} {row['time_unit']} | {row['predict_steps']} | "
            f"{row['n_seeds']} | "
            f"{row['ode_mean']:+.4f} ± {row['ode_sd']:.4f} | "
            f"{row['gru_mean']:+.4f} ± {row['gru_sd']:.4f} | "
            f"{row['ode_minus_gru']:+.4f} | "
            f"{row['persistence']:+.4f} | "
            f"{row['ode_minus_persistence']:+.4f} |"
        )
    lines.append("")
    lines.append("## Reading notes")
    lines.append("")
    lines.append(
        "- Horizons differ by dataset; gaps are not pooled or compared as if "
        "they were the same experiment."
    )
    lines.append(
        "- The internal dataset drops from 0.684 (36 h) to 0.219 (48 h); "
        "GSE148794 runs the other way for every arm including persistence "
        "(0.559 -> 0.667 ODE, 0.559 -> 0.666 persistence), so there the "
        "longer-gap endpoint is intrinsically closer to the source profile. "
        "Decay is not a stable law across datasets; this table shows each "
        "shape rather than fitting a universal one."
    )
    lines.append(
        "- GSE149622 has a single holdout horizon, so its decay shape is a "
        "single point by design."
    )
    lines.append(
        "- GSE305979 contributes the preregistered 2->5 and 2->7 day "
        "vehicle-control horizons with three seeds; its ODE and persistence "
        "curves are close, and the table does not upgrade this stress test to "
        "a universal decay claim."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORT_DIR / "horizon_decay_table.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPORT_DIR / "horizon_decay_table.md",
    )
    args = parser.parse_args()

    created_at = datetime.now(timezone.utc).isoformat()
    rows = build_rows()
    report = {
        "record_type": "horizon_decay_table",
        "created_at": created_at,
        "metric": "population_mean_across_gene_pearson_r",
        "policy": (
            "All horizons and arms retained regardless of sign; persistence "
            "is deterministic; gaps are native time units and are not pooled "
            "across datasets."
        ),
        "rows": rows,
    }
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(build_markdown(rows, created_at), encoding="utf-8")
    print(f"saved: {args.output_json}")
    print(f"saved: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
