#!/usr/bin/env python3
"""Aggregate Phase-2 external runs into a single auditable report.

Reads only tagged external artifacts; never touches the frozen internal
result, the manuscript, reviewer-control aggregates, or the submission gate.

Outputs:

* ``artifacts/reports/external_phase2_aggregate_v1.json``  — per-seed and
  per-horizon metrics, seed mean/SD, exact 27-resample seed-bootstrap interval;
* ``artifacts/reports/external_phase2_report_v1.md``       — short human
  summary that always names negative/failed endpoints instead of hiding them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "artifacts" / "results" / "ode_benchmark"
REPORT_DIR = ROOT / "artifacts" / "reports"

PROTOCOL = REPORT_DIR / "external_phase2_protocol_v1.json"

ARM_SPECS = {
    "GSE148794": {
        "tag": "ext_gse148794_v2",
        "train_source_time": 9,
        "primary_horizon": 12,
        "secondary_horizon": 15,
    },
    "GSE149622": {
        "tag": "ext_gse149622_v2",
        "train_source_time": 3,
        "primary_horizon": 4,
        "secondary_horizon": None,
    },
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def load_seed(tag: str, model: str, seed: int) -> dict:
    path = RESULT_DIR / f"validation_results_{model}_{tag}_seed{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    pair_block = payload.get("time_extrapolation", {}).get("pairs")
    if not isinstance(pair_block, dict):
        raise ValueError(f"{path}: missing time_extrapolation.pairs")
    pairs = {}
    for horizon, row in pair_block.items():
        pairs[str(horizon)] = {
            "from_time": int(row["from_time"]),
            "corr": float(row["corr"]),
            "predict_steps": int(row.get("predict_steps", 1)),
        }
    holdout = payload.get("time_holdout", {}).get("mean_correlation")
    epochs = payload.get("run_metadata", {}).get("config", {}).get("epochs")
    history_path = RESULT_DIR / f"training_history_{model}_{tag}_seed{seed}.json"
    actual_epochs = None
    history_record = None
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        actual_epochs = len(history.get("train", []))
        history_record = file_record(history_path)
    return {
        "file": file_record(path),
        "training_history": history_record,
        "pairs": pairs,
        "holdout_recon_corr": float(holdout) if holdout is not None else None,
        "epoch_cap": epochs,
        "actual_epochs": actual_epochs,
    }


def exact_seed_bootstrap(values: list[float]) -> dict:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    means = np.asarray(
        [arr[list(index)].mean() for index in product(range(len(arr)), repeat=len(arr))],
        dtype=float,
    )
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def load_persistence_report() -> dict | None:
    path = REPORT_DIR / "external_persistence_v1.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_markdown(aggregate: dict) -> str:
    lines = [
        "# External Phase-2 production report (3 seeds)",
        "",
        f"Generated: {aggregate['created_at']}",
        "",
        "Protocol: `artifacts/reports/external_phase2_protocol_v1.json` (locked "
        "before any 100-epoch external run was inspected).",
        "",
        "All values are `population_mean_across_gene_pearson_r` on the same "
        "2,991-gene panel and train-only preprocessing. External results are "
        "stress tests and do not modify the frozen internal primary result.",
        "",
    ]
    for accession, block in aggregate["datasets"].items():
        lines.append(f"## {accession}")
        lines.append("")
        lines.append(f"- result tag: `{block['result_tag']}`")
        lines.append(
            f"- train -> holdout: {block['train_timepoints']} -> "
            f"{block['holdout_timepoints']}"
        )
        lines.append(f"- replicate note: {block['replicate_note']}")
        lines.append("")
        lines.append("| endpoint | model | per-seed | mean | SD | 27-resample CI95 |")
        lines.append("|---|---|---|---:|---:|---|")
        for horizon, horizon_block in block["horizons"].items():
            for model in ("ode", "gru"):
                arm = horizon_block[model]
                per_seed = ", ".join(
                    f"{value:+.4f}" for value in arm["per_seed"].values()
                )
                lines.append(
                    f"| {horizon} | {model.upper()} | {per_seed} | "
                    f"{arm['summary']['mean']:+.4f} | {arm['summary']['sd']:.4f} | "
                        f"[{arm['summary']['ci95_low']:+.4f}, "
                        f"{arm['summary']['ci95_high']:+.4f}] |"
                    )
            if horizon_block.get("persistence") is not None:
                persistence = horizon_block["persistence"]
                if persistence.get("status") == "completed":
                    lines.append(
                        f"| {horizon} | persistence (deterministic) | fixed | "
                        f"{persistence['corr']:+.4f} | — | — |"
                    )
                else:
                    lines.append(
                        f"| {horizon} | persistence | n/a | n/a | n/a | "
                        f"{persistence.get('status')} |"
                    )
        lines.append("")
        lines.append("Actual completed epochs (the locked 100-epoch setting is a cap; early stopping is allowed):")
        lines.append("")
        for model in ("ode", "gru"):
            values = block["actual_epochs"][model]
            lines.append(
                f"- {model.upper()}: "
                + ", ".join(f"seed {seed}: {count}" for seed, count in values.items())
            )
        lines.append("")
    lines.append("## Contrasts")
    lines.append("")
    lines.append(
        "| dataset | endpoint | ODE - GRU (mean) | ODE - persistence (mean) |"
    )
    lines.append("|---|---|---:|---:|")
    for accession, block in aggregate["datasets"].items():
        for horizon, horizon_block in block["horizons"].items():
            ode = horizon_block["ode"]["summary"]["mean"]
            gru = horizon_block["gru"]["summary"]["mean"]
            persistence = horizon_block.get("persistence")
            persistence_value = (
                persistence["corr"]
                if persistence and persistence.get("status") == "completed"
                else None
            )
            lines.append(
                f"| {accession} | {horizon} | {ode - gru:+.4f} | "
                + (f"{ode - persistence_value:+.4f}" if persistence_value is not None else "n/a")
                + " |"
            )
    lines.append("")
    lines.append("## Failure-boundary statement")
    lines.append("")
    lines.append(
        "All endpoints above are reported regardless of sign. Endpoints where "
        "ODE does not exceed matched GRU or persistence are retained as "
        "boundary results and must not be filtered out of the manuscript or "
        "supplement."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "external_phase2_aggregate_v1.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=REPORT_DIR / "external_phase2_report_v1.md",
    )
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if len(seeds) < 2:
        raise SystemExit("at least two seeds are required for uncertainty")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol_datasets = {row["accession"]: row for row in protocol["datasets"]}

    datasets_block = {}
    for accession, spec in ARM_SPECS.items():
        rows = {
            model: {str(seed): load_seed(spec["tag"], model, seed) for seed in seeds}
            for model in ("ode", "gru")
        }
        horizons = [str(spec["primary_horizon"])]
        if spec["secondary_horizon"] is not None:
            horizons.append(str(spec["secondary_horizon"]))
        horizons_block = {}
        for horizon in horizons:
            source_time = spec["train_source_time"]
            block = {}
            for model in ("ode", "gru"):
                per_seed = {}
                for seed in seeds:
                    pair = rows[model][str(seed)]["pairs"].get(horizon)
                    if pair is None:
                        raise ValueError(
                            f"{accession} {spec['tag']} {model} seed {seed}: "
                            f"missing horizon {horizon}"
                        )
                    per_seed[str(seed)] = pair["corr"]
                block[model] = {
                    "per_seed": per_seed,
                    "summary": exact_seed_bootstrap(list(per_seed.values())),
                }
            persistence = None
            persistence_report = load_persistence_report()
            if persistence_report and accession in persistence_report.get("datasets", {}):
                entry = persistence_report["datasets"][accession]["horizons"].get(horizon)
                if isinstance(entry, dict):
                    persistence = {
                        "status": "completed",
                        "corr": float(entry["corr"]),
                        "deterministic": bool(entry.get("deterministic", True)),
                        "n_source": entry.get("n_source"),
                        "n_target": entry.get("n_target"),
                    }
            block["persistence"] = persistence
            block["source_time"] = source_time
            horizons_block[horizon] = block

        datasets_block[accession] = {
            "result_tag": spec["tag"],
            "train_timepoints": protocol_datasets[accession]["train_timepoints"],
            "holdout_timepoints": protocol_datasets[accession]["holdout_timepoints"],
            "replicate_note": protocol_datasets[accession]["replicate_note"],
            "actual_epochs": {
                model: {
                    str(seed): rows[model][str(seed)]["actual_epochs"]
                    for seed in seeds
                }
                for model in ("ode", "gru")
            },
            "horizons": horizons_block,
            "seed_artifacts": {
                model: {
                    str(seed): rows[model][str(seed)]["file"] for seed in seeds
                }
                for model in ("ode", "gru")
            },
        }

    aggregate = {
        "record_type": "external_phase2_aggregate",
        "created_at": utcnow(),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "seeds": seeds,
        "successful_runs": {
            accession: {
                model: [
                    seed
                    for seed in seeds
                    if load_seed(ARM_SPECS[accession]["tag"], model, seed)["epoch_cap"]
                    is not None
                ]
                for model in ("ode", "gru")
            }
            for accession in ARM_SPECS
        },
        "datasets": datasets_block,
    }
    args.output.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(build_markdown(aggregate), encoding="utf-8")
    print(f"saved: {args.output}")
    print(f"saved: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
