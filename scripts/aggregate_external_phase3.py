#!/usr/bin/env python3
"""Phase-3 aggregate for external candidates with per-dataset seed panels.

Reads the tagged external validation artifacts only. Never touches the frozen
internal result, the manuscript, the reviewer-control aggregates, or the
submission gate.

Outputs:

* ``artifacts/reports/external_phase3_aggregate_v2.json`` — per-seed and
  per-horizon metrics using each accession's locked panel (five seeds for the
  two mouse candidates and three seeds for GSE305979), seed mean/SD, exact
  seed-panel bootstrap interval, persistence contrast;
* ``artifacts/reports/external_phase3_report_v2.md`` — readable summary that
  names every endpoint, including negative/null ones;
* ``artifacts/reports/external_phase3_forest.json`` and ``.png`` — descriptive
  cross-endpoint ODE-GRU forest plot with a DerSimonian-Laird I^2 summary.
  Endpoints differ by dataset/protocol/horizon, so this is explicitly
  descriptive, not a pooled claim.
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

PHASE3_SEEDS = [42, 123, 2026, 7, 13]

# Per-dataset seed panels, each locked before its results existed.
# ``external_phase3_preregistration.json`` expanded the two mouse candidates to
# five seeds; the GSE305979 protocol
# (``training_contract.seeds_after_smoke``) locks three seeds and reserves any
# 7/13 extension for a separate decision.  A single global seed list would
# therefore either drop GSE305979 (no 7/13 artifacts) or silently downgrade the
# two five-seed datasets to three seeds — and the "pass --seeds 42,123,2026 so
# the third dataset completes" workaround does exactly the latter.  Each
# dataset therefore reports on its own preregistered panel; extending a panel
# is a visible edit to this table plus the protocol, not a CLI side effect.
SEEDS_BY_DATASET = {
    "GSE148794": [42, 123, 2026, 7, 13],
    "GSE149622": [42, 123, 2026, 7, 13],
    "GSE305979": [42, 123, 2026],
}

# Where each dataset's immutable split metadata lives.  The two mouse
# candidates were preregistered in the shared phase-2 protocol; GSE305979 was
# preregistered later in its own protocol file, so reading a single shared
# document left the third dataset without train/holdout/replicate metadata.
PROTOCOL_FILES = {
    "GSE148794": "external_phase2_protocol_v1.json",
    "GSE149622": "external_phase2_protocol_v1.json",
    "GSE305979": "external_gse305979_protocol.json",
}

ARM_SPECS = {
    "GSE148794": {
        "tag": "ext_gse148794_v2",
        "train_source_time": 9,
        "targets": {12: "primary", 15: "secondary"},
    },
    "GSE149622": {
        "tag": "ext_gse149622_v2",
        "train_source_time": 3,
        "targets": {4: "primary"},
    },
    "GSE305979": {
        "tag": "ext_gse305979_control_v2",
        "train_source_time": 2,
        "targets": {5: "primary", 7: "secondary"},
    },
}

# Cross-dataset forest rows: (label, dataset, tag, model file names, source, target, kind)
FOREST_ROWS = [
    ("internal 27->36", "primary_internal_freeze", 27, 36),
    ("internal 27->48", "primary_internal_freeze", 27, 48),
    ("GSE148794 9->12", "ext_gse148794_v2", 9, 12),
    ("GSE148794 9->15", "ext_gse148794_v2", 9, 15),
    ("GSE149622 3->4", "ext_gse149622_v2", 3, 4),
    ("GSE305979 2->5", "ext_gse305979_control_v2", 2, 5),
    ("GSE305979 2->7", "ext_gse305979_control_v2", 2, 7),
]

# Historic external endpoints that only exist as pre-aggregated reports with a
# per-seed ode/gru "extrap" block. They are included in the descriptive forest
# so the boundary evidence is visible in one place; their protocols differ from
# the locked phase-2/3 panel and they are never pooled into a claim.
HISTORIC_ROWS = [
    (
        "GSE224518 3->5 (balanced)",
        REPORT_DIR / "stage2_gse224518_3to5h_balanced_agg.json",
        3,
        5,
    ),
    ("GSE224518 3->8", REPORT_DIR / "stage2_gse224518_agg.json", 3, 8),
    ("GSE226824 24->72", REPORT_DIR / "stage2_gse226824_agg.json", 24, 72),
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def seeds_for(accession: str, requested: list[int]) -> list[int]:
    """Seed panel actually used for one dataset.

    A dataset with a preregistered panel always reports on it, so a request
    that would drop one of its locked seeds raises instead of silently
    shrinking the panel.  ``--seeds`` still governs any dataset without a
    locked panel.
    """
    locked = SEEDS_BY_DATASET.get(accession)
    if locked is None:
        return list(requested)
    missing = [seed for seed in locked if seed not in requested]
    if missing:
        raise ValueError(
            f"{accession} preregisters seeds {locked}; requested panel "
            f"{requested} is missing {missing}"
        )
    return list(locked)


def protocol_split(accession: str) -> dict:
    """Split metadata for one dataset, from its own preregistration.

    Returns the train/holdout timepoints and the replicate note exactly as the
    dataset's protocol recorded them, so the aggregate never invents a split.
    """
    name = PROTOCOL_FILES.get(accession)
    if name is None:
        raise KeyError(f"no protocol file registered for {accession}")
    payload = json.loads((REPORT_DIR / name).read_text(encoding="utf-8"))
    if "datasets" in payload:  # shared phase-2 protocol: list of rows
        rows = {row["accession"]: row for row in payload["datasets"]}
        if accession not in rows:
            raise KeyError(f"{accession} absent from {name}")
        row = rows[accession]
    else:  # single-accession protocol
        row = payload
    return {
        "train_timepoints": row["train_timepoints"],
        "holdout_timepoints": row["holdout_timepoints"],
        "replicate_note": row["replicate_note"],
        "protocol_file": name,
    }


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


def exact_seed_bootstrap(values: list[float], max_exact: int = 4096) -> dict:
    """Exact enumeration when the count is small, else a deterministic subset."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    n = arr.size
    total = n ** n
    if total <= max_exact:
        means = np.asarray(
            [arr[list(index)].mean() for index in product(range(n), repeat=n)],
            dtype=float,
        )
        method = f"exact_seed_bootstrap_enumeration_{total}"
    else:
        rng = np.random.default_rng(20261002)
        draws = rng.integers(0, n, size=(max_exact, n))
        means = arr[draws].mean(axis=1)
        method = f"deterministic_subsample_bootstrap_{max_exact}_of_{total}"
    return {
        "n": int(n),
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "ci_method": method,
    }


def load_persistence_report() -> dict | None:
    path = REPORT_DIR / "external_persistence_v1.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_markdown(aggregate: dict) -> str:
    lines = [
        "# External Phase-3 report (per-dataset seed panels)",
        "",
        f"Generated: {aggregate['created_at']}",
        "",
        "Protocol: `artifacts/reports/external_phase2_protocol_v1.json` plus the",
        "confirmation preregistration `artifacts/reports/external_phase3_preregistration.json`.",
        "Seeds 42/123/2026 were locked in phase 2. Seeds 7 and 13 were preregistered",
        "in phase 3 before any of their results existed and apply to the two mouse",
        "candidates; GSE305979 was preregistered later with a three-seed panel, and",
        "its 7/13 extension is a separate decision. Each row states its own panel.",
        "",
        "All values are `population_mean_across_gene_pearson_r` on the same",
        "2,991-gene panel and train-only preprocessing. External results are",
        "stress tests and do not modify the frozen internal primary result.",
        "",
    ]
    for accession, block in aggregate["datasets"].items():
        lines.append(f"## {accession}")
        lines.append("")
        if block.get("status") == "pending_production_runs":
            lines.append(f"- status: `pending_production_runs` ({block.get('reason')})")
            lines.append("")
            continue
        lines.append(f"- result tag: `{block['result_tag']}`")
        lines.append(
            f"- train -> holdout: {block['train_timepoints']} -> {block['holdout_timepoints']}"
        )
        lines.append(f"- replicate note: {block['replicate_note']}")
        lines.append("")
        lines.append("| endpoint | model | per-seed | mean | SD | seed CI95 |")
        lines.append("|---|---|---|---:|---:|---|")
        for horizon, horizon_block in block["horizons"].items():
            for model in ("ode", "gru"):
                arm = horizon_block[model]
                per_seed = ", ".join(f"{v:+.4f}" for v in arm["per_seed"].values())
                lines.append(
                    f"| {horizon} | {model.upper()} | {per_seed} | "
                    f"{arm['summary']['mean']:+.4f} | {arm['summary']['sd']:.4f} | "
                    f"[{arm['summary']['ci95_low']:+.4f}, {arm['summary']['ci95_high']:+.4f}] |"
                )
            persistence = horizon_block.get("persistence")
            if persistence and persistence.get("status") == "completed":
                lines.append(
                    f"| {horizon} | persistence (deterministic) | fixed | "
                    f"{persistence['corr']:+.4f} | — | — |"
                )
        lines.append("")
    lines.append("## Contrasts")
    lines.append("")
    lines.append("| dataset | endpoint | ODE - GRU (mean) | ODE - persistence (mean) |")
    lines.append("|---|---|---:|---:|")
    for accession, block in aggregate["datasets"].items():
        if block.get("status") == "pending_production_runs":
            continue
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
                + (
                    f"{ode - persistence_value:+.4f}"
                    if persistence_value is not None
                    else "n/a"
                )
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


def forest_rows() -> list[dict]:
    """ODE-GRU differences per endpoint, from stored per-seed artifacts."""
    rows = []
    for label, tag, source, target in FOREST_ROWS:
        if tag == "primary_internal_freeze":
            seeds = [42, 123, 2026]
        else:
            accession = tag.split("_")[1].upper()
            seeds = seeds_for(accession, PHASE3_SEEDS)
        diffs = []
        for seed in seeds:
            try:
                ode = load_seed(tag, "ode", seed)
                gru = load_seed(tag, "gru", seed)
            except FileNotFoundError:
                continue
            pair_ode = ode["pairs"].get(str(target))
            pair_gru = gru["pairs"].get(str(target))
            if pair_ode is None or pair_gru is None:
                continue
            diffs.append(pair_ode["corr"] - pair_gru["corr"])
        if not diffs:
            continue
        n = len(diffs)
        mean = statistics.mean(diffs)
        sd = statistics.stdev(diffs) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 1 else 0.0
        rows.append(
            {
                "label": label,
                "tag": tag,
                "source_time": source,
                "target_time": target,
                "n_seeds": n,
                "per_seed_diff": diffs,
                "mean_diff": mean,
                "sd_diff": sd,
                "se_diff": se,
                "ci95": [mean - 1.96 * se, mean + 1.96 * se] if n > 1 else None,
            }
        )
    for label, path, source, target in HISTORIC_ROWS:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        per_seed = payload.get("per_seed", {})
        diffs = []
        for seed, block in sorted(per_seed.items(), key=lambda item: int(item[0])):
            try:
                ode = float(block["ode"]["extrap"])
                gru = float(block["gru"]["extrap"])
            except (KeyError, TypeError, ValueError):
                continue
            diffs.append(ode - gru)
        if not diffs:
            continue
        n = len(diffs)
        mean = statistics.mean(diffs)
        sd = statistics.stdev(diffs) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 1 else 0.0
        rows.append(
            {
                "label": label,
                "source": str(path.relative_to(ROOT)),
                "source_time": source,
                "target_time": target,
                "n_seeds": n,
                "per_seed_diff": diffs,
                "mean_diff": mean,
                "sd_diff": sd,
                "se_diff": se,
                "ci95": [mean - 1.96 * se, mean + 1.96 * se] if n > 1 else None,
                "note": "historic pre-aggregated endpoint; different protocol/panel",
            }
        )
    return rows


def der_simonian_laird(rows: list[dict]) -> dict:
    usable = [row for row in rows if row["n_seeds"] > 1 and row["se_diff"] > 0]
    if not usable:
        return {"status": "not_enough_rows"}
    weights = [1.0 / (row["se_diff"] ** 2) for row in usable]
    total_w = sum(weights)
    fixed = sum(w * row["mean_diff"] for w, row in zip(weights, usable)) / total_w
    q = sum(w * (row["mean_diff"] - fixed) ** 2 for w, row in zip(weights, usable))
    df = len(usable) - 1
    c = total_w - sum(w * w for w in weights) / total_w
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    random_weights = [1.0 / (row["se_diff"] ** 2 + tau2) for row in usable]
    random_total = sum(random_weights)
    pooled = (
        sum(w * row["mean_diff"] for w, row in zip(random_weights, usable))
        / random_total
    )
    pooled_se = math.sqrt(1.0 / random_total)
    i2 = max(0.0, (q - df) / q) * 100.0 if q > 0 else 0.0
    return {
        "status": "computed_descriptive_only",
        "n_endpoints": len(usable),
        "fixed_effect_mean": fixed,
        "q": q,
        "df": df,
        "tau2": tau2,
        "i_squared_percent": i2,
        "random_effect_mean": pooled,
        "random_effect_ci95": [pooled - 1.96 * pooled_se, pooled + 1.96 * pooled_se],
        "note": (
            "Endpoints differ by dataset, protocol and horizon and are not "
            "exchangeable; this summary is descriptive and is not promoted "
            "to a pooled claim."
        ),
    }


def build_forest_plot(rows: list[dict], meta: dict, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 0.6 * len(rows) + 1.6))
    ys = list(range(len(rows)))[::-1]
    for y, row in zip(ys, rows):
        color = "#1f77b4" if row["mean_diff"] >= 0 else "#d62728"
        ax.errorbar(
            row["mean_diff"],
            y,
            xerr=(
                [[row["mean_diff"] - row["ci95"][0]],
                 [row["ci95"][1] - row["mean_diff"]]]
                if row["ci95"]
                else None
            ),
            fmt="o",
            color=color,
            capsize=3,
        )
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(meta["random_effect_mean"], color="gray", linewidth=1.0, linestyle=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['label']} (n={r['n_seeds']})" for r in rows])
    ax.set_xlabel("ODE - GRU  (population mean gene-expression r)")
    ax.set_title(
        "Cross-endpoint ODE-GRU differences (descriptive; I²="
        f"{meta.get('i_squared_percent', float('nan')):.0f}%)"
    )
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,123,2026,7,13")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "external_phase3_aggregate_v2.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=REPORT_DIR / "external_phase3_report_v2.md",
    )
    parser.add_argument(
        "--forest",
        type=Path,
        default=REPORT_DIR / "external_phase3_forest.json",
    )
    parser.add_argument(
        "--forest-png",
        type=Path,
        default=REPORT_DIR / "external_phase3_forest.png",
    )
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    protocol_path = REPORT_DIR / "external_phase2_protocol_v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    persistence_report = load_persistence_report()

    datasets_block = {}
    for accession, spec in ARM_SPECS.items():
        dataset_seeds = seeds_for(accession, seeds)
        split = protocol_split(accession)
        missing_seed = False
        for model in ("ode", "gru"):
            for seed in dataset_seeds:
                path = RESULT_DIR / (
                    f"validation_results_{model}_{spec['tag']}_seed{seed}.json"
                )
                if not path.is_file():
                    missing_seed = True
                    break
            if missing_seed:
                break
        if missing_seed:
            datasets_block[accession] = {
                "result_tag": spec["tag"],
                "status": "pending_production_runs",
                "reason": (
                    "not all (model, seed) validation artifacts exist yet; "
                    "rerun the aggregate after the runs land"
                ),
            }
            continue
        rows = {
            model: {
                str(seed): load_seed(spec["tag"], model, seed)
                for seed in dataset_seeds
            }
            for model in ("ode", "gru")
        }
        horizons_block = {}
        for target, role in spec["targets"].items():
            block = {}
            for model in ("ode", "gru"):
                per_seed = {}
                for seed in dataset_seeds:
                    pair = rows[model][str(seed)]["pairs"].get(str(target))
                    if pair is None:
                        raise ValueError(
                            f"{accession} {spec['tag']} {model} seed {seed}: "
                            f"missing horizon {target}"
                        )
                    per_seed[str(seed)] = pair["corr"]
                block[model] = {
                    "per_seed": per_seed,
                    "summary": exact_seed_bootstrap(list(per_seed.values())),
                }
            persistence = None
            if persistence_report and accession in persistence_report.get("datasets", {}):
                entry = persistence_report["datasets"][accession]["horizons"].get(
                    str(target)
                )
                if isinstance(entry, dict):
                    persistence = {
                        "status": "completed",
                        "corr": float(entry["corr"]),
                        "deterministic": bool(entry.get("deterministic", True)),
                        "n_source": entry.get("n_source"),
                        "n_target": entry.get("n_target"),
                    }
            block["persistence"] = persistence
            block["source_time"] = spec["train_source_time"]
            block["role"] = role
            horizons_block[str(target)] = block

        datasets_block[accession] = {
            "result_tag": spec["tag"],
            "train_timepoints": split["train_timepoints"],
            "holdout_timepoints": split["holdout_timepoints"],
            "replicate_note": split["replicate_note"],
            "protocol": split["protocol_file"],
            "actual_epochs": {
                model: {
                    str(seed): rows[model][str(seed)]["actual_epochs"]
                    for seed in dataset_seeds
                }
                for model in ("ode", "gru")
            },
            "horizons": horizons_block,
            "seed_artifacts": {
                model: {
                    str(seed): rows[model][str(seed)]["file"]
                    for seed in dataset_seeds
                }
                for model in ("ode", "gru")
            },
        }

    aggregate = {
        "record_type": "external_phase3_aggregate",
        "created_at": utcnow(),
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": sha256(protocol_path),
        "protocol_per_dataset": {
            accession: {
                "file": PROTOCOL_FILES[accession],
                "sha256": sha256(REPORT_DIR / PROTOCOL_FILES[accession]),
            }
            for accession in ARM_SPECS
        },
        "preregistration": "artifacts/reports/external_phase3_preregistration.json",
        "seeds": seeds,
        "seed_panels": {
            accession: seeds_for(accession, seeds) for accession in ARM_SPECS
        },
        "seed_panel_note": (
            "per-dataset panels: the two mouse candidates use the five seeds "
            "locked in external_phase3_preregistration.json; GSE305979 uses "
            "the three seeds locked in its own protocol, because its 7/13 "
            "extension is a separate decision"
        ),
        "datasets": datasets_block,
    }
    args.output.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(build_markdown(aggregate), encoding="utf-8")

    rows = forest_rows()
    meta = der_simonian_laird(rows)
    forest = {
        "record_type": "external_phase3_forest",
        "created_at": utcnow(),
        "metric": "ODE - GRU per-seed difference of population_mean_across_gene_pearson_r",
        "endpoints": rows,
        "heterogeneity": meta,
    }
    args.forest.write_text(
        json.dumps(forest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    build_forest_plot(rows, meta, args.forest_png)

    print(f"saved: {args.output}")
    print(f"saved: {args.markdown}")
    print(f"saved: {args.forest}")
    print(f"saved: {args.forest_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
