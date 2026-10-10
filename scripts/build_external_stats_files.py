#!/usr/bin/env python3
"""Assemble the per-accession stats files promised by plan section 13.

The deliverable convention lists ``external_<accession>_stats.json`` for every
new dataset. The underlying numbers already exist, spread over the phase-2/3
aggregates, the persistence report, the sample-uncertainty report and the
classical-baseline report; this tool only assembles them into one stats file
per accession so a reviewer never has to chase four files.

What goes into each stats file:

* endpoint definition (train/holdout/source time, metric, replicate note);
* per-horizon ODE/GRU mean, SD and seed-bootstrap CI using each accession's
  preregistered seed panel;
* deterministic persistence values;
* ODE-minus-GRU and ODE-minus-persistence contrasts;
* the cell-level and sample-cluster uncertainty summary;
* the classical baselines recorded so far (the file is regenerated when the
  optional HGB/MLP run lands);
* the mechanism proxies for both arms with the training-coupling note.

Nothing is recomputed from raw data and no result artifact is modified;
missing optional blocks are recorded as absent rather than imputed.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "artifacts" / "reports"

ACCESSIONS = {
    "GSE148794": {
        "protocol": "artifacts/reports/external_gse148794_protocol.json",
        "qc": "artifacts/reports/external_gse148794_qc.json",
    },
    "GSE149622": {
        "protocol": "artifacts/reports/external_gse149622_protocol.json",
        "qc": "artifacts/reports/external_gse149622_qc.json",
    },
    "GSE305979": {
        "protocol": "artifacts/reports/external_gse305979_protocol.json",
        "qc": "artifacts/reports/external_gse305979_qc.json",
    },
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stats_seeds(phase3: dict, accession: str) -> list[int]:
    """Seed panel actually used for one accession.

    Each dataset reports on its own preregistered panel (the two mouse
    candidates at five seeds, GSE305979 at three), so the per-accession stats
    file must not inherit the aggregate-wide seed list: that would claim seeds
    the dataset never ran.
    """
    panels = phase3.get("seed_panels")
    if isinstance(panels, dict) and accession in panels:
        return list(panels[accession])
    return list(phase3["seeds"])


def build_stats(accession: str) -> dict:
    phase3 = read_json(REPORT_DIR / "external_phase3_aggregate_v2.json")
    if accession not in phase3["datasets"]:
        return {
            "record_type": "external_accession_stats",
            "accession": accession,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending_production_runs",
            "reason": (
                "no phase-3 aggregate block yet; the stats file is generated "
                "once that accession's preregistered production runs land"
            ),
            "provenance": {
                "protocol": ACCESSIONS[accession]["protocol"],
                "qc": ACCESSIONS[accession]["qc"],
            },
        }
    persistence = read_json(REPORT_DIR / "external_persistence_v1.json")
    uncertainty = read_json(REPORT_DIR / "external_phase3_sample_uncertainty.json")
    mechanism = read_json(REPORT_DIR / "mechanism_metrics_aggregate.json")
    baselines = read_json(REPORT_DIR / "external_phase3_classical_baselines.json")

    phase3_block = phase3["datasets"][accession]
    horizons: dict = {}
    for horizon, hb in phase3_block["horizons"].items():
        persistence_block = persistence["datasets"][accession]["horizons"].get(
            horizon
        )
        unc = uncertainty["datasets"][accession]["horizons"].get(horizon, {})
        horizons[horizon] = {
            "source_time": hb.get("source_time"),
            "role": hb.get("role"),
            "ode": {
                "mean": hb["ode"]["summary"]["mean"],
                "sd": hb["ode"]["summary"]["sd"],
                "n": hb["ode"]["summary"]["n"],
                "ci95": [
                    hb["ode"]["summary"]["ci95_low"],
                    hb["ode"]["summary"]["ci95_high"],
                ],
                "per_seed": hb["ode"]["per_seed"],
            },
            "gru": {
                "mean": hb["gru"]["summary"]["mean"],
                "sd": hb["gru"]["summary"]["sd"],
                "n": hb["gru"]["summary"]["n"],
                "ci95": [
                    hb["gru"]["summary"]["ci95_low"],
                    hb["gru"]["summary"]["ci95_high"],
                ],
                "per_seed": hb["gru"]["per_seed"],
            },
            "persistence": (
                {
                    "corr": persistence_block["corr"],
                    "deterministic": True,
                }
                if persistence_block
                else None
            ),
            "ode_minus_gru": (
                hb["ode"]["summary"]["mean"] - hb["gru"]["summary"]["mean"]
            ),
            "ode_minus_persistence": (
                hb["ode"]["summary"]["mean"] - persistence_block["corr"]
                if persistence_block
                else None
            ),
            "data_uncertainty": {
                "cell_level": unc.get("cell_level"),
                "sample_cluster": unc.get("sample_cluster"),
                "note": (
                    "data-sampling uncertainty at fixed checkpoints; not "
                    "training variance and not seed uncertainty"
                ),
            },
        }

    mechanism_layer = mechanism.get("layers", {}).get(accession.lower())
    mechanism_arms = {}
    if mechanism_layer:
        for arm in ("ode", "gru"):
            mechanism_arms[arm] = {
                key: {
                    "mean": mechanism_layer["arms"][arm][key]["summary"]["mean"],
                    "sd": mechanism_layer["arms"][arm][key]["summary"]["sd"],
                    "n": mechanism_layer["arms"][arm][key]["summary"]["n"],
                }
                for key in (
                    "tf_direction_consistency",
                    "pathway_activation_order",
                    "marker_rank_spearman",
                )
            }

    baseline_block = baselines.get("datasets", {}).get(accession) or {}
    pure_mse_path = REPORT_DIR / "external_pure_mse_contrast.json"
    pure_mse_block = None
    if pure_mse_path.is_file():
        pure_payload = read_json(pure_mse_path)
        dataset_block = pure_payload.get("datasets", {}).get(accession)
        if dataset_block:
            pure_mse_block = {
                "preregistration": pure_payload.get("preregistration"),
                "endpoints": {
                    endpoint: {
                        "pure_mse_mean": eb["pure_mse"]["mean"],
                        "full_mean": eb["full"]["mean"],
                        "paired_pure_minus_full": eb["paired_pure_minus_full"],
                    }
                    for endpoint, eb in dataset_block["endpoints"].items()
                },
                "note": (
                    "preregistered paired contrast; every endpoint kept "
                    "regardless of sign"
                ),
            }
    return {
        "record_type": "external_accession_stats",
        "accession": accession,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_tag": phase3_block["result_tag"],
        "seeds": stats_seeds(phase3, accession),
        "train_timepoints": phase3_block["train_timepoints"],
        "holdout_timepoints": phase3_block["holdout_timepoints"],
        "replicate_note": phase3_block["replicate_note"],
        "metric": "population_mean_across_gene_pearson_r",
        "horizons": horizons,
        "classical_baselines": baseline_block.get("baselines"),
        "pure_mse_contrast": pure_mse_block,
        "mechanism_proxies": {
            "arms": mechanism_arms,
            "coupling_note": (
                "Prior-consistency proxies only: pathway sets enter the "
                "training loss (pathway_score 0.2, pathway_delta 0.1) and the "
                "TF/marker sets share the same priors object; not independent "
                "mechanism validation."
            ),
        },
        "provenance": {
            "protocol": ACCESSIONS[accession]["protocol"],
            "qc": ACCESSIONS[accession]["qc"],
            "phase3_aggregate": "artifacts/reports/external_phase3_aggregate_v2.json",
            "persistence": "artifacts/reports/external_persistence_v1.json",
            "sample_uncertainty": "artifacts/reports/external_phase3_sample_uncertainty.json",
            "classical_baselines": "artifacts/reports/external_phase3_classical_baselines.json",
            "mechanism_metrics": "artifacts/reports/mechanism_metrics_aggregate.json",
            "pure_mse_contrast": (
                "artifacts/reports/external_pure_mse_contrast.json"
                if pure_mse_block
                else None
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir", type=Path, default=REPORT_DIR
    )
    args = parser.parse_args()

    for accession in ACCESSIONS:
        stats = build_stats(accession)
        out = args.report_dir / f"external_{accession.lower()}_stats.json"
        out.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
