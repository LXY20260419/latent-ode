#!/usr/bin/env python3
"""Create a compact, provenance-aware report for one external smoke run.

The report is intentionally descriptive.  It does not merge the external
result into the frozen primary aggregate or submission gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "artifacts" / "results" / "ode_benchmark"
REPORT_DIR = ROOT / "artifacts" / "reports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric(payload: dict, section: str) -> dict:
    block = payload.get(section, {})
    return {
        "mean_correlation": block.get("mean_correlation"),
        "std": block.get("std"),
        "pairs": block.get("pairs"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accession", required=True)
    parser.add_argument("--result-tag", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    qc = json.loads(args.qc.read_text(encoding="utf-8"))
    ode_path = RESULT_DIR / (
        f"validation_results_ode_{args.result_tag}_seed{args.seed}.json"
    )
    gru_path = RESULT_DIR / (
        f"validation_results_gru_{args.result_tag}_seed{args.seed}.json"
    )
    required = [ode_path, gru_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing smoke artifacts: {missing}")

    ode = json.loads(ode_path.read_text(encoding="utf-8"))
    gru = json.loads(gru_path.read_text(encoding="utf-8"))
    payload = {
        "record_type": "external_candidate_one_seed_smoke",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "accession": args.accession,
        "result_tag": args.result_tag,
        "seed": args.seed,
        "smoke_only": True,
        "qc_status": qc.get("status"),
        "replicate_note": qc.get("replicate_note"),
        "dataset": qc.get("input_shape_gene_by_cell_or_cell_by_gene"),
        "timepoints": {
            "all": qc.get("timepoints"),
            "train": qc.get("train_timepoints"),
            "holdout": qc.get("holdout_timepoints"),
            "secondary_holdout": qc.get("secondary_holdout_timepoints"),
        },
        "leakage_control": qc.get("leakage_control"),
        "metrics": {
            "ode": {
                "holdout": metric(ode, "time_holdout"),
                "extrapolation": metric(ode, "time_extrapolation"),
            },
            "gru": {
                "holdout": metric(gru, "time_holdout"),
                "extrapolation": metric(gru, "time_extrapolation"),
            },
        },
        "artifacts": {
            "ode_validation": str(ode_path.relative_to(ROOT)),
            "gru_validation": str(gru_path.relative_to(ROOT)),
            "ode_validation_sha256": sha256(ode_path),
            "gru_validation_sha256": sha256(gru_path),
        },
        "interpretation": (
            "This is a one-seed pipeline smoke, not evidence of biological "
            "replication or a replacement for the frozen primary result. "
            "Production seed expansion is allowed only after QC and artifact "
            "integrity review."
        ),
        "status": (
            "PASS_FOR_PRODUCTION_REVIEW"
            if qc.get("status") == "PASS_FOR_SMOKE"
            and all(
                metric(payload, "time_extrapolation").get("mean_correlation")
                is not None
                for payload in (ode, gru)
            )
            else "FAIL"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "PASS_FOR_PRODUCTION_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
