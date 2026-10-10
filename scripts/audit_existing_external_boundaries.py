#!/usr/bin/env python3
"""Audit existing external bundles and boundary results without retraining."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "artifacts" / "reports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_summary(path: Path, grouping: list[str]) -> dict:
    meta = pd.read_csv(path, index_col=0)
    time_counts = (
        meta.groupby("timepoint_numeric", sort=True)
        .size()
        .astype(int)
        .to_dict()
    )
    by_time = {}
    for time, frame in meta.groupby("timepoint_numeric", sort=True):
        block = {}
        for column in grouping:
            if column in frame:
                values = frame[column].astype(str)
                block[column] = {
                    "n_unique": int(values.nunique()),
                    "counts": {
                        str(k): int(v)
                        for k, v in values.value_counts().head(20).items()
                    },
                }
        by_time[str(int(time))] = block
    return {
        "metadata_path": str(path.relative_to(ROOT)),
        "metadata_sha256": sha256(path),
        "n_cells": int(len(meta)),
        "metadata_columns": list(meta.columns),
        "timepoint_counts": {str(int(k)): int(v) for k, v in time_counts.items()},
        "grouping_by_time": by_time,
    }


def load_json(name: str) -> dict:
    path = REPORT_DIR / name
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "payload": json.loads(path.read_text(encoding="utf-8")),
    }


def main() -> int:
    gse224 = metadata_summary(
        ROOT / "data" / "leworld_input_gse224518" / "metadata.csv",
        ["Sample_Tag", "Sample_Name", "batch", "replicate"],
    )
    gse226 = metadata_summary(
        ROOT / "data" / "leworld_input_gse226824" / "metadata.csv",
        ["hashtags", "clusters"],
    )
    short = load_json("stage2_gse224518_3to5h_agg.json")
    long = load_json("stage2_gse224518_agg.json")
    ifn = load_json("stage2_gse226824_agg.json")
    payload = {
        "record_type": "existing_external_boundary_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "read-only audit; no retraining; no manuscript or primary-gate edits",
        "datasets": {
            "GSE224518": {
                "bundle": gse224,
                "short_horizon": short,
                "long_horizon": long,
                "interpretation": [
                    "The 3h->5h ODE endpoint is positive but does not beat matched GRU; Ridge is stronger on the same population extrapolation endpoint.",
                    "The 3h->8h ODE endpoint is negative with low seed spread and is retained as a pre-specified long-horizon failure boundary.",
                    "Sample-tag, batch, and replicate composition changes across timepoints; cells must not be treated as independent biological replicates.",
                    "The bundle is a targeted 498-gene panel, so its transfer result is not directly comparable to the full-transcriptome internal bundle.",
                ],
                "claim_level": "exploratory boundary evidence; no broad generalization claim",
            },
            "GSE226824": {
                "bundle": gse226,
                "aggregate": ifn,
                "interpretation": [
                    "The 24h->72h ODE mean is positive but seed variance is large, so the result is exploratory rather than stable confirmation.",
                    "Hashtags and cell clusters are multiplexing/annotation fields, not independent donor-level biological replicates by themselves.",
                    "No sample/donor-level inferential uncertainty is claimed from the current bundle.",
                ],
                "claim_level": "exploratory stress test; no broad generalization claim",
            },
        },
        "cross_dataset_decision": {
            "pooling": "forbidden",
            "meta_analysis": "not yet justified because endpoints, horizons, gene panels, and replicate structures differ",
            "next_step": "complete the two new candidate Phase-1 QC/smoke runs before any production expansion",
        },
    }
    json_path = REPORT_DIR / "existing_external_boundary_audit_2026-10-01.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Existing external boundary audit — 2026-10-01",
        "",
        "Scope: read-only audit; no retraining, manuscript edit, primary-freeze edit, or gate edit.",
        "",
        "## GSE224518",
        "",
        "- 3→5 h: ODE is positive but does not beat matched GRU; Ridge is stronger on the same population extrapolation endpoint.",
        "- 3→8 h: ODE is negative with low seed spread; retain as the long-horizon failure boundary.",
        "- Sample-tag, batch, and replicate composition changes across timepoints; cells are not independent biological replicates.",
        "- The bundle uses a targeted 498-gene panel, so it is not directly comparable to the full-transcriptome internal bundle.",
        "",
        "## GSE226824",
        "",
        "- 24→72 h: ODE mean is positive but seed variance is large; treat as exploratory stress-test evidence.",
        "- Hashtags and clusters are not donor-level biological replication by themselves.",
        "- No sample/donor-level inferential uncertainty is claimed.",
        "",
        "## Decision",
        "",
        "- Do not pool these endpoints or promote them to broad generalization claims.",
        "- Finish Phase-1 QC/smoke for GSE148794 and GSE149622 before any production expansion.",
        "",
        f"Machine-readable source: `{json_path.relative_to(ROOT)}`",
    ]
    (REPORT_DIR / "existing_external_boundary_audit_2026-10-01.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
