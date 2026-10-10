#!/usr/bin/env python3
"""Population-persistence baseline for the external Phase-2 datasets.

For each (accession, source-time -> target-time) horizon, copies the source
timepoint population mean forward and scores it against the target timepoint
population mean with the same across-gene Pearson metric used by the model
arms.  The score is deterministic, so the same value is valid for every seed.

Read-only with respect to model artifacts; writes a single JSON report.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "reports" / "external_persistence_v1.json"

SPECS = {
    "GSE148794": {
        "data_dir": ROOT / "data" / "leworld_input_gse148794",
        "horizons": [(9, 12), (9, 15)],
    },
    "GSE149622": {
        "data_dir": ROOT / "data" / "leworld_input_gse149622",
        "horizons": [(3, 4)],
    },
    "GSE305979": {
        "data_dir": ROOT / "data" / "leworld_input_gse305979",
        "horizons": [(2, 5), (2, 7)],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def across_gene_pearson(left, right) -> float:
    import numpy as np

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    if denominator == 0.0:
        return float("nan")
    return float(left @ right) / denominator


def load_bundle(data_dir: Path):
    import numpy as np

    matrix = np.load(data_dir / "expression_matrix.npy")
    times = []
    with (data_dir / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            times.append(int(float(row["timepoint_numeric"])))
    if len(times) != matrix.shape[0]:
        raise ValueError(f"{data_dir}: metadata rows do not match matrix")
    return matrix, np.asarray(times)


def main() -> int:
    report = {
        "record_type": "external_persistence_baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric": "population_mean_across_gene_pearson_r",
        "method": "population_persistence",
        "datasets": {},
    }
    for accession, spec in SPECS.items():
        matrix, times = load_bundle(spec["data_dir"])
        horizons = {}
        for source_time, target_time in spec["horizons"]:
            source_index = times == source_time
            target_index = times == target_time
            if not source_index.any() or not target_index.any():
                raise ValueError(
                    f"{accession}: missing cells for {source_time}->{target_time}"
                )
            source_mean = matrix[source_index].mean(axis=0)
            target_mean = matrix[target_index].mean(axis=0)
            corr = across_gene_pearson(source_mean, target_mean)
            horizons[str(target_time)] = {
                "from_time": int(source_time),
                "corr": float(corr),
                "n_source": int(source_index.sum()),
                "n_target": int(target_index.sum()),
                "deterministic": True,
            }
        report["datasets"][accession] = {
            "data_dir": str(spec["data_dir"].relative_to(ROOT)),
            "metadata_sha256": sha256(spec["data_dir"] / "metadata.csv"),
            "horizons": horizons,
        }
    OUTPUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"saved: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
