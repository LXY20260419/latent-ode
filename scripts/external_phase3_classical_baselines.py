#!/usr/bin/env python3
"""Classical (non-neural) baselines on the external population endpoint.

The Level-B plan asks for at least one non-neural baseline on the same
endpoint as the model arms. This tool evaluates, for each external dataset
and holdout horizon:

* ``Naive_LastTimeMean`` — the source-time population mean carried forward
  (numerically identical to the persistence baseline already recorded; kept
  here so the baseline table is complete in one artifact);
* ``Ridge_Poly`` — per-gene Ridge on the same four time features used
  internally (t, t^2, log1p(t), sin(2*pi*t/max_time)). Deterministic.
* ``HGB_Regressor`` and ``MLP_Sklearn`` — same hyperparameters as the internal
  baseline suite, per requested seed (these two are seed-random).

The endpoint is ``population_mean_across_gene_pearson_r`` on the full
2,991-gene panel; unlike the legacy internal baseline file this tool evaluates
all 2,991 genes instead of truncating to 2,000, so it matches the model arms.

Outputs ``artifacts/reports/external_phase3_classical_baselines.json``.
This is read-only with respect to the model results: it never writes into the
ode_benchmark result namespace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "artifacts" / "reports"
RESULT_DIR = ROOT / "artifacts" / "results" / "ode_benchmark"

ARMS = {
    "GSE148794": {
        "data_dir": "data/leworld_input_gse148794",
        "tag": "ext_gse148794_v2",
        "train_times": [0, 3, 6, 9],
        "targets": [12, 15],
        "max_time": 15.0,
    },
    "GSE149622": {
        "data_dir": "data/leworld_input_gse149622",
        "tag": "ext_gse149622_v2",
        "train_times": [0, 1, 2, 3],
        "targets": [4],
        "max_time": 4.0,
    },
    "GSE305979": {
        "data_dir": "data/leworld_input_gse305979",
        "tag": "ext_gse305979_control_v2",
        "train_times": [0, 1, 2],
        "targets": [5, 7],
        "max_time": 7.0,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_time_features(t, max_time: float):
    import numpy as np

    t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    return np.hstack(
        [t, t**2, np.log1p(t), np.sin(2 * np.pi * t / max_time)]
    )


def safe_corr(left, right) -> float:
    import numpy as np

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.std() < 1e-12 or right.std() < 1e-12:
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    return float(left @ right) / math.sqrt(float(left @ left) * float(right @ right))


def endpoint(pred, x_target, target_times, target_time: int) -> float:
    import numpy as np

    mask = target_times == target_time
    return safe_corr(x_target[mask].mean(axis=0), pred[mask].mean(axis=0))


def train_classical(model_name: str, x_train, t_train, x_test, t_test, max_time, seed):
    import numpy as np

    n_genes = x_train.shape[1]
    f_train = make_time_features(t_train, max_time)
    f_test = make_time_features(t_test, max_time)
    preds = np.zeros_like(x_test)

    if model_name == "Ridge_Poly":
        from sklearn.linear_model import Ridge

        for g in range(n_genes):
            model = Ridge(alpha=1.0)
            model.fit(f_train, x_train[:, g])
            preds[:, g] = model.predict(f_test)
    elif model_name == "HGB_Regressor":
        from sklearn.ensemble import HistGradientBoostingRegressor

        for g in range(n_genes):
            model = HistGradientBoostingRegressor(
                max_depth=3,
                min_samples_leaf=50,
                learning_rate=0.05,
                max_iter=50,
                random_state=seed,
            )
            model.fit(f_train, x_train[:, g])
            preds[:, g] = model.predict(f_test)
    elif model_name == "MLP_Sklearn":
        from sklearn.neural_network import MLPRegressor

        for g in range(n_genes):
            model = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=200,
                random_state=seed,
                activation="relu",
                learning_rate="adaptive",
            )
            model.fit(f_train, x_train[:, g])
            preds[:, g] = model.predict(f_test)
    else:
        raise ValueError(model_name)
    return preds


def main() -> int:
    import numpy as np
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="Naive_LastTimeMean,Ridge_Poly",
        help="comma-separated subset of Naive_LastTimeMean,Ridge_Poly,HGB_Regressor,MLP_Sklearn",
    )
    parser.add_argument("--model-seeds", default="42", help="seeds for the seed-random baselines")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "external_phase3_classical_baselines.json",
    )
    args = parser.parse_args()

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    model_seeds = [int(s) for s in args.model_seeds.split(",") if s.strip()]
    deterministic = {"Naive_LastTimeMean", "Ridge_Poly"}

    report = {
        "record_type": "external_phase3_classical_baselines",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric": "population_mean_across_gene_pearson_r",
        "notes": [
            "Same endpoint, gene panel (2,991 genes), train timepoints and horizons as the model arms.",
            "Naive_LastTimeMean duplicates the persistence baseline definition; kept for a complete baseline table.",
            "Ridge_Poly is deterministic; HGB/MLP are seed-random and run for the requested model seeds.",
        ],
        "datasets": {},
    }

    for accession, spec in ARMS.items():
        data_dir = ROOT / spec["data_dir"]
        matrix_path = data_dir / "expression_matrix.npy"
        if not matrix_path.is_file():
            report["datasets"][accession] = {
                "data_dir": spec["data_dir"],
                "status": "pending_data",
                "reason": "converted bundle not present yet; rerun after conversion",
            }
            continue
        x = np.load(matrix_path)
        meta = pd.read_csv(data_dir / "metadata.csv")
        times = meta["timepoint_numeric"].to_numpy()

        train_mask = np.isin(times, spec["train_times"])
        test_mask = np.isin(times, spec["targets"])
        x_train = x[train_mask]
        t_train = times[train_mask]
        x_test = x[test_mask]
        t_test = times[test_mask]
        source_time = max(spec["train_times"])
        source_mean = x_train[t_train == source_time].mean(axis=0)

        block = {
            "data_dir": spec["data_dir"],
            "expression_matrix_sha256": sha256(matrix_path),
            "n_train_cells": int(train_mask.sum()),
            "n_test_cells": int(test_mask.sum()),
            "n_genes": int(x.shape[1]),
            "baselines": {},
        }

        for model_name in wanted:
            if model_name == "Naive_LastTimeMean":
                pred = np.tile(source_mean, (x_test.shape[0], 1))
                block["baselines"][model_name] = {
                    "deterministic": True,
                    "horizons": {
                        str(target): endpoint(pred, x_test, t_test, target)
                        for target in spec["targets"]
                    },
                }
                continue
            seeds = [None] if model_name in deterministic else model_seeds
            per_seed = {}
            for seed in seeds:
                pred = train_classical(
                    model_name,
                    x_train,
                    t_train,
                    x_test,
                    t_test,
                    spec["max_time"],
                    seed,
                )
                key = "deterministic" if seed is None else str(seed)
                per_seed[key] = {
                    str(target): endpoint(pred, x_test, t_test, target)
                    for target in spec["targets"]
                }
                print(f"{accession} {model_name} seed={key} done", flush=True)
            block["baselines"][model_name] = {
                "deterministic": model_name in deterministic,
                "per_seed": per_seed,
            }

        # point comparison against stored ODE/GRU artifacts (5 seeds when present)
        comparisons = {}
        model_seeds_all = [42, 123, 2026, 7, 13]
        for model in ("ode", "gru"):
            per_seed_vals = {}
            for seed in model_seeds_all:
                path = RESULT_DIR / f"validation_results_{model}_{spec['tag']}_seed{seed}.json"
                if not path.is_file():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                pairs = payload["time_extrapolation"]["pairs"]
                per_seed_vals[str(seed)] = {
                    str(target): float(pairs[str(target)]["corr"])
                    for target in spec["targets"]
                    if str(target) in pairs
                }
            comparisons[model] = per_seed_vals
        block["model_arms"] = comparisons
        report["datasets"][accession] = block

    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
