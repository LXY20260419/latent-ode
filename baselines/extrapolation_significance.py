#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Endpoint-aligned extrapolation significance for ODE vs classical baselines.

Evaluates baselines on the same population-level extrapolation endpoint used by
LEWorldModel validation, then performs one-sided paired t-tests (ODE > baseline)
across seeds.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_rel
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = BASE_DIR / "artifacts" / "results" / "baselines"
ODE_RESULT_DIR = BASE_DIR / "artifacts" / "results" / "ode_benchmark"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _parse_int_list(env_name: str, default: list[int]) -> list[int]:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


class SignificanceConfig:
    DATA_DIR = Path(os.getenv("LEWORLD_DATA_DIR", str(BASE_DIR / "data" / "leworld_input")))
    TRAIN_TIMES = _parse_int_list("LEWORLD_TRAIN_TIMEPOINTS", [0, 1, 4, 16, 27])
    TARGET_TIMES = _parse_int_list("LEWORLD_HOLDOUT_TIMEPOINTS", [36, 48])
    MAX_TIME = float(os.getenv("LEWORLD_MAX_TIME", "48.0"))
    SEEDS = _parse_int_list("LEWORLD_SEEDS", [42, 123, 2026])
    N_GENES = int(os.getenv("LEWORLD_N_GENES", "2000"))
    RESULT_TAG = os.getenv("LEWORLD_RESULT_TAG", "").strip()
    SOURCE_TIME = int(
        os.getenv(
            "LEWORLD_SOURCE_TIME",
            str(max(TRAIN_TIMES) if TRAIN_TIMES else 27),
        )
    )

    @classmethod
    def endpoint_label(cls) -> str:
        pairs = [f"{cls.SOURCE_TIME}->{t}" for t in cls.TARGET_TIMES]
        return f"population mean gene-expression correlation ({', '.join(pairs)})"

    @classmethod
    def model_result_path(cls, model: str, seed: int) -> Path:
        if cls.RESULT_TAG:
            return ODE_RESULT_DIR / f"validation_results_{model}_{cls.RESULT_TAG}_seed{seed}.json"
        return ODE_RESULT_DIR / f"validation_results_{model}_seed{seed}.json"

    @classmethod
    def output_path(cls) -> Path:
        suffix = f"_{cls.RESULT_TAG}" if cls.RESULT_TAG else ""
        return RESULT_DIR / f"extrapolation_significance{suffix}.json"


cfg = SignificanceConfig()


def make_time_features(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t).reshape(-1, 1)
    max_time = cfg.MAX_TIME
    return np.hstack([t, t**2, np.log1p(t), np.sin(2 * np.pi * t / max_time)])


def load_data() -> tuple[np.ndarray, np.ndarray]:
    x = np.load(cfg.DATA_DIR / "expression_matrix.npy")
    meta = pd.read_csv(cfg.DATA_DIR / "metadata.csv", index_col=0)
    times = meta["timepoint_numeric"].to_numpy()
    return x, times


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(pearsonr(a, b)[0])


def population_extrap_metrics(pred: np.ndarray, x_test: np.ndarray, t_test: np.ndarray) -> dict:
    pair_corr = {}
    for tt in cfg.TARGET_TIMES:
        m = t_test == tt
        pred_mean = pred[m].mean(axis=0)
        true_mean = x_test[m].mean(axis=0)
        pair_corr[str(tt)] = safe_corr(true_mean, pred_mean)
    vals = [pair_corr[str(tt)] for tt in cfg.TARGET_TIMES]
    return {
        "pair_corr": pair_corr,
        "mean_extrapolation_corr": float(np.nanmean(vals)),
    }


def run_single_seed(seed: int, x: np.ndarray, times: np.ndarray) -> dict:
    np.random.seed(seed)

    train_mask = np.isin(times, cfg.TRAIN_TIMES)
    test_mask = np.isin(times, cfg.TARGET_TIMES)

    x_train = x[train_mask]
    t_train = times[train_mask]
    x_test = x[test_mask]
    t_test = times[test_mask]

    n_genes = min(cfg.N_GENES, x_train.shape[1], x_test.shape[1])
    x_train = x_train[:, :n_genes]
    x_test = x_test[:, :n_genes]

    f_train = make_time_features(t_train)
    f_test = make_time_features(t_test)

    preds = {}

    src_mask = t_train == cfg.SOURCE_TIME
    src_mean = x_train[src_mask].mean(axis=0)
    preds["Naive_LastTimeMean"] = np.tile(src_mean, (x_test.shape[0], 1))

    ridge_pred = np.zeros_like(x_test)
    hgb_pred = np.zeros_like(x_test)
    mlp_pred = np.zeros_like(x_test)
    for g in tqdm(range(n_genes), desc=f"  baselines seed {seed}", leave=False):
        y = x_train[:, g]
        ridge = Ridge(alpha=1.0)
        ridge.fit(f_train, y)
        ridge_pred[:, g] = ridge.predict(f_test)

        hgb = HistGradientBoostingRegressor(
            max_depth=3,
            min_samples_leaf=50,
            learning_rate=0.05,
            max_iter=50,
            random_state=seed,
        )
        hgb.fit(f_train, y)
        hgb_pred[:, g] = hgb.predict(f_test)

        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=200,
            random_state=seed,
            activation="relu",
            learning_rate="adaptive",
        )
        mlp.fit(f_train, y)
        mlp_pred[:, g] = mlp.predict(f_test)

    preds["Ridge_Poly"] = ridge_pred
    preds["HGB_Regressor"] = hgb_pred
    preds["MLP_Sklearn"] = mlp_pred

    return {name: population_extrap_metrics(pred, x_test, t_test) for name, pred in preds.items()}


def load_model_seed_results(model: str, seed: int) -> dict:
    path = cfg.model_result_path(model, seed)
    d = json.loads(path.read_text())
    pairs_raw = d["time_extrapolation"]["pairs"]
    pair_corr = {str(k): float(v["corr"]) for k, v in pairs_raw.items()}
    vals = [pair_corr[str(tt)] for tt in cfg.TARGET_TIMES if str(tt) in pair_corr]
    return {
        "pair_corr": pair_corr,
        "mean_extrapolation_corr": float(d["time_extrapolation"]["mean_correlation"])
        if vals
        else float("nan"),
    }


def paired_stats(left_vals: list[float], right_vals: list[float]) -> dict:
    left_arr = np.asarray(left_vals, dtype=float)
    right_arr = np.asarray(right_vals, dtype=float)
    diff = left_arr - right_arr
    t = ttest_rel(left_arr, right_arr, alternative="greater")
    return {
        "left_mean": float(np.mean(left_arr)),
        "right_mean": float(np.mean(right_arr)),
        "diff_mean": float(np.mean(diff)),
        "diff_std": float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
        "t_stat": float(t.statistic),
        "p_value": float(t.pvalue),
    }


def build_baseline_summary(ode_by_seed: dict, seed_details: dict) -> dict:
    baselines = ["Naive_LastTimeMean", "Ridge_Poly", "HGB_Regressor", "MLP_Sklearn"]
    summary = {}
    for b in baselines:
        ode_mean_vals = [ode_by_seed[str(s)]["mean_extrapolation_corr"] for s in cfg.SEEDS]
        base_mean_vals = [seed_details[str(s)][b]["mean_extrapolation_corr"] for s in cfg.SEEDS]
        entry = {
            "mean_endpoint": paired_stats(ode_mean_vals, base_mean_vals),
        }
        for tt in cfg.TARGET_TIMES:
            key = f"pair_{cfg.SOURCE_TIME}_to_{tt}"
            ode_vals = [ode_by_seed[str(s)]["pair_corr"][str(tt)] for s in cfg.SEEDS]
            base_vals = [seed_details[str(s)][b]["pair_corr"][str(tt)] for s in cfg.SEEDS]
            entry[key] = paired_stats(ode_vals, base_vals)
        summary[b] = entry
    return summary


def build_summary_report(out: dict) -> dict:
    ode_mean = float(
        np.mean([out["ode_seed_results"][str(s)]["mean_extrapolation_corr"] for s in cfg.SEEDS])
    )
    paired = {}
    for name, stats in out["paired_significance"].items():
        m = stats["mean_endpoint"]
        paired[name] = {
            "baseline_mean": m["right_mean"],
            "ode_mean": m["left_mean"],
            "diff_mean": m["diff_mean"],
            "p_value": m["p_value"],
        }
    report = {
        "endpoint": out["config"]["endpoint"],
        "dataset": cfg.DATA_DIR.name,
        "result_tag": cfg.RESULT_TAG or "internal",
        "seeds": cfg.SEEDS,
        "ode_mean_extrapolation": ode_mean,
        "paired_significance_mean_endpoint": paired,
        "notes": [
            "One-sided paired t-test: ODE > baseline on mean extrapolation endpoint.",
            "Computed on seed-level endpoint values.",
            "Baselines use population mean correlation (same metric as ODE extrapolation).",
        ],
    }
    if out.get("ode_vs_gru"):
        g = out["ode_vs_gru"]["mean_endpoint"]
        report["ode_vs_gru"] = {
            "ode_mean": g["left_mean"],
            "gru_mean": g["right_mean"],
            "diff_mean": g["diff_mean"],
            "p_value": g["p_value"],
        }
    return report


def main() -> None:
    print("=" * 70)
    print("Endpoint-aligned extrapolation significance")
    print(f"  data: {cfg.DATA_DIR}")
    print(f"  train: {cfg.TRAIN_TIMES}, holdout/extrap: {cfg.TARGET_TIMES}")
    print(f"  source: {cfg.SOURCE_TIME}h, seeds: {cfg.SEEDS}, n_genes: {cfg.N_GENES}")
    print("=" * 70)

    x, times = load_data()

    seed_details = {}
    ode_by_seed = {}
    gru_by_seed = {}
    for s in cfg.SEEDS:
        print(f"\nSeed {s}: fitting baselines...")
        seed_details[str(s)] = run_single_seed(s, x, times)
        ode_by_seed[str(s)] = load_model_seed_results("ode", s)
        gru_path = cfg.model_result_path("gru", s)
        if gru_path.exists():
            gru_by_seed[str(s)] = load_model_seed_results("gru", s)

    paired_significance = build_baseline_summary(ode_by_seed, seed_details)

    ode_vs_gru = None
    if len(gru_by_seed) == len(cfg.SEEDS):
        ode_vals = [ode_by_seed[str(s)]["mean_extrapolation_corr"] for s in cfg.SEEDS]
        gru_vals = [gru_by_seed[str(s)]["mean_extrapolation_corr"] for s in cfg.SEEDS]
        ode_vs_gru = {"mean_endpoint": paired_stats(ode_vals, gru_vals)}

    out = {
        "seeds": cfg.SEEDS,
        "config": {
            "data_dir": str(cfg.DATA_DIR),
            "train_times": cfg.TRAIN_TIMES,
            "target_times": cfg.TARGET_TIMES,
            "source_time": cfg.SOURCE_TIME,
            "max_time": cfg.MAX_TIME,
            "n_genes": cfg.N_GENES,
            "result_tag": cfg.RESULT_TAG,
            "endpoint": cfg.endpoint_label(),
        },
        "ode_seed_results": ode_by_seed,
        "gru_seed_results": gru_by_seed or None,
        "baseline_seed_results": seed_details,
        "paired_significance": paired_significance,
        "ode_vs_gru": ode_vs_gru,
    }

    out_path = cfg.output_path()
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved: {out_path}")

    summary = build_summary_report(out)
    summary_name = (
        f"stage2_{cfg.RESULT_TAG}_extrapolation_significance_summary.json"
        if cfg.RESULT_TAG
        else "stage1_extrapolation_significance_summary.json"
    )
    summary_path = BASE_DIR / "artifacts" / "reports" / summary_name
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"saved: {summary_path}")

    print("\nPaired significance (ODE > baseline, mean endpoint):")
    for name, row in summary["paired_significance_mean_endpoint"].items():
        print(
            f"  {name:25s}  baseline={row['baseline_mean']:.4f}  "
            f"ODE={row['ode_mean']:.4f}  p={row['p_value']:.4g}"
        )
    if summary.get("ode_vs_gru"):
        g = summary["ode_vs_gru"]
        print(
            f"\nODE vs GRU: ODE={g['ode_mean']:.4f}  GRU={g['gru_mean']:.4f}  p={g['p_value']:.4g}"
        )


if __name__ == "__main__":
    main()
