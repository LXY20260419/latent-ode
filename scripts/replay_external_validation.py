#!/usr/bin/env python3
"""Replay diagnostic + validation stages from existing checkpoints.

Why this exists
---------------
The 2026-10-02 phase-3 confirmation runs for GSE149622 seeds 7 and 13 finished
all 100 training epochs and wrote both checkpoints, but the process exited
with status 1 during the post-training validation stage. The host journal
shows a ``/tmp`` (tmpfs with usrquota) EDQUOT episode at that time; the run
logs are truncated precisely at 4096/8192-byte page boundaries, which is
consistent with a failed stdout flush rather than a model/data error.

Re-running the shared training entry point would retrain from scratch, so this
tool instead replays only the deterministic post-training stages
(``DiagnosticAnalyzer`` + ``TripleValidation`` + figures + report) against the
checkpoints that the completed training left on disk. It imports the training
entry point as a module so the model classes, validation code and filename
rules are exactly the audited ones; the entry point itself is never modified
and its sha256 is recorded in the resulting artifacts.

The replayed validation artifacts carry the checkpoint's own ``run_metadata``
(dataset/protocol/config/source hashes), so they are attributable in exactly
the same way as a normal run.

Usage (env must match the original run):

    LEWORLD_DATA_DIR=data/leworld_input_gse149622 \
    LEWORLD_RESULT_TAG=ext_gse149622_v2 LEWORLD_SEED=7 \
    ... python scripts/replay_external_validation.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "ode_benchmark" / "train_ode_benchmark.py"


def load_training_module():
    spec = importlib.util.spec_from_file_location("train_ode_benchmark", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_ode_benchmark"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    import numpy as np
    import torch

    module = load_training_module()
    cfg = module.cfg

    seed = module.RUN_SEED
    tag = module.RESULT_TAG
    if not tag:
        raise SystemExit("LEWORLD_RESULT_TAG must be set for a replay run")

    print(f"🔁 checkpoint replay: tag={tag} seed={seed} data={cfg.DATA_DIR}")

    # 1. rebuild the exact data context used by the original run
    expression_matrix = np.load(cfg.DATA_DIR / "expression_matrix.npy")
    metadata = module.pd.read_csv(cfg.DATA_DIR / "metadata.csv", index_col=0)
    with open(cfg.DATA_DIR / "gene_names.txt", "r") as handle:
        gene_names = [line.strip() for line in handle]
    priors = module.load_biology_priors(cfg.BIOLOGY_PRIORS_PATH)
    priors["source"] = {"base_json": module.portable_path(cfg.BIOLOGY_PRIORS_PATH)}
    ext_pathways = module.load_gene_sets_from_gmt(cfg.EXTERNAL_PATHWAY_GMT)
    ext_tf_targets = module.load_tf_targets_from_csv(cfg.EXTERNAL_TF_TARGET_CSV)
    priors = module.merge_priors(
        priors, extra_pathways=ext_pathways, extra_tf_targets=ext_tf_targets
    )
    priors["source"]["external_gmt"] = (
        module.portable_path(cfg.EXTERNAL_PATHWAY_GMT) if ext_pathways else None
    )
    priors["source"]["external_tf_csv"] = (
        module.portable_path(cfg.EXTERNAL_TF_TARGET_CSV) if ext_tf_targets else None
    )
    biology_context = module.match_gene_sets(
        gene_names, priors, min_genes=cfg.PATHWAY_MIN_GENES
    )
    biology_context["source"] = priors.get("source", {})
    cfg.INPUT_DIM = expression_matrix.shape[1]
    dataset = module.SCTimeSeriesDataset(expression_matrix, metadata, gene_names)

    all_times = sorted(dataset.time_to_cells.keys())
    train_times = [t for t in all_times if t in cfg.TRAIN_TIMEPOINTS]
    eligible_train_idx = module.build_train_indices(dataset, train_times)
    train_idx, val_idx = module.split_internal_validation(
        eligible_train_idx,
        dataset.timepoints,
        cfg.INTERNAL_VAL_FRACTION,
        seed,
    )
    holdout_idx = [
        i for t in cfg.HOLDOUT_TIMEPOINTS for i in dataset.get_cells_by_timepoint(t)
    ]
    if set(train_idx) & set(holdout_idx) or set(val_idx) & set(holdout_idx):
        raise RuntimeError("hold-out cells leaked into training or checkpoint selection")
    run_metadata = module.build_run_metadata(
        dataset, gene_names, train_times, train_idx, val_idx
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. load the checkpoints the completed training left on disk
    ckpt_paths = {
        "gru": module.checkpoint_path("gru"),
        "ode": module.checkpoint_path("ode"),
    }
    checkpoints = {}
    for model_name, path in ckpt_paths.items():
        if not path.is_file():
            raise SystemExit(f"missing checkpoint for {model_name}: {path}")
        checkpoints[model_name] = torch.load(path, map_location=device, weights_only=True)
        print(
            f"📥 loaded {model_name} checkpoint: {path.name} "
            f"(epoch={checkpoints[model_name].get('epoch')}, "
            f"val_loss={checkpoints[model_name].get('val_loss')})"
        )

    model_gru = module.LEWorldModel_GRU(cfg)
    model_gru.load_state_dict(checkpoints["gru"]["model_state_dict"])
    model_ode = module.LEWorldModel_ODE(cfg)
    model_ode.load_state_dict(checkpoints["ode"]["model_state_dict"])

    # 3. diagnostics (ODE), exactly as the original main()
    print("\n🔍 运行诊断...")
    diag = module.DiagnosticAnalyzer(model_ode, dataset, gene_names, cfg, device=device)
    diag_payload = diag.run_diagnosis()
    with open(
        cfg.RESULT_DIR / module.tagged_filename("diagnosis_results", "json", tag, f"seed{seed}"),
        "w",
    ) as handle:
        json.dump(module.convert_to_native(diag_payload), handle, indent=2, cls=module.NumpyEncoder)

    # 4. validation for both arms
    print("\n🎯 运行验证...")
    val_ode = module.TripleValidation(
        model_ode,
        dataset,
        gene_names,
        cfg,
        biology_context=biology_context,
        run_metadata=run_metadata,
        checkpoint=ckpt_paths["ode"],
        device=device,
    )
    val_ode.run_all(module.result_suffix("ode"))
    val_gru = module.TripleValidation(
        model_gru,
        dataset,
        gene_names,
        cfg,
        biology_context=biology_context,
        run_metadata=run_metadata,
        checkpoint=ckpt_paths["gru"],
        device=device,
    )
    val_gru.run_all(module.result_suffix("gru"))

    # 5. figures and benchmark report (same helpers as main(); val losses from
    #    the checkpoints because no trainer objects exist in a replay)
    module.plot_gap_decay_comparison(
        model_gru, model_ode, dataset, gene_names, cfg, device, cfg.RESULT_DIR
    )
    module.plot_gene_trajectory(
        model_ode, dataset, gene_names, cfg, device, cfg.RESULT_DIR, gene_list=cfg.LPS_GENES
    )
    gap_fig = module.tagged_filename("gap_decay_comparison", "png", tag)
    traj_glob = module.tagged_filename("*_trajectory", "png", tag)
    report = f"""
# LEWorldModel Benchmark Report (v2.0)
## Training Results
- GRU Best Val Loss: {checkpoints['gru'].get('val_loss'):.4f}
- ODE Best Val Loss: {checkpoints['ode'].get('val_loss'):.4f}
## Validation (Hold-out)
- GRU Correlation: {val_gru.results.get('time_holdout', {}).get('mean_correlation', 'N/A'):.3f}
- ODE Correlation: {val_ode.results.get('time_holdout', {}).get('mean_correlation', 'N/A'):.3f}
## Mechanistic Consistency
- GRU TF direction consistency: {val_gru.results.get('tf_direction_consistency', {}).get('overall', 'N/A')}
- ODE TF direction consistency: {val_ode.results.get('tf_direction_consistency', {}).get('overall', 'N/A')}
- GRU pathway order accuracy: {val_gru.results.get('pathway_activation_order', {}).get('overall', 'N/A')}
- ODE pathway order accuracy: {val_ode.results.get('pathway_activation_order', {}).get('overall', 'N/A')}
## Gap Decay Analysis
- See: {gap_fig}
## Gene Trajectories
- See: {traj_glob} for {cfg.LPS_GENES}
"""
    for suffix in ("", f"seed{seed}"):
        with open(
            cfg.RESULT_DIR / module.tagged_filename("benchmark_report", "md", tag, suffix),
            "w",
        ) as handle:
            handle.write(report)

    print("\n✅ replay complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
