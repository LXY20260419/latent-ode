#!/usr/bin/env python3
"""Unit tests for scripts/aggregate_mechanism_metrics.py (pure stdlib).

Two properties must stay pinned: a structurally not-evaluable metric is
recorded as missing for the affected seeds instead of being imputed, and a
missing source artifact aborts the aggregation rather than silently shrinking
the seed set.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import aggregate_mechanism_metrics as agg  # noqa: E402


def write_artifact(
    result_dir: Path,
    tag: str,
    model: str,
    seed: int,
    *,
    tf: float | None = 0.5,
    pathway: float | None = 0.5,
    marker: float = 0.0,
    pathway_present: bool = True,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "time_extrapolation": {
            "mean_correlation": 0.4,
            "pairs": {"1": {"from_time": 0, "corr": 0.4, "predict_steps": 1}},
        },
    }
    if tf is not None:
        payload["tf_direction_consistency"] = {
            "overall": tf,
            "per_tf": {"NFKB1": tf},
        }
    if pathway_present:
        payload["pathway_activation_order"] = {
            "overall": pathway,
            "per_pathway": {"NFkB_signaling": pathway},
        }
    payload["marker_rank_spearman"] = {
        "overall": marker,
        "per_set": {"early": marker},
    }
    (result_dir / f"validation_results_{model}_{tag}_seed{seed}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


class AggregateMechanismMetricsTests(unittest.TestCase):
    def test_summarize_ignores_none_and_keeps_all_finite_values(self):
        summary = agg.summarize([0.5, None, -0.5, 0.0])
        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["mean"], 0.0)

    def test_all_none_summarizes_to_empty(self):
        summary = agg.summarize([None, None])
        self.assertEqual(summary["n"], 0)
        self.assertIsNone(summary["mean"])

    def test_missing_metric_is_recorded_not_imputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            tag = "ext_test_v1"
            for seed in (1, 2):
                write_artifact(result_dir, tag, "ode", seed, pathway_present=False)
                write_artifact(result_dir, tag, "gru", seed)
            spec = {"label": "test", "tag": tag, "seeds": [1, 2]}
            layer = agg.aggregate_layer(result_dir, spec)
        ode_pathway = layer["arms"]["ode"][agg.PATHWAY_KEY]
        self.assertEqual(ode_pathway["summary"]["n"], 0)
        self.assertEqual(ode_pathway["not_evaluable_seeds"], [1, 2])
        gru_pathway = layer["arms"]["gru"][agg.PATHWAY_KEY]
        self.assertEqual(gru_pathway["summary"]["n"], 2)
        self.assertNotIn("not_evaluable_seeds", gru_pathway)

    def test_missing_artifact_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            tag = "ext_test_v1"
            write_artifact(result_dir, tag, "ode", 1)
            # GRU seed 1 missing entirely
            spec = {"label": "test", "tag": tag, "seeds": [1]}
            with self.assertRaises(FileNotFoundError):
                agg.aggregate_layer(result_dir, spec)

    def test_contrast_reports_none_when_an_arm_is_not_evaluable(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            tag = "ext_test_v1"
            for seed in (1, 2):
                write_artifact(result_dir, tag, "ode", seed, pathway_present=False)
                write_artifact(result_dir, tag, "gru", seed, pathway_present=False)
            spec = {"label": "test", "tag": tag, "seeds": [1, 2]}
            layer = agg.aggregate_layer(result_dir, spec)
        self.assertIsNone(
            layer["ode_minus_gru"][agg.PATHWAY_KEY]["ode_minus_gru_mean"]
        )


if __name__ == "__main__":
    unittest.main()
