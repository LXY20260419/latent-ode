#!/usr/bin/env python3
"""Unit tests for scripts/external_pure_mse_contrast.py (pure stdlib).

Two properties must stay pinned: a provenance mismatch aborts instead of
producing a number, and a pure-MSE arm that is nominally *better* is surfaced
as such (not hidden) with an interval that spans zero reported as
"indistinguishable".
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import external_pure_mse_contrast as contrast  # noqa: E402


def write_artifact(
    result_dir: Path,
    tag: str,
    model: str,
    seed: int,
    *,
    composite: float,
    horizon: int,
    horizon_value: float,
    pathway_weight: float,
    provenance_tag: str | None = None,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "time_extrapolation": {
            "mean_correlation": composite,
            "pairs": {
                str(horizon): {
                    "from_time": 0,
                    "corr": horizon_value,
                    "predict_steps": 1,
                }
            },
        },
        "time_holdout": {"mean_correlation": 0.3},
        "run_metadata": {
            "protocol": {
                "seed": seed,
                "result_tag": provenance_tag or tag,
            },
            "config": {
                "pathway_weight": pathway_weight,
                "pathway_delta_weight": 0.1 if pathway_weight else 0.0,
            },
            "dataset": {"n_genes": 2991},
        },
    }
    (result_dir / f"validation_results_{model}_{tag}_seed{seed}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def populate(result_dir: Path) -> None:
    for seed in (42, 123, 2026):
        write_artifact(
            result_dir,
            "ext_fix_v1",
            "ode",
            seed,
            composite=0.45,
            horizon=4,
            horizon_value=0.40,
            pathway_weight=0.2,
        )
        write_artifact(
            result_dir,
            "ext_fix_v1",
            "gru",
            seed,
            composite=-0.20,
            horizon=4,
            horizon_value=-0.10,
            pathway_weight=0.2,
        )
        # pure-MSE arm is nominally *better* in this fixture
        write_artifact(
            result_dir,
            "ext_fix_puremse_v1",
            "ode",
            seed,
            composite=0.46,
            horizon=4,
            horizon_value=0.41,
            pathway_weight=0.0,
        )
        write_artifact(
            result_dir,
            "ext_fix_puremse_v1",
            "gru",
            seed,
            composite=-0.18,
            horizon=4,
            horizon_value=-0.09,
            pathway_weight=0.0,
        )


class ExternalPureMseContrastTests(unittest.TestCase):
    def _with_fixture(self, mutate=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "artifacts" / "results" / "ode_benchmark"
            populate(result_dir)
            if mutate:
                mutate(result_dir)
            orig_result = contrast.RESULT_DIR
            orig_arms = contrast.ARMS
            orig_root = contrast.ROOT
            try:
                contrast.RESULT_DIR = result_dir
                contrast.ROOT = root
                contrast.ARMS = {
                    "FIXTURE": {
                        "full_tag": "ext_fix_v1",
                        "pure_mse_tag": "ext_fix_puremse_v1",
                        "endpoints": [4],
                        "primary_endpoint": 4,
                        "secondary_endpoint": None,
                    }
                }
                payload = contrast.build()
            finally:
                contrast.RESULT_DIR = orig_result
                contrast.ARMS = orig_arms
                contrast.ROOT = orig_root
        return payload

    def test_pure_mse_better_is_surfaced(self):
        payload = self._with_fixture()
        headline = payload["headline"]["FIXTURE"]
        self.assertGreater(headline["pure_mse_minus_full"], 0)
        self.assertIn("pure_mse_better", headline["verdict"])
        # endpoint-level detail is kept too
        endpoint = payload["datasets"]["FIXTURE"]["endpoints"]["4"]
        self.assertGreater(
            endpoint["paired_pure_minus_full"]["diff_mean"], 0
        )

    def test_provenance_mismatch_aborts(self):
        def mutate(result_dir: Path) -> None:
            write_artifact(
                result_dir,
                "ext_fix_puremse_v1",
                "ode",
                123,
                composite=0.46,
                horizon=4,
                horizon_value=0.41,
                pathway_weight=0.0,
                provenance_tag="wrong_tag",
            )

        with self.assertRaises(SystemExit):
            self._with_fixture(mutate)

    def test_interval_spanning_zero_reads_indistinguishable(self):
        def mutate(result_dir: Path) -> None:
            # make pure-MSE differences asymmetric so the CI spans zero
            for seed, value in ((42, 0.50), (123, 0.30), (2026, 0.40)):
                write_artifact(
                    result_dir,
                    "ext_fix_puremse_v1",
                    "ode",
                    seed,
                    composite=value,
                    horizon=4,
                    horizon_value=value,
                    pathway_weight=0.0,
                )

        payload = self._with_fixture(mutate)
        headline = payload["headline"]["FIXTURE"]
        self.assertTrue(headline["ci95_includes_zero"])
        self.assertEqual(headline["verdict"], "indistinguishable")


if __name__ == "__main__":
    unittest.main()
