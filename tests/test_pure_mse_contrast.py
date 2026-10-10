#!/usr/bin/env python3
"""Unit tests for scripts/pure_mse_contrast.py (pure stdlib).

The contrast this script computes is the evidence behind the manuscript's
honest degradation of the pathway-prior claim, so two properties are pinned
here: a null contrast must be reported as null, and a provenance mismatch in
the source artifacts must abort rather than silently produce a number.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import pure_mse_contrast as contrast  # noqa: E402


SEEDS = (42, 123, 2026)


def write_arm(
    result_dir: Path,
    tag: str,
    model: str,
    per_seed: dict,
    *,
    pathway_weight: float,
    pathway_delta_weight: float,
    n_genes: int = 2991,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    for seed, values in per_seed.items():
        payload = {
            "time_extrapolation": {
                "mean_correlation": values["composite"],
                "pairs": {
                    horizon: {"corr": value}
                    for horizon, value in values["horizons"].items()
                },
            },
            "time_holdout": {"mean_correlation": values["reconstruction"]},
            "run_metadata": {
                "protocol": {"seed": seed, "result_tag": tag},
                "config": {
                    "pathway_weight": pathway_weight,
                    "pathway_delta_weight": pathway_delta_weight,
                },
                "dataset": {"n_genes": n_genes},
            },
        }
        (
            result_dir / f"validation_results_{model}_{tag}_seed{seed}.json"
        ).write_text(json.dumps(payload), encoding="utf-8")


# Per-seed dispersion is the whole reason the real contrast is uninformative,
# so the fixtures carry spread rather than three identical values: a constant
# arm would give a zero-variance paired difference and a degenerate interval.
PRIMARY_COMPOSITES = {42: 0.4437, 123: 0.4575, 2026: 0.4535}
PURE_COMPOSITES = {42: 0.4585, 123: 0.4478, 2026: 0.4651}


def arm_values(composites: dict) -> dict:
    return {
        seed: {
            "composite": value,
            "horizons": {"36": 0.68, "48": 0.22},
            "reconstruction": 0.35,
        }
        for seed, value in composites.items()
    }


class PairedStatisticsTests(unittest.TestCase):
    def test_t2_sf_matches_the_t_table_critical_value(self):
        # t(0.95, df=2) = 2.919986 => one-sided upper tail = 0.05
        self.assertAlmostEqual(contrast.t2_sf(2.919986), 0.05, places=7)

    def test_zero_difference_is_exactly_uninformative(self):
        result = contrast.paired([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.assertEqual(result["diff_mean"], 0.0)
        self.assertEqual(result["p_two_sided"], 1.0)
        self.assertLessEqual(result["ci95_difference"][0], 0.0)
        self.assertGreaterEqual(result["ci95_difference"][1], 0.0)

    def test_small_offset_inside_the_seed_spread_is_not_significant(self):
        # The real pure-MSE contrast has this shape: a tiny mean shift inside a
        # seed spread of comparable size, so it must not come out significant.
        primary = [0.4437, 0.4575, 0.4535]
        pure = [0.4585, 0.4478, 0.4651]
        result = contrast.paired(pure, primary)
        self.assertGreater(result["p_two_sided"], 0.05)
        self.assertLessEqual(result["ci95_difference"][0], 0.0)
        self.assertGreaterEqual(result["ci95_difference"][1], 0.0)

    def test_t_ci95_widens_with_spread(self):
        tight = contrast._t_ci95([0.5, 0.5001, 0.4999])
        wide = contrast._t_ci95([0.4, 0.5, 0.6])
        self.assertLess(tight[1] - tight[0], wide[1] - wide[0])


class LoadArmTests(unittest.TestCase):
    def test_loads_tagged_results_per_seed(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            result_dir = tmp / "results"
            write_arm(
                result_dir,
                "pure_mse",
                "ode",
                arm_values(PURE_COMPOSITES),
                pathway_weight=0.0,
                pathway_delta_weight=0.0,
            )
            original = contrast.RESULT_DIR
            contrast.RESULT_DIR = result_dir
            try:
                loaded = contrast.load_arm("pure_mse")
            finally:
                contrast.RESULT_DIR = original
        self.assertEqual(sorted(loaded), ["123", "2026", "42"])
        self.assertAlmostEqual(
            loaded["42"]["composite"], PURE_COMPOSITES[42]
        )
        self.assertEqual(loaded["42"]["pathway_weight"], 0.0)
        self.assertEqual(loaded["42"]["n_genes"], 2991)

    def test_provenance_mismatch_aborts(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            result_dir = tmp / "results"
            write_arm(
                result_dir,
                "pure_mse",
                "ode",
                arm_values(PURE_COMPOSITES),
                pathway_weight=0.0,
                pathway_delta_weight=0.0,
            )
            # Rewrite one seed so its recorded tag no longer matches its path.
            bad = result_dir / "validation_results_ode_pure_mse_seed123.json"
            payload = json.loads(bad.read_text(encoding="utf-8"))
            payload["run_metadata"]["protocol"]["result_tag"] = (
                "primary_internal_freeze"
            )
            bad.write_text(json.dumps(payload), encoding="utf-8")

            original = contrast.RESULT_DIR
            contrast.RESULT_DIR = result_dir
            try:
                with self.assertRaises(SystemExit):
                    contrast.load_arm("pure_mse")
            finally:
                contrast.RESULT_DIR = original

    def test_missing_seed_aborts(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            result_dir = tmp / "results"
            partial = arm_values(PURE_COMPOSITES)
            partial.pop(2026)
            write_arm(
                result_dir,
                "pure_mse",
                "ode",
                partial,
                pathway_weight=0.0,
                pathway_delta_weight=0.0,
            )
            original = contrast.RESULT_DIR
            contrast.RESULT_DIR = result_dir
            try:
                with self.assertRaises(SystemExit):
                    contrast.load_arm("pure_mse")
            finally:
                contrast.RESULT_DIR = original


class ReportTests(unittest.TestCase):
    """The published headline must follow from the numbers, not the reverse."""

    def _run(self, pure_shift: float) -> dict:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            result_dir = tmp / "results"
            shifted = {
                seed: value + pure_shift
                for seed, value in PURE_COMPOSITES.items()
            }
            for tag, model, composites, pw, pdw in (
                ("primary_internal_freeze", "ode", PRIMARY_COMPOSITES, 0.2, 0.1),
                ("primary_internal_freeze", "gru", PRIMARY_COMPOSITES, 0.2, 0.1),
                ("pure_mse", "ode", shifted, 0.0, 0.0),
                ("pure_mse", "gru", shifted, 0.0, 0.0),
            ):
                write_arm(
                    result_dir,
                    tag,
                    model,
                    arm_values(composites),
                    pathway_weight=pw,
                    pathway_delta_weight=pdw,
                )
            original_dir = contrast.RESULT_DIR
            original_out = contrast.OUTPUT
            original_root = contrast.ROOT
            contrast.RESULT_DIR = result_dir
            contrast.OUTPUT = tmp / "out.json"
            contrast.ROOT = tmp
            try:
                contrast.main()
            finally:
                contrast.RESULT_DIR = original_dir
                contrast.OUTPUT = original_out
                contrast.ROOT = original_root
            return json.loads((tmp / "out.json").read_text(encoding="utf-8"))

    def test_null_contrast_is_reported_as_no_measurable_benefit(self):
        report = self._run(pure_shift=0.004)
        headline = report["headline_assessment"]
        self.assertFalse(headline["pathway_priors_improve_primary_endpoint"])
        self.assertTrue(headline["ci95_includes_zero"])
        self.assertGreater(headline["p_two_sided"], 0.05)
        self.assertIn("must not attribute", headline["summary"])

    def test_report_keeps_the_panel_matched_caveat_free_framing(self):
        report = self._run(pure_shift=0.004)
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["panel_matched"])
        self.assertEqual(report["n_genes"], 2991)
        self.assertEqual(sorted(report["horizons"]), ["36", "48"])
        self.assertEqual(len(report["source_files"]), 12)
        self.assertEqual(
            report["config_contrast"]["pure_mse"]["pathway_weight"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
