#!/usr/bin/env python3
"""Unit tests for scripts/corrected_baseline_significance.py (pure stdlib)."""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import corrected_baseline_significance as correction  # noqa: E402


def write_legacy(path: Path, ablation: dict, baselines: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "ode_seed_results": {
                    str(seed): {"mean_extrapolation_corr": value}
                    for seed, value in ablation.items()
                },
                "baseline_seed_results": {
                    str(seed): {
                        name: {"mean_extrapolation_corr": values[seed]}
                        for name, values in baselines.items()
                    }
                    for seed in ablation
                },
            }
        ),
        encoding="utf-8",
    )


def write_primary(result_dir: Path, tag: str, values: dict) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    for seed, value in values.items():
        (result_dir / f"validation_results_ode_{tag}_seed{seed}.json").write_text(
            json.dumps(
                {"time_extrapolation": {"mean_correlation": value}}
            ),
            encoding="utf-8",
        )


class PairedTests(unittest.TestCase):
    def test_closed_form_matches_t_table_critical_value(self):
        # t(0.95, df=2) = 2.919986 => one-sided p = 0.05
        self.assertAlmostEqual(correction.t2_sf(2.919986), 0.05, places=7)

    def test_zero_variance_positive_difference_is_maximally_significant(self):
        result = correction.paired_one_sided([0.02, 0.02, 0.02])
        self.assertEqual(result["p_one_sided_greater"], 0.0)

    def test_negative_difference_is_not_significant(self):
        result = correction.paired_one_sided([-0.05, -0.04, -0.06])
        self.assertGreater(result["p_one_sided_greater"], 0.5)


class BuildReportTests(unittest.TestCase):
    def test_separates_ablation_arm_from_primary_arm(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            legacy = tmp / "extrapolation_significance.json"
            result_dir = tmp / "recovered"
            baselines = {"Naive": {42: 0.44, 123: 0.44, 2026: 0.44}}
            write_legacy(
                legacy, {42: 0.476, 123: 0.511, 2026: 0.500}, baselines
            )
            write_primary(
                result_dir, "recovered", {42: 0.474, 123: 0.481, 2026: 0.495}
            )
            report = correction.build_report(
                legacy_path=legacy,
                result_dir=result_dir,
                result_tag="recovered",
            )
        self.assertAlmostEqual(report["ablation_arm_mean"], 0.495667, places=5)
        self.assertAlmostEqual(report["primary_arm_mean"], 0.483333, places=5)
        block = report["baselines"]["Naive"]
        # The two arms must produce different p-values; that difference is the
        # whole point of the correction.
        self.assertNotAlmostEqual(
            block["vs_legacy_ablation_arm"]["p_one_sided_greater"],
            block["vs_primary_arm"]["p_one_sided_greater"],
            places=4,
        )
        self.assertLess(
            block["vs_primary_arm"]["p_one_sided_greater"], 0.05
        )

    def test_records_panel_mismatch_caveat(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            legacy = tmp / "extrapolation_significance.json"
            result_dir = tmp / "recovered"
            write_legacy(
                legacy,
                {42: 0.4, 123: 0.4, 2026: 0.4},
                {"Naive": {42: 0.3, 123: 0.3, 2026: 0.3}},
            )
            write_primary(
                result_dir, "recovered", {42: 0.4, 123: 0.4, 2026: 0.4}
            )
            report = correction.build_report(
                legacy_path=legacy,
                result_dir=result_dir,
                result_tag="recovered",
            )
        joined = " ".join(report["caveats"])
        self.assertIn(str(correction.BASELINE_GENES), joined)
        self.assertIn(str(correction.PRIMARY_GENES), joined)

    def test_records_legacy_source_digest(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            legacy = tmp / "extrapolation_significance.json"
            result_dir = tmp / "recovered"
            write_legacy(
                legacy,
                {42: 0.4, 123: 0.4, 2026: 0.4},
                {"Naive": {42: 0.3, 123: 0.3, 2026: 0.3}},
            )
            write_primary(
                result_dir, "recovered", {42: 0.4, 123: 0.4, 2026: 0.4}
            )
            report = correction.build_report(
                legacy_path=legacy,
                result_dir=result_dir,
                result_tag="recovered",
            )
        digest = report["sources"]["legacy_significance"]["sha256"]
        self.assertEqual(len(digest), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in digest))

    def test_missing_primary_seed_raises(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            legacy = tmp / "extrapolation_significance.json"
            result_dir = tmp / "recovered"
            write_legacy(
                legacy,
                {42: 0.4, 123: 0.4, 2026: 0.4},
                {"Naive": {42: 0.3, 123: 0.3, 2026: 0.3}},
            )
            write_primary(result_dir, "recovered", {42: 0.4, 123: 0.4})
            with self.assertRaises(FileNotFoundError):
                correction.build_report(
                    legacy_path=legacy,
                    result_dir=result_dir,
                    result_tag="recovered",
                )


if __name__ == "__main__":
    unittest.main()
