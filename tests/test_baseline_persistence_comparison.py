#!/usr/bin/env python3
"""Unit tests for scripts/baseline_persistence_comparison.py (pure stdlib).

All fixtures are written to a temporary directory; the repo artifacts tree is
never read or written by these tests.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import baseline_persistence_comparison as comparison  # noqa: E402


def write_primary(path: Path, corr36: float, corr48: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "time_extrapolation": {
                    "pairs": {
                        "36": {"from_time": 27, "corr": corr36},
                        "48": {"from_time": 27, "corr": corr48},
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def write_persistence(path: Path, corr36: float, corr48: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metric": comparison.METRIC,
                "method": "population_persistence",
                "target_times": [36, 48],
                "pairs": {
                    "36": {"from_time": 27, "corr": corr36},
                    "48": {"from_time": 27, "corr": corr48},
                },
            }
        ),
        encoding="utf-8",
    )


class T2SurvivalFunctionTests(unittest.TestCase):
    def test_matches_reference_values(self):
        # Reference values from any t table with df = 2.
        for t_stat, expected in (
            (0.0, 0.5),
            (2.0, 0.09175170953613698),
            (4.302653, 0.025),
            (-4.302653, 0.975),
        ):
            self.assertAlmostEqual(
                comparison.t2_sf(t_stat), expected, places=6
            )

    def test_is_monotone_decreasing(self):
        values = [comparison.t2_sf(t) for t in (-4, -1, 0, 1, 4)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_critical_value_matches_t_table(self):
        # t(0.95, df=2) = 2.919986 => one-sided p = 0.05
        self.assertAlmostEqual(comparison.t2_sf(2.919986), 0.05, places=6)


class PairedStatsTests(unittest.TestCase):
    def test_three_seed_closed_form_matches_hand_calculation(self):
        stats = comparison.paired_stats([0.01, 0.02, 0.03])
        mean = 0.02
        sd = 0.01
        t_stat = mean / (sd / math.sqrt(3))
        self.assertAlmostEqual(stats["mean_difference"], mean)
        self.assertAlmostEqual(stats["t_statistic"], t_stat)
        self.assertAlmostEqual(
            stats["p_one_sided_greater"],
            0.5 - 0.5 * t_stat / math.sqrt(2 + t_stat**2),
        )

    def test_zero_variance_positive_is_significant(self):
        stats = comparison.paired_stats([0.05, 0.05, 0.05])
        self.assertEqual(stats["p_one_sided_greater"], 0.0)

    def test_non_frozen_seed_count_refuses_closed_form(self):
        stats = comparison.paired_stats([0.01, 0.02])
        self.assertIsNone(stats["p_one_sided_greater"])

    def test_ci_is_ordered(self):
        stats = comparison.paired_stats([0.01, 0.02, 0.03])
        low, high = stats["ci95_difference"]
        self.assertLess(low, stats["mean_difference"])
        self.assertLess(stats["mean_difference"], high)


class BuildReportTests(unittest.TestCase):
    def _build(self, tmp: Path, primary, persistence):
        result_dir = tmp / "results"
        persistence_path = tmp / "persistence_population.json"
        for seed, (corr36, corr48) in primary.items():
            write_primary(
                comparison.tagged_validation_path(
                    result_dir, "ode", "tag", seed
                ),
                corr36,
                corr48,
            )
        write_persistence(persistence_path, *persistence)
        return comparison.build_report(
            result_dir=result_dir,
            persistence_path=persistence_path,
            result_tag="tag",
            model="ode",
            seeds=[42, 123, 2026],
            horizons=["36", "48"],
            source_time=27,
        )

    def test_composite_is_within_seed_mean_of_horizons(self):
        with tempfile.TemporaryDirectory() as raw:
            report = self._build(
                Path(raw),
                {42: (0.60, 0.20), 123: (0.62, 0.22), 2026: (0.64, 0.24)},
                (0.55, 0.15),
            )
        # seed 42 composite = (0.60 + 0.20) / 2 = 0.40
        self.assertAlmostEqual(
            report["composite"]["primary_per_seed"]["42"], 0.40
        )
        self.assertAlmostEqual(report["composite"]["baseline_corr"], 0.35)
        self.assertAlmostEqual(report["composite"]["primary_mean"], 0.42)
        self.assertAlmostEqual(
            report["composite"]["paired_vs_baseline"]["mean_difference"], 0.07
        )

    def test_records_source_digests(self):
        with tempfile.TemporaryDirectory() as raw:
            report = self._build(
                Path(raw),
                {42: (0.6, 0.2), 123: (0.6, 0.2), 2026: (0.6, 0.2)},
                (0.5, 0.1),
            )
        for key in ("persistence", "ode_42", "ode_123", "ode_2026"):
            self.assertIn(key, report["source_files"])
            self.assertEqual(len(report["source_files"][key]["sha256"]), 64)

    def test_rejects_wrong_metric(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            persistence_path = tmp / "persistence_population.json"
            write_persistence(persistence_path, 0.5, 0.1)
            payload = json.loads(persistence_path.read_text())
            payload["metric"] = "cell_level_correlation"
            persistence_path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                comparison.load_persistence(
                    persistence_path, ["36", "48"], 27
                )

    def test_rejects_wrong_source_time(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            path = tmp / "validation_results_ode_tag_seed42.json"
            write_primary(path, 0.6, 0.2)
            payload = json.loads(path.read_text())
            payload["time_extrapolation"]["pairs"]["36"]["from_time"] = 16
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                comparison.load_primary_seed(path, ["36", "48"], 27)


if __name__ == "__main__":
    unittest.main()
