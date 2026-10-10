#!/usr/bin/env python3
"""Unit tests for scripts/population_endpoint_bootstrap.py.

The percentile helper and the pure-stdlib paths are exercised directly. Tests
that need the bootstrap resampling are skipped when numpy is unavailable.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import population_endpoint_bootstrap as bootstrap  # noqa: E402

try:
    import numpy  # noqa: F401

    HAVE_NUMPY = True
except ImportError:  # pragma: no cover - numpy is required to run the analysis
    HAVE_NUMPY = False


class PercentileTests(unittest.TestCase):
    def test_matches_numpy_style_linear_interpolation(self):
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(bootstrap.percentile(values, 0.0), 1.0)
        self.assertAlmostEqual(bootstrap.percentile(values, 1.0), 4.0)
        self.assertAlmostEqual(bootstrap.percentile(values, 0.5), 2.5)

    def test_clamps_out_of_range_probabilities(self):
        values = [5.0, 6.0]
        self.assertAlmostEqual(bootstrap.percentile(values, -1.0), 5.0)
        self.assertAlmostEqual(bootstrap.percentile(values, 2.0), 6.0)

    def test_single_value_sample(self):
        self.assertAlmostEqual(bootstrap.percentile([7.0], 0.5), 7.0)

    def test_empty_sample_is_rejected(self):
        with self.assertRaises(ValueError):
            bootstrap.percentile([], 0.5)


class AcrossGenePearsonTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_NUMPY, "numpy is required")
    def test_matches_scipy_reference(self):
        import numpy as np
        from scipy.stats import pearsonr

        rng = np.random.default_rng(0)
        left = rng.normal(size=50)
        right = left * 0.5 + rng.normal(size=50)
        self.assertAlmostEqual(
            bootstrap.across_gene_pearson(left, right),
            float(pearsonr(left, right)[0]),
            places=10,
        )

    @unittest.skipUnless(HAVE_NUMPY, "numpy is required")
    def test_constant_profile_returns_nan(self):
        import numpy as np

        values = np.ones(10)
        self.assertTrue(
            np.isnan(bootstrap.across_gene_pearson(values, values))
        )


class PrimaryContextTests(unittest.TestCase):
    def _write(self, result_dir: Path, tag: str, seed: int, corr36, corr48):
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / f"validation_results_ode_{tag}_seed{seed}.json").write_text(
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

    def test_returns_none_when_a_seed_file_is_missing(self):
        with tempfile.TemporaryDirectory() as raw:
            result_dir = Path(raw)
            self._write(result_dir, "tag", 42, 0.6, 0.2)
            self.assertIsNone(bootstrap.primary_context(result_dir, "tag"))

    def test_collects_per_seed_horizon_values(self):
        with tempfile.TemporaryDirectory() as raw:
            result_dir = Path(raw)
            for seed, (a, b) in {
                42: (0.60, 0.20),
                123: (0.62, 0.22),
                2026: (0.64, 0.24),
            }.items():
                self._write(result_dir, "tag", seed, a, b)
            context = bootstrap.primary_context(result_dir, "tag")
        self.assertEqual(sorted(context), ["36", "48"])
        self.assertEqual(len(context["36"]), 3)
        self.assertAlmostEqual(context["36"][0], 0.60)
        self.assertAlmostEqual(context["48"][2], 0.24)


if __name__ == "__main__":
    unittest.main()
