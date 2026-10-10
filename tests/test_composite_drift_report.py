#!/usr/bin/env python3
"""Unit tests for scripts/composite_drift_report.py.

This report is the evidence behind two manuscript statements that are easy to
get wrong in the flattering direction: the rerun composite drifted down from
the locked value, and the ODE advantage over the trivial persistence control
is not significant at the composite endpoint. The tests pin the sign of the
drift, the significance verdicts, and that the report is regenerated from
artifacts rather than transcribed.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import composite_drift_report as drift  # noqa: E402


SEEDS = (42, 123, 2026)

# A zero-variance seed triple makes the paired t-test degenerate (scipy returns
# an infinite statistic) and the report refuses non-finite JSON, so every
# fixture carries a small deterministic per-seed offset exactly as real seeds
# do. Each set sums to zero so the arm means stay exactly where the test sets
# them. The two arms use different offsets, otherwise the ODE-minus-GRU
# difference would be constant and the GRU contrast would degenerate instead.
ODE_OFFSETS = {42: 0.010, 123: -0.006, 2026: -0.004}
GRU_OFFSETS = {42: -0.004, 123: 0.008, 2026: -0.004}


def spread(corr_36: float, corr_48: float, offsets: dict = None) -> dict:
    return {
        seed: {"36": corr_36 + offset, "48": corr_48 + offset}
        for seed, offset in (offsets or ODE_OFFSETS).items()
    }


def write_validation(
    result_dir: Path,
    tag: str,
    model: str,
    per_seed: dict,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    for seed, pairs in per_seed.items():
        payload = {
            "time_extrapolation": {
                "mean_correlation": sum(pairs.values()) / len(pairs),
                "pairs": {
                    horizon: {"corr": corr}
                    for horizon, corr in pairs.items()
                },
            }
        }
        (
            result_dir / f"validation_results_{model}_{tag}_seed{seed}.json"
        ).write_text(json.dumps(payload), encoding="utf-8")


class DriftReportTests(unittest.TestCase):
    def _run(self, ode_pairs, gru_pairs, persistence, locked):
        tmp_ctx = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_ctx.cleanup)
        tmp = Path(tmp_ctx.name)
        result_dir = tmp / "results"
        write_validation(result_dir, drift.TAG, "ode", ode_pairs)
        write_validation(result_dir, drift.TAG, "gru", gru_pairs)

        manifest = tmp / "manifest.json"
        manifest.write_text(
            json.dumps({"locked_claims": locked}), encoding="utf-8"
        )
        persistence_path = tmp / "persistence.json"
        persistence_path.write_text(
            json.dumps(persistence), encoding="utf-8"
        )
        output = tmp / "out.json"

        originals = (
            drift.RESULT_DIR,
            drift.MANIFEST,
            drift.PERSISTENCE,
            drift.OUTPUT,
        )
        drift.RESULT_DIR = result_dir
        drift.MANIFEST = manifest
        drift.PERSISTENCE = persistence_path
        drift.OUTPUT = output
        try:
            drift.main()
        finally:
            (
                drift.RESULT_DIR,
                drift.MANIFEST,
                drift.PERSISTENCE,
                drift.OUTPUT,
            ) = originals
        return json.loads(output.read_text(encoding="utf-8"))

    @staticmethod
    def _persistence(composite: float) -> dict:
        # 36h and 48h chosen so their mean is the stated composite.
        return {
            "pairs": {
                "36": {"corr": composite + 0.2},
                "48": {"corr": composite - 0.2},
            }
        }

    def test_drift_sign_follows_the_rerun_not_the_locked_value(self):
        # Rerun below the locked composite must produce a negative drift, which
        # is the actual situation: 0.4516 rerun versus 0.4833 locked.
        report = self._run(
            ode_pairs=spread(0.68, 0.22),
            gru_pairs=spread(-0.20, -0.26, GRU_OFFSETS),
            persistence=self._persistence(0.4467),
            locked={"ode_composite_mean": 0.4833, "gru_composite_mean": -0.1444},
        )
        self.assertAlmostEqual(report["rerun"]["ode"]["mean"], 0.45, places=6)
        self.assertAlmostEqual(
            report["drift"]["ode"]["absolute"], 0.45 - 0.4833, places=6
        )
        self.assertLess(report["drift"]["ode"]["absolute"], 0.0)
        self.assertLess(report["drift"]["ode"]["relative"], 0.0)

    def test_uninformative_persistence_margin_is_not_significant(self):
        # ODE just above persistence with a seed spread wide enough to swamp
        # it: this must be reported as not significant, matching the
        # manuscript's refusal to claim a persistence margin.
        report = self._run(
            ode_pairs=spread(0.68, 0.19),
            gru_pairs=spread(-0.20, -0.26, GRU_OFFSETS),
            persistence=self._persistence(0.4467),
            locked={"ode_composite_mean": 0.4833, "gru_composite_mean": -0.1444},
        )
        headline = report["headline_assessment"]
        self.assertFalse(headline["ode_vs_persistence_significant_at_0.05"])
        self.assertIn("not significant", headline["summary"])
        self.assertIn("must not claim", headline["summary"])

    def test_ode_versus_gru_separation_is_significant(self):
        report = self._run(
            ode_pairs=spread(0.68, 0.22),
            gru_pairs=spread(-0.20, -0.26, GRU_OFFSETS),
            persistence=self._persistence(0.4467),
            locked={"ode_composite_mean": 0.4833, "gru_composite_mean": -0.1444},
        )
        headline = report["headline_assessment"]
        self.assertTrue(headline["ode_vs_gru_significant_at_0.05"])
        self.assertTrue(headline["ode_above_persistence_composite"])
        contrast = report["endpoint_aligned_contrasts"]["ode_vs_gru"]["composite"]
        self.assertGreater(contrast["diff_mean"], 0.6)
        self.assertLess(contrast["p_value"], 0.01)

    def test_report_shape_is_stable(self):
        report = self._run(
            ode_pairs=spread(0.68, 0.22),
            gru_pairs=spread(-0.20, -0.26, GRU_OFFSETS),
            persistence=self._persistence(0.4467),
            locked={"ode_composite_mean": 0.4833, "gru_composite_mean": -0.1444},
        )
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["result_tag"], drift.TAG)
        self.assertEqual(report["seeds"], list(SEEDS))
        # Both horizons must be contrasted, not just the composite.
        horizons = report["endpoint_aligned_contrasts"][
            "ode_vs_persistence"
        ]["horizons"]
        self.assertEqual(sorted(horizons), ["36", "48"])


if __name__ == "__main__":
    unittest.main()
