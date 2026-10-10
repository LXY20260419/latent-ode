#!/usr/bin/env python3
"""Unit tests for scripts/build_failure_boundary_table.py (pure stdlib).

The failure-boundary table is the transparency artifact: every endpoint must
appear regardless of sign, and a flattering classification must be impossible
when the evidence does not support it. These tests pin the classifier
boundaries and the row construction from a realistic fixture.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import build_failure_boundary_table as table  # noqa: E402


class ClassifierTests(unittest.TestCase):
    def test_interval_above_zero_is_win(self):
        self.assertEqual(table.classify(0.1, 0.01, 0.2), "win")

    def test_interval_below_zero_is_loss(self):
        self.assertEqual(table.classify(-0.1, -0.2, -0.01), "loss")

    def test_interval_spanning_zero_is_tie(self):
        self.assertEqual(table.classify(0.1, -0.01, 0.2), "tie")

    def test_exact_zero_is_tie(self):
        self.assertEqual(table.classify(0.0, None, None), "tie")

    def test_positive_without_interval_is_point_estimate_only(self):
        self.assertEqual(table.classify(0.03, None, None), "positive_point_estimate")

    def test_negative_without_interval_is_point_estimate_only(self):
        self.assertEqual(table.classify(-0.03, None, None), "negative_point_estimate")

    def test_p_value_above_threshold_is_not_significant(self):
        self.assertEqual(table.classify_with_p(0.005, 0.145), "not_significant")

    def test_p_value_below_threshold_is_win_p_only(self):
        self.assertEqual(table.classify_with_p(0.68, 5.6e-5), "win_p_only")

    def test_null_inside_quantiles_is_null(self):
        self.assertEqual(table.classify_null_against_quantiles(0.45, 0.44, 0.46), "null")

    def test_null_above_quantiles_is_flagged(self):
        self.assertEqual(
            table.classify_null_against_quantiles(0.50, 0.44, 0.46), "above_null"
        )

    def test_null_below_quantiles_is_flagged(self):
        self.assertEqual(
            table.classify_null_against_quantiles(0.40, 0.44, 0.46), "below_null"
        )


def build_rows_with_fixture(external_pure_mse: dict | None = None) -> list[dict]:
    """Build boundary rows from a minimal temp fixture tree.

    Passing ``external_pure_mse`` writes the optional contrast file so the
    external mechanism-control block is exercised; omitting it pins the
    graceful-skip path.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        report_dir = root / "artifacts" / "reports"
        report_dir.mkdir(parents=True)
        (report_dir / "composite_drift_report.json").write_text(
            json.dumps(
                {
                    "endpoint_aligned_contrasts": {
                        "ode_vs_gru": {
                            "composite": {
                                "left_mean": 0.45,
                                "right_mean": -0.23,
                                "diff_mean": 0.68,
                                "p_value": 5.6e-5,
                            }
                        },
                        "ode_vs_persistence": {
                            "composite": {
                                "left_mean": 0.45,
                                "right_mean": 0.44,
                                "diff_mean": 0.004,
                                "p_value": 0.145,
                            }
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        (report_dir / "external_phase3_aggregate_v2.json").write_text(
            json.dumps({"datasets": {}}), encoding="utf-8"
        )
        (report_dir / "external_phase3_forest.json").write_text(
            json.dumps({"endpoints": []}), encoding="utf-8"
        )
        (report_dir / "pure_mse_vs_primary_comparison.json").write_text(
            json.dumps(
                {
                    "composite": {
                        "pure_mse": {"mean": 0.46},
                        "primary": {"mean": 0.45},
                        "paired_pure_minus_primary": {
                            "diff_mean": 0.01,
                            "ci95_difference": [-0.02, 0.04],
                            "p_two_sided": 0.5,
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        (report_dir / "random_gene_set_negative_control.json").write_text(
            json.dumps(
                {
                    "comparison": {
                        "composite": {
                            "observed": 0.45,
                            "null_mean": 0.454,
                            "null_q025": 0.448,
                            "null_q975": 0.462,
                            "empirical_p_ge": 0.67,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        for name in (
            "stage2_gse224518_3to5h_agg.json",
            "stage2_gse224518_3to5h_balanced_agg.json",
            "stage2_gse224518_agg.json",
            "stage2_gse226824_agg.json",
        ):
            (report_dir / name).write_text(
                json.dumps(
                    {
                        "agg3": {
                            "ode_extrap": {"mean": 0.13},
                            "gru_extrap": {"mean": 0.17},
                        }
                    }
                ),
                encoding="utf-8",
            )
        if external_pure_mse is not None:
            (report_dir / "external_pure_mse_contrast.json").write_text(
                json.dumps(external_pure_mse), encoding="utf-8"
            )

        original_root = table.ROOT
        original_report_dir = table.REPORT_DIR
        try:
            table.ROOT = root
            table.REPORT_DIR = report_dir
            return table.build_rows()
        finally:
            table.ROOT = original_root
            table.REPORT_DIR = original_report_dir


class RowConstructionTests(unittest.TestCase):
    def test_random_null_uses_quantiles_not_difference_ci(self):
        """The random-prior row must not treat the null spread as a diff CI."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "artifacts" / "reports"
            report_dir.mkdir(parents=True)
            (report_dir / "composite_drift_report.json").write_text(
                json.dumps(
                    {
                        "endpoint_aligned_contrasts": {
                            "ode_vs_gru": {
                                "composite": {
                                    "left_mean": 0.4516,
                                    "right_mean": -0.2326,
                                    "diff_mean": 0.6842,
                                    "p_value": 5.6e-5,
                                }
                            },
                            "ode_vs_persistence": {
                                "composite": {
                                    "left_mean": 0.4516,
                                    "right_mean": 0.4468,
                                    "diff_mean": 0.0048,
                                    "p_value": 0.145,
                                }
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            (report_dir / "external_phase3_aggregate_v2.json").write_text(
                json.dumps({"datasets": {}}), encoding="utf-8"
            )
            (report_dir / "external_phase3_forest.json").write_text(
                json.dumps({"endpoints": []}), encoding="utf-8"
            )
            (report_dir / "pure_mse_vs_primary_comparison.json").write_text(
                json.dumps(
                    {
                        "composite": {
                            "pure_mse": {"mean": 0.4571},
                            "primary": {"mean": 0.4516},
                            "paired_pure_minus_primary": {
                                "diff_mean": 0.0055,
                                "ci95_difference": [-0.0277, 0.0387],
                                "p_two_sided": 0.5486,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            (report_dir / "random_gene_set_negative_control.json").write_text(
                json.dumps(
                    {
                        "comparison": {
                            "composite": {
                                "observed": 0.4516,
                                "null_mean": 0.4543,
                                "null_q025": 0.4476,
                                "null_q975": 0.4623,
                                "empirical_p_ge": 0.6667,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            for name in (
                "stage2_gse224518_3to5h_agg.json",
                "stage2_gse224518_3to5h_balanced_agg.json",
                "stage2_gse224518_agg.json",
                "stage2_gse226824_agg.json",
            ):
                (report_dir / name).write_text(
                    json.dumps(
                        {
                            "agg3": {
                                "ode_extrap": {"mean": 0.13},
                                "gru_extrap": {"mean": 0.17},
                            }
                        }
                    ),
                    encoding="utf-8",
                )
            original_root = table.ROOT
            original_report_dir = table.REPORT_DIR
            try:
                table.ROOT = root
                table.REPORT_DIR = report_dir
                rows = table.build_rows()
            finally:
                table.ROOT = original_root
                table.REPORT_DIR = original_report_dir

        pure_row = next(r for r in rows if r["left_label"] == "ODE pure-MSE")
        self.assertEqual(pure_row["verdict"], "tie")
        self.assertEqual(pure_row["ci95"], [-0.0277, 0.0387])

        null_row = next(r for r in rows if r["right_label"] == "random-prior null mean")
        self.assertEqual(null_row["verdict"], "null")
        self.assertEqual(null_row["ci95"], [0.4476, 0.4623])

        pers_row = next(
            r
            for r in rows
            if r["layer"] == "internal_frozen" and "persistence" in r["comparator"]
        )
        self.assertEqual(pers_row["verdict"], "not_significant")

        gru_row = next(
            r
            for r in rows
            if r["layer"] == "internal_frozen" and r["comparator"] == "matched GRU"
        )
        self.assertEqual(gru_row["verdict"], "win_p_only")

    def test_external_pure_mse_rows_are_optional_and_kept_when_present(self):
        """A checkout without the external pure-MSE run still builds a table;
        when the contrast file exists, its rows appear with their verdicts."""
        rows_without = build_rows_with_fixture()
        self.assertFalse(
            any(
                r["layer"] == "mechanism_control_external" for r in rows_without
            )
        )
        rows_with = build_rows_with_fixture(
            external_pure_mse={
                "datasets": {
                    "FIXTURE": {
                        "endpoints": {
                            "12": {
                                "full": {"mean": 0.50},
                                "pure_mse": {"mean": 0.51},
                                "paired_pure_minus_full": {
                                    "diff_mean": 0.01,
                                    "ci95_difference": [-0.02, 0.04],
                                    "p_two_sided": 0.5,
                                },
                            }
                        }
                    }
                }
            }
        )
        external_rows = [
            r for r in rows_with if r["layer"] == "mechanism_control_external"
        ]
        self.assertEqual(len(external_rows), 1)
        self.assertEqual(external_rows[0]["verdict"], "tie")


if __name__ == "__main__":
    unittest.main()
