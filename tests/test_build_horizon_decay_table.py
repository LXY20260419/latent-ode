#!/usr/bin/env python3
"""Unit tests for scripts/build_horizon_decay_table.py (pure stdlib).

The decay table must (a) keep every horizon even when GRU wins or ties and
(b) keep a single-horizon dataset as a single row rather than fabricating a
curve.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import build_horizon_decay_table as decay  # noqa: E402


def write_validation(
    result_dir: Path,
    tag: str,
    model: str,
    seed: int,
    pairs: dict,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / f"validation_results_{model}_{tag}_seed{seed}.json").write_text(
        json.dumps(
            {
                "time_extrapolation": {
                    "pairs": {
                        h: {"corr": v, "from_time": 0, "predict_steps": int(h)}
                        for h, v in pairs.items()
                    }
                }
            }
        ),
        encoding="utf-8",
    )


class HorizonDecayTests(unittest.TestCase):
    def _run_with_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "artifacts" / "results" / "ode_benchmark"
            report_dir = root / "artifacts" / "reports"
            report_dir.mkdir(parents=True)

            tag = "ext_fixture_v1"
            for seed in (1, 2):
                write_validation(result_dir, tag, "ode", seed, {"2": 0.5})
                write_validation(result_dir, tag, "gru", seed, {"2": 0.6})
            (report_dir / "external_persistence_v1.json").write_text(
                json.dumps(
                    {
                        "datasets": {
                            "FIXTURE": {
                                "horizons": {"2": {"corr": 0.55}}
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            orig = (decay.RESULT_DIR, decay.REPORT_DIR, decay.LAYERS, decay.PERSISTENCE, decay.ROOT)
            try:
                decay.RESULT_DIR = result_dir
                decay.REPORT_DIR = report_dir
                decay.ROOT = root
                decay.LAYERS = [
                    {
                        "key": "fixture",
                        "label": "FIXTURE",
                        "tag": tag,
                        "seeds": [1, 2],
                        "source_time": 0,
                        "time_unit": "d",
                    }
                ]
                decay.PERSISTENCE = {
                    "fixture": {
                        "path": "artifacts/reports/external_persistence_v1.json",
                        "kind": "external",
                        "accession": "FIXTURE",
                    }
                }
                rows = decay.build_rows()
            finally:
                (
                    decay.RESULT_DIR,
                    decay.REPORT_DIR,
                    decay.LAYERS,
                    decay.PERSISTENCE,
                    decay.ROOT,
                ) = orig
        return rows

    def test_gru_win_is_kept_with_negative_difference(self):
        rows = self._run_with_fixture()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertAlmostEqual(row["ode_minus_gru"], -0.1)

    def test_persistence_contrast_present(self):
        row = self._run_with_fixture()[0]
        self.assertAlmostEqual(row["persistence"], 0.55)
        self.assertAlmostEqual(row["ode_minus_persistence"], 0.5 - 0.55)

    def test_single_horizon_stays_single_row(self):
        rows = self._run_with_fixture()
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
