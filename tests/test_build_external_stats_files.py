#!/usr/bin/env python3
"""Unit tests for scripts/build_external_stats_files.py (pure stdlib).

The assembled stats files must carry the endpoint numbers unchanged from the
phase-3 aggregate (no re-rounding drift), keep the deterministic persistence
contrast, and state the mechanism-coupling caveat.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import build_external_stats_files as stats  # noqa: E402


class ExternalStatsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all_stats = {a: stats.build_stats(a) for a in stats.ACCESSIONS}
        # Accessions whose production runs have not landed yet are recorded as
        # pending and carry no horizon block; the structural assertions apply
        # only to the datasets whose numbers exist.
        cls.stats = {
            a: payload
            for a, payload in cls.all_stats.items()
            if payload.get("status") != "pending_production_runs"
        }

    def test_completed_datasets_present(self):
        for accession in ("GSE148794", "GSE149622"):
            self.assertIn(accession, self.stats)

    def test_ode_mean_matches_phase3_aggregate(self):
        phase3 = json.loads(
            (stats.REPORT_DIR / "external_phase3_aggregate_v2.json").read_text(
                encoding="utf-8"
            )
        )
        for accession, payload in self.stats.items():
            for horizon, hb in payload["horizons"].items():
                expected = phase3["datasets"][accession]["horizons"][horizon]["ode"][
                    "summary"
                ]["mean"]
                self.assertAlmostEqual(hb["ode"]["mean"], expected)

    def test_persistence_contrast_present(self):
        for payload in self.stats.values():
            for horizon, hb in payload["horizons"].items():
                self.assertIsNotNone(hb["persistence"])
                self.assertAlmostEqual(
                    hb["ode_minus_persistence"],
                    hb["ode"]["mean"] - hb["persistence"]["corr"],
                )

    def test_mechanism_coupling_note_present(self):
        for payload in self.stats.values():
            note = payload["mechanism_proxies"]["coupling_note"]
            self.assertIn("not independent", note)
            self.assertIn("training loss", note)

    def test_single_horizon_dataset_kept_as_single_horizon(self):
        self.assertEqual(len(self.stats["GSE149622"]["horizons"]), 1)

    def test_seed_panel_is_the_dataset_own_preregistered_panel(self):
        # The per-accession stats file must not inherit the aggregate-wide
        # seed list: that would claim seeds the dataset never ran.
        phase3 = json.loads(
            (
                stats.REPORT_DIR / "external_phase3_aggregate_v2.json"
            ).read_text(encoding="utf-8")
        )
        panels = phase3.get("seed_panels")
        if panels is None:
            # An aggregate written before per-dataset panels existed has only
            # the shared list; the fallback then applies to every dataset.
            panels = {accession: phase3["seeds"] for accession in self.stats}
        for accession, payload in self.stats.items():
            self.assertEqual(payload["seeds"], panels[accession])
            for horizon, hb in payload["horizons"].items():
                for model in ("ode", "gru"):
                    self.assertEqual(
                        sorted(int(s) for s in hb[model]["per_seed"]),
                        sorted(panels[accession]),
                    )

    def test_stats_seeds_falls_back_to_the_aggregate_seed_list(self):
        # A dataset without its own panel still follows the aggregate list.
        phase3 = {"seeds": [1, 2], "seed_panels": {"GSE305979": [42]}}
        self.assertEqual(stats.stats_seeds(phase3, "GSE305979"), [42])
        self.assertEqual(stats.stats_seeds(phase3, "GSE000000"), [1, 2])

    def test_pure_mse_block_matches_preregistered_contrast(self):
        contrast = json.loads(
            (
                stats.REPORT_DIR / "external_pure_mse_contrast.json"
            ).read_text(encoding="utf-8")
        )
        for accession, payload in self.stats.items():
            block = payload["pure_mse_contrast"]
            if accession not in contrast["datasets"]:
                # A dataset whose protocol gates pure-MSE on the primary
                # endpoint surviving review carries no pure-MSE arm; that is
                # a recorded decision, not a missing contrast.
                self.assertIsNone(block)
                continue
            self.assertIsNotNone(block)
            for endpoint, eb in block["endpoints"].items():
                expected = contrast["datasets"][accession]["endpoints"][endpoint][
                    "paired_pure_minus_full"
                ]
                self.assertAlmostEqual(
                    eb["paired_pure_minus_full"]["diff_mean"],
                    expected["diff_mean"],
                )


if __name__ == "__main__":
    unittest.main()
