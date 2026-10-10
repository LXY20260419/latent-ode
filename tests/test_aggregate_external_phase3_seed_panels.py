#!/usr/bin/env python3
"""Seed-panel contract for scripts/aggregate_external_phase3.py.

``external_phase3_preregistration.json`` expanded the two mouse candidates to
five seeds, while the GSE305979 protocol locks three seeds and reserves its
7/13 extension for a separate decision.  The aggregate must therefore resolve
one seed panel per dataset: a single global list would either drop GSE305979
as ``pending`` or silently downgrade the five-seed datasets to three seeds.
Every panel here was locked before its results existed, so a shrinking
request has to fail loudly rather than quietly restate weaker evidence.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import aggregate_external_phase3 as agg  # noqa: E402


class SeedPanelTests(unittest.TestCase):
    def test_mouse_datasets_keep_the_preregistered_five_seeds(self):
        for accession in ("GSE148794", "GSE149622"):
            self.assertEqual(
                agg.seeds_for(accession, agg.PHASE3_SEEDS),
                [42, 123, 2026, 7, 13],
            )

    def test_gse305979_uses_its_locked_three_seed_panel(self):
        self.assertEqual(
            agg.seeds_for("GSE305979", agg.PHASE3_SEEDS), [42, 123, 2026]
        )

    def test_requesting_a_panel_that_drops_a_locked_seed_raises(self):
        with self.assertRaisesRegex(ValueError, "preregisters seeds"):
            agg.seeds_for("GSE305979", [42, 123])

    def test_mouse_datasets_refuse_a_narrowed_panel(self):
        # The tempting "pass --seeds 42,123,2026 so the third dataset
        # completes" workaround must not quietly shrink the two five-seed
        # datasets that are already committed.
        for accession in ("GSE148794", "GSE149622"):
            with self.assertRaisesRegex(ValueError, "preregisters seeds"):
                agg.seeds_for(accession, [42, 123, 2026])

    def test_every_locked_panel_is_resolved_from_the_default_request(self):
        for accession, panel in agg.SEEDS_BY_DATASET.items():
            self.assertEqual(agg.seeds_for(accession, agg.PHASE3_SEEDS), panel)

    def test_unknown_dataset_follows_the_requested_panel(self):
        self.assertEqual(agg.seeds_for("GSE000000", [1, 2]), [1, 2])

    def test_every_arm_spec_resolves_to_a_nonempty_panel(self):
        for accession in agg.ARM_SPECS:
            self.assertTrue(agg.seeds_for(accession, agg.PHASE3_SEEDS))


if __name__ == "__main__":
    unittest.main()
