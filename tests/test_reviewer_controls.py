#!/usr/bin/env python3
"""Pure-stdlib tests for reviewer-control preparation and aggregation."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import reviewer_controls as controls  # noqa: E402


def validation_payload(
    *,
    seed: int,
    tag: str,
    corr36: float,
    corr48: float,
    pathway_weight: float = 0.2,
    pathway_delta_weight: float = 0.1,
    priors_hash: str = "a" * 64,
) -> dict:
    return {
        "run_metadata": {
            "protocol": {
                "seed": seed,
                "result_tag": tag,
                "checkpoint_selection": {
                    "holdout_used_for_selection": False
                },
            },
            "config": {
                "pathway_weight": pathway_weight,
                "pathway_delta_weight": pathway_delta_weight,
            },
            "source_hashes": {
                "biology_priors_sha256": priors_hash,
            },
        },
        "time_extrapolation": {
            "mean_correlation": (corr36 + corr48) / 2,
            "pairs": {
                "36": {"from_time": 27, "corr": corr36},
                "48": {"from_time": 27, "corr": corr48},
            },
        },
    }


class ReviewerControlAggregationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="reviewer_controls_")
        self.addCleanup(self._tmp.cleanup)
        self.result_dir = Path(self._tmp.name)

    def write_seed(
        self,
        tag: str,
        seed: int,
        corr36: float,
        corr48: float,
        **kwargs,
    ) -> Path:
        path = controls.tagged_validation_path(
            self.result_dir, "ode", tag, seed
        )
        path.write_text(
            json.dumps(
                validation_payload(
                    seed=seed,
                    tag=tag,
                    corr36=corr36,
                    corr48=corr48,
                    **kwargs,
                )
            ),
            encoding="utf-8",
        )
        return path

    def test_uncertainty_reads_tagged_seed_files(self) -> None:
        tag = "primary_internal_freeze"
        for seed, corr36, corr48 in (
            (42, 0.6, 0.1),
            (123, 0.7, 0.2),
            (2026, 0.8, 0.3),
        ):
            self.write_seed(tag, seed, corr36, corr48)
        payload = controls.build_uncertainty(
            result_dir=self.result_dir,
            result_tag=tag,
            model="ode",
            seeds=[42, 123, 2026],
        )
        self.assertEqual(payload["status"], "completed")
        self.assertAlmostEqual(payload["horizons"]["36"]["mean"], 0.7)
        self.assertAlmostEqual(payload["horizons"]["48"]["mean"], 0.2)
        self.assertEqual(
            payload["horizons"]["36"]["bootstrap_resamples"], 27
        )
        self.assertEqual(len(payload["source_files"]), 3)

    def test_uncertainty_rejects_missing_and_wrong_source(self) -> None:
        tag = "primary_internal_freeze"
        self.write_seed(tag, 42, 0.6, 0.1)
        with self.assertRaises(FileNotFoundError):
            controls.build_uncertainty(
                result_dir=self.result_dir,
                result_tag=tag,
                model="ode",
                seeds=[42, 123],
            )
        path = controls.tagged_validation_path(
            self.result_dir, "ode", tag, 123
        )
        bad = validation_payload(
            seed=123,
            tag=tag,
            corr36=0.7,
            corr48=0.2,
        )
        bad["time_extrapolation"]["pairs"]["48"]["from_time"] = 3
        path.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "48h must extrapolate from 27h"):
            controls.build_uncertainty(
                result_dir=self.result_dir,
                result_tag=tag,
                model="ode",
                seeds=[42, 123],
            )

    def test_pure_mse_requires_both_zero_weights(self) -> None:
        tag = "pure_mse"
        self.write_seed(
            tag,
            42,
            0.5,
            0.2,
            pathway_weight=0.0,
            pathway_delta_weight=0.0,
        )
        payload = controls.build_pure_mse(
            result_dir=self.result_dir,
            result_tag=tag,
            model="ode",
            seeds=[42],
        )
        self.assertEqual(payload["config"]["pathway_weight"], 0.0)
        self.assertEqual(
            payload["config"]["pathway_delta_weight"], 0.0
        )
        bad_tag = "not_pure_mse"
        self.write_seed(
            bad_tag,
            42,
            0.5,
            0.2,
            pathway_weight=0.0,
            pathway_delta_weight=0.1,
        )
        with self.assertRaisesRegex(
            ValueError, "pathway_delta_weight must equal 0.0"
        ):
            controls.build_pure_mse(
                result_dir=self.result_dir,
                result_tag=bad_tag,
                model="ode",
                seeds=[42],
            )

    def test_random_prior_plan_is_reproducible_and_matched(self) -> None:
        genes = [f"G{i:03d}" for i in range(50)]
        priors = {
            "pathways": {
                "a": ["G001", "G002"],
                "b": ["G003", "G004", "G005"],
            },
            "tf_targets": {},
        }
        first = controls.generate_random_prior_plan(
            gene_names=genes,
            base_priors=priors,
            replicates=4,
            seed=7,
            output_dir=None,
            tag_prefix="random",
        )
        second = controls.generate_random_prior_plan(
            gene_names=genes,
            base_priors=priors,
            replicates=4,
            seed=7,
            output_dir=None,
            tag_prefix="random",
        )
        self.assertEqual(first, second)
        self.assertEqual(first["matched_sizes"], {"a": 2, "b": 3})
        for record in first["controls"]:
            flattened = [
                gene
                for values in record["pathways"].values()
                for gene in values
            ]
            self.assertEqual(len(flattened), len(set(flattened)))

    def test_random_prior_aggregation_uses_retrained_null(self) -> None:
        seeds = [42, 123]
        primary_tag = "primary_internal_freeze"
        for seed, value in ((42, 0.8), (123, 0.6)):
            self.write_seed(primary_tag, seed, value, value - 0.2)
        plan = {
            "status": "prepared",
            "matched_sizes": {"a": 2},
            "gene_universe_size": 10,
            "rng": {"seed": 7},
            "controls": [],
        }
        for replicate, null_value in enumerate((0.1, 0.3)):
            tag = f"random_r{replicate:03d}"
            digest = str(replicate + 1) * 64
            plan["controls"].append(
                {
                    "replicate": replicate,
                    "result_tag": tag,
                    "config_path": f"{tag}.json",
                    "config_sha256": digest,
                }
            )
            for seed in seeds:
                self.write_seed(
                    tag,
                    seed,
                    null_value,
                    null_value,
                    priors_hash=digest,
                )
        payload = controls.aggregate_random_prior_controls(
            plan=plan,
            result_dir=self.result_dir,
            model="ode",
            seeds=seeds,
            primary_result_tag=primary_tag,
        )
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["replicates"], 2)
        self.assertEqual(len(payload["per_control"]), 2)
        self.assertAlmostEqual(
            payload["comparison"]["composite"]["null_mean"], 0.2
        )
        self.assertAlmostEqual(
            payload["comparison"]["composite"]["empirical_p_ge"],
            1 / 3,
        )

    def test_exact_bootstrap_is_deterministic(self) -> None:
        first = controls.exact_seed_bootstrap([0.1, 0.2, 0.3])
        second = controls.exact_seed_bootstrap([0.1, 0.2, 0.3])
        self.assertEqual(first, second)
        self.assertEqual(first["bootstrap_resamples"], 27)


if __name__ == "__main__":
    unittest.main()
