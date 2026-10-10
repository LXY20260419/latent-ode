#!/usr/bin/env python3
"""Unit tests for scripts/audit_freeze_integrity.py (pure stdlib).

All validation JSON fixtures are written to a temporary directory; the repo
artifacts/ tree is never read or written by these tests.
"""
from __future__ import annotations

import json
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import audit_freeze_integrity as audit  # noqa: E402


MANIFEST = {
    "dataset": {
        "dataset_id": "internal_lps_primary",
        "source_accession": "GSE151658",
        "n_cells": 38439,
        "n_genes": 2991,
        "timepoints": [0, 1, 4, 16, 27, 36, 48],
        "train_timepoints": [0, 1, 4, 16, 27],
        "holdout_timepoints": [36, 48],
        "source_time_for_extrapolation": 27,
    },
    "protocol": {
        "protocol_id": "paper-v1-freeze-internal-population-extrapolation",
        "primary_metric": "population_mean_across_gene_pearson_r",
        "seeds": [42],
    },
    "locked_claims": {
        "ode_composite_mean": 0.4833,
        "gru_composite_mean": -0.1444,
    },
}


def run_metadata(
    seed=42,
    tag="primary_internal_freeze",
    *,
    model="ode",
    source_hashes=None,
    checkpoint_sha=None,
) -> dict:
    digest = "a" * 64
    hashes = source_hashes or {}
    return {
        "dataset": {
            "dataset_id": "internal_lps_primary",
            "source_accession": "GSE151658",
            "n_cells": 38439,
            "n_genes": 2991,
            "timepoints": [0, 1, 4, 16, 27, 36, 48],
            "gene_manifest_sha256": hashes.get(
                "dataset.gene_manifest_sha256", digest
            ),
            "metadata_sha256": hashes.get("dataset.metadata_sha256", digest),
            "expression_matrix_sha256": hashes.get(
                "dataset.expression_matrix_sha256", digest
            ),
        },
        "protocol": {
            "protocol_id": "paper-v1-freeze-internal-population-extrapolation",
            "train_timepoints": [0, 1, 4, 16, 27],
            "holdout_timepoints": [36, 48],
            "source_time_for_extrapolation": 27,
            "primary_metric": "population_mean_across_gene_pearson_r",
            "seed": seed,
            "result_tag": tag,
            "checkpoint_selection": {"holdout_used_for_selection": False},
        },
        "source_hashes": {
            "training_script_sha256": hashes.get(
                "source_hashes.training_script_sha256", digest
            )
        },
        "checkpoint": {
            "path": (
                f"artifacts/models/ode_benchmark/"
                f"best_{model}_{tag}_seed{seed}.pt"
            ),
            "sha256": checkpoint_sha or digest,
        },
    }


def validation_payload(
    targets=(36, 48),
    from_time=27,
    corr=0.5,
    metadata=None,
) -> dict:
    payload = {
        "time_extrapolation": {
            "mean_correlation": corr,
            "pairs": {str(t): {"from_time": from_time, "corr": corr} for t in targets},
        }
    }
    if metadata is not None:
        payload["run_metadata"] = metadata
    return payload


class FilenameTest(unittest.TestCase):
    def test_legacy_filename_without_tag(self) -> None:
        self.assertEqual(
            audit.result_filename("ode", 42, None),
            "validation_results_ode_seed42.json",
        )

    def test_tagged_filename(self) -> None:
        self.assertEqual(
            audit.result_filename("gru", 123, "primary_internal_freeze"),
            "validation_results_gru_primary_internal_freeze_seed123.json",
        )


class ResolveResultTagTest(unittest.TestCase):
    def test_explicit_tag_accepted(self) -> None:
        self.assertEqual(
            audit.resolve_result_tag("primary_internal_freeze.1-2_3"),
            "primary_internal_freeze.1-2_3",
        )

    def test_none_when_unset(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(audit.resolve_result_tag())
        self.assertIsNone(audit.resolve_result_tag(""))
        self.assertIsNone(audit.resolve_result_tag("   "))

    def test_env_var_fallback(self) -> None:
        with mock.patch.dict(os.environ, {"LEWORLD_RESULT_TAG": "env_tag"}):
            self.assertEqual(audit.resolve_result_tag(), "env_tag")
            self.assertEqual(audit.resolve_result_tag(None), "env_tag")

    def test_explicit_arg_overrides_env(self) -> None:
        with mock.patch.dict(os.environ, {"LEWORLD_RESULT_TAG": "env_tag"}):
            self.assertEqual(audit.resolve_result_tag("cli_tag"), "cli_tag")

    def test_illegal_tags_rejected(self) -> None:
        for bad in [
            "../evil",
            "..",
            ".",
            "a/b",
            "a\\b",
            "a b",
            "tag;rm",
            "$(whoami)",
        ]:
            with self.subTest(tag=bad), mock.patch.dict(
                os.environ, {"LEWORLD_RESULT_TAG": bad}
            ):
                with self.assertRaises(ValueError):
                    audit.resolve_result_tag()


class AuditFreezeTest(unittest.TestCase):
    """Run the audit against a temp dir; repo artifacts are never touched."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="leworld_audit_test_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.result_dir = self.root / "artifacts" / "results" / "ode_benchmark"
        self.result_dir.mkdir(parents=True)
        for relative_path, content in {
            "data/leworld_input/gene_names.txt": b"GeneA\nGeneB\n",
            "data/leworld_input/metadata.csv": b"cell,timepoint_numeric\nc1,0\n",
            "data/leworld_input/expression_matrix.npy": b"frozen-matrix-bytes",
            "ode_benchmark/train_ode_benchmark.py": b"# frozen training script\n",
        }.items():
            path = self.root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.source_hashes, source_failures = audit.current_source_hashes(self.root)
        self.assertEqual(source_failures, [])
        for model in ("ode", "gru"):
            name = audit.result_filename(model, 42, None)
            (self.result_dir / name).write_text(
                json.dumps(validation_payload()), encoding="utf-8"
            )

    def audit(self, tag=None, manifest=MANIFEST) -> list[str]:
        return audit.audit_freeze(
            manifest, self.result_dir, tag, root=self.root
        )

    def tagged_metadata(
        self,
        model: str,
        *,
        seed: int = 42,
        tag: str = "primary_internal_freeze",
    ) -> dict:
        checkpoint_relative = Path(
            f"artifacts/models/ode_benchmark/best_{model}_{tag}_seed{seed}.pt"
        )
        checkpoint = self.root / checkpoint_relative
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"{model}:{tag}:{seed}".encode())
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        return run_metadata(
            seed=seed,
            tag=tag,
            model=model,
            source_hashes=self.source_hashes,
            checkpoint_sha=checkpoint_sha,
        )

    def write_tagged_pair(
        self,
        tag: str = "primary_internal_freeze",
        metadata_mutator=None,
    ) -> None:
        for model in ("ode", "gru"):
            metadata = self.tagged_metadata(model, tag=tag)
            if metadata_mutator is not None:
                metadata_mutator(metadata, model)
            path = self.result_dir / audit.result_filename(model, 42, tag)
            path.write_text(
                json.dumps(validation_payload(metadata=metadata)),
                encoding="utf-8",
            )

    def test_legacy_files_pass_with_expected_targets(self) -> None:
        self.assertEqual(self.audit(tag=None), [])

    def test_wrong_target_fails(self) -> None:
        for model in ("ode", "gru"):
            name = audit.result_filename(model, 42, None)
            (self.result_dir / name).write_text(
                json.dumps(validation_payload(targets=(36, 49))), encoding="utf-8"
            )
        failures = self.audit(tag=None)
        self.assertEqual(len(failures), 2)
        for failure in failures:
            self.assertIn("targets [36, 49] != expected [36, 48]", failure)

    def test_wrong_from_time_fails(self) -> None:
        for model in ("ode", "gru"):
            name = audit.result_filename(model, 42, None)
            (self.result_dir / name).write_text(
                json.dumps(validation_payload(from_time=3)), encoding="utf-8"
            )
        failures = self.audit(tag=None)
        self.assertEqual(len(failures), 4)
        for failure in failures:
            self.assertIn("comes from 3h, expected 27h", failure)

    def test_tagged_files_pass_and_legacy_missing_when_tag_set(self) -> None:
        tag = "primary_internal_freeze"
        self.write_tagged_pair(tag)
        self.assertEqual(self.audit(tag=tag), [])

        # Without the tagged files the same tag reports missing entries.
        for model in ("ode", "gru"):
            (self.result_dir / audit.result_filename(model, 42, tag)).unlink()
        failures = self.audit(tag=tag)
        self.assertEqual(len(failures), 2)
        for failure in failures:
            self.assertIn("missing:", failure)
            self.assertIn(tag, failure)

    def test_tagged_file_requires_run_metadata(self) -> None:
        tag = "primary_internal_freeze"
        for model in ("ode", "gru"):
            path = self.result_dir / audit.result_filename(model, 42, tag)
            path.write_text(json.dumps(validation_payload()), encoding="utf-8")
        failures = self.audit(tag=tag)
        self.assertEqual(len(failures), 2)
        self.assertTrue(all("missing run_metadata" in failure for failure in failures))

    def test_tagged_metadata_must_match_seed_tag_and_no_holdout_selection(self) -> None:
        tag = "primary_internal_freeze"
        def corrupt(metadata, _model):
            metadata["protocol"]["seed"] = 123
            metadata["protocol"]["result_tag"] = "wrong"
            metadata["protocol"]["checkpoint_selection"][
                "holdout_used_for_selection"
            ] = True

        self.write_tagged_pair(tag, corrupt)
        failures = self.audit(tag=tag)
        self.assertEqual(len(failures), 6)
        joined = "\n".join(failures)
        self.assertIn("protocol.seed=123, expected 42", joined)
        self.assertIn("protocol.result_tag='wrong'", joined)
        self.assertIn("holdout_used_for_selection must be false", joined)

    def test_tagged_metadata_requires_real_sha256_fields(self) -> None:
        tag = "primary_internal_freeze"
        def corrupt(metadata, _model):
            metadata["dataset"]["gene_manifest_sha256"] = "short"
            metadata["dataset"]["metadata_sha256"] = None
            metadata["source_hashes"]["training_script_sha256"] = ""
            metadata["checkpoint"]["sha256"] = "z" * 64

        self.write_tagged_pair(tag, corrupt)
        failures = self.audit(tag=tag)
        self.assertEqual(len(failures), 8)
        self.assertTrue(all("SHA-256 digest" in failure for failure in failures))

    def test_tagged_checkpoint_path_must_be_exact_per_seed_path(self) -> None:
        tag = "primary_internal_freeze"

        def corrupt(metadata, _model):
            metadata["checkpoint"]["path"] = (
                "artifacts/models/ode_benchmark/best_shared.pt"
            )

        self.write_tagged_pair(tag, corrupt)
        failures = self.audit(tag=tag)
        self.assertEqual(len(failures), 2)
        self.assertTrue(all("checkpoint.path=" in failure for failure in failures))

    def test_tagged_checkpoint_file_must_exist(self) -> None:
        tag = "primary_internal_freeze"
        self.write_tagged_pair(tag)
        for model in ("ode", "gru"):
            (self.root / (
                f"artifacts/models/ode_benchmark/"
                f"best_{model}_{tag}_seed42.pt"
            )).unlink()
        failures = self.audit(tag=tag)
        self.assertEqual(len(failures), 2)
        self.assertTrue(all("checkpoint file is missing" in f for f in failures))

    def test_tagged_checkpoint_sha_must_match_file_bytes(self) -> None:
        tag = "primary_internal_freeze"
        self.write_tagged_pair(tag)
        for model in ("ode", "gru"):
            (self.root / (
                f"artifacts/models/ode_benchmark/"
                f"best_{model}_{tag}_seed42.pt"
            )).write_bytes(b"tampered")
        failures = self.audit(tag=tag)
        self.assertEqual(len(failures), 2)
        self.assertTrue(all("checkpoint.sha256 does not match" in f for f in failures))

    def test_tagged_source_hashes_must_match_current_files(self) -> None:
        tag = "primary_internal_freeze"
        self.write_tagged_pair(tag)
        (self.root / "data/leworld_input/gene_names.txt").write_bytes(b"changed\n")
        failures = self.audit(tag=tag)
        self.assertEqual(len(failures), 2)
        self.assertTrue(
            all("gene_manifest_sha256 does not match current" in f for f in failures)
        )

    def test_nonfinite_correlations_fail_closed(self) -> None:
        for model in ("ode", "gru"):
            name = audit.result_filename(model, 42, None)
            (self.result_dir / name).write_text(
                json.dumps(validation_payload(corr=float("nan"))), encoding="utf-8"
            )
        failures = self.audit(tag=None)
        self.assertEqual(len(failures), 6)
        self.assertTrue(all("not finite" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
