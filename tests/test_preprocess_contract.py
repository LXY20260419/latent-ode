#!/usr/bin/env python3
"""Safety-contract tests for the isolated GSE151658 preprocessing job."""
from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


PREPROCESS_DIR = Path(__file__).resolve().parent.parent / "preprocess"
sys.path.insert(0, str(PREPROCESS_DIR))

import preprocess_contract as contract  # noqa: E402


class EnvironmentContractTest(unittest.TestCase):
    def test_default_paths_are_isolated(self) -> None:
        root = Path("/example-repository")
        rds, output, genes = contract.resolve_preprocess_paths(root, {})
        self.assertEqual(
            rds,
            root / "data/raw/GSE151658_integrated.0h.1h_4h_16h_27h_36h_48h.rds.gz",
        )
        self.assertEqual(output, root / "data/integrated_processed/leworld_input")
        self.assertEqual(genes, root / "data/leworld_input/gene_names.txt")

    def test_environment_overrides_are_resolved_from_project_root(self) -> None:
        root = Path("/example-repository")
        rds, output, genes = contract.resolve_preprocess_paths(
            root,
            {
                "LEWORLD_RDS_FILE": "source/input.rds.gz",
                "LEWORLD_PREPROCESS_OUTPUT_DIR": "candidate/leworld_input",
                "LEWORLD_FROZEN_GENE_MANIFEST": "freeze/genes.txt",
            },
        )
        self.assertEqual(rds, root / "source/input.rds.gz")
        self.assertEqual(output, root / "candidate/leworld_input")
        self.assertEqual(genes, root / "freeze/genes.txt")

    def test_frozen_output_directory_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "refusing to preprocess"):
            contract.resolve_preprocess_paths(
                Path("/example-repository"),
                {"LEWORLD_PREPROCESS_OUTPUT_DIR": "data/leworld_input"},
            )

    def test_boolean_environment_values_are_strict(self) -> None:
        self.assertTrue(contract.parse_env_bool("FLAG", False, {"FLAG": "yes"}))
        self.assertFalse(contract.parse_env_bool("FLAG", True, {"FLAG": "0"}))
        with self.assertRaisesRegex(ValueError, "FLAG must be"):
            contract.parse_env_bool("FLAG", True, {"FLAG": "sometimes"})


class HashAndManifestContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="preprocess_contract_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_hash_verification_accepts_only_expected_bytes(self) -> None:
        path = self.root / "source.bin"
        path.write_bytes(b"known bytes")
        expected = hashlib.sha256(b"known bytes").hexdigest()
        self.assertEqual(
            contract.verify_file_sha256(path, expected, "fixture"), expected
        )
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            contract.verify_file_sha256(path, "0" * 64, "fixture")

    def test_manifest_requires_count_hash_and_unique_casefolded_names(self) -> None:
        path = self.root / "genes.txt"
        path.write_text("GeneA\nGeneB\n", encoding="utf-8")
        digest = contract.file_sha256(path)
        genes, observed = contract.load_frozen_gene_manifest(
            path, expected_sha256=digest, expected_count=2
        )
        self.assertEqual(genes, ["GeneA", "GeneB"])
        self.assertEqual(observed, digest)

        path.write_text("GeneA\ngenea\n", encoding="utf-8")
        duplicate_digest = contract.file_sha256(path)
        with self.assertRaisesRegex(ValueError, "case-insensitive duplicates"):
            contract.load_frozen_gene_manifest(
                path, expected_sha256=duplicate_digest, expected_count=2
            )


class FrozenGeneResolutionTest(unittest.TestCase):
    def test_exact_and_unique_casefolded_matches_preserve_frozen_order(self) -> None:
        indices, remapped = contract.resolve_frozen_gene_indices(
            ["third", "GENEA", "GeneB"], ["GeneB", "GeneA"]
        )
        self.assertEqual(indices, [2, 1])
        self.assertEqual(remapped, {"GeneA": "GENEA"})

    def test_missing_and_ambiguous_matches_fail_together(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "missing frozen genes.*ambiguous frozen genes"
        ):
            contract.resolve_frozen_gene_indices(
                ["GENEA", "genea", "GeneB"], ["GeneA", "Missing"]
            )

    def test_exact_duplicate_source_names_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ambiguous frozen genes"):
            contract.resolve_frozen_gene_indices(
                ["GeneA", "GeneA"], ["GeneA"]
            )


if __name__ == "__main__":
    unittest.main()
