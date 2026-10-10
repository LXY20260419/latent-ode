#!/usr/bin/env python3
"""Unit tests for tag handling in ode_benchmark/train_ode_benchmark.py (pure stdlib).

train_ode_benchmark.py imports torch/numpy/scipy at module scope, so it cannot
be imported in a bare environment.  These tests therefore never import it.
Instead they:

* extract the small pure helpers (``RESULT_TAG_PATTERN``, ``validate_result_tag``,
  ``tagged_filename``) from the source via ``ast`` and exec them in an isolated
  namespace that provides nothing but ``re``;
* statically verify the sidecar call sites and the fail-closed ordering on the
  module AST.

The repo artifacts/ tree is never read or written by these tests.
"""
from __future__ import annotations

import ast
import random
import re
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "ode_benchmark" / "train_ode_benchmark.py"
PURE_HELPER_NAMES = (
    "RESULT_TAG_PATTERN",
    "validate_result_tag",
    "tagged_filename",
    "checkpoint_filename",
    "split_internal_validation",
)


def _parse() -> ast.Module:
    return ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))


def _top_level_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.FunctionDef):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {t.id for t in node.targets if isinstance(t, ast.Name)}
    return set()


def load_pure_helpers() -> dict:
    """Compile and exec only the module-level pure tag helpers.

    Any accidental dependency on torch/numpy inside the extracted helpers
    raises NameError as soon as they run, so the extraction stays honest.
    """
    tree = _parse()
    body = [n for n in tree.body if _top_level_names(n) & set(PURE_HELPER_NAMES)]
    found: set[str] = set()
    for node in body:
        found |= _top_level_names(node)
    missing = set(PURE_HELPER_NAMES) - found
    assert not missing, f"pure tag helpers missing from {SCRIPT}: {sorted(missing)}"
    namespace: dict = {
        "re": re,
        "random": random,
        "__name__": "train_ode_benchmark_pure_helpers",
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace


class TaggedFilenameTest(unittest.TestCase):
    """tagged_filename: legacy names when untagged, one consistent tag slot."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tagged_filename = staticmethod(load_pure_helpers()["tagged_filename"])

    def test_untagged_names_keep_legacy_filenames(self) -> None:
        cases = [
            (("gap_decay_comparison", "png", ""), "gap_decay_comparison.png"),
            (("Il6_trajectory", "png", ""), "Il6_trajectory.png"),
            (("diagnosis_results", "json", ""), "diagnosis_results.json"),
            (("diagnosis_results", "json", "", "seed42"), "diagnosis_results_seed42.json"),
            (("benchmark_report", "md", ""), "benchmark_report.md"),
            (("benchmark_report", "md", "", "seed7"), "benchmark_report_seed7.md"),
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                self.assertEqual(self.tagged_filename(*args), expected)

    def test_tagged_names_put_tag_after_stem_before_suffix_and_ext(self) -> None:
        tag = "primary_internal_freeze"
        cases = [
            (("gap_decay_comparison", "png", tag), f"gap_decay_comparison_{tag}.png"),
            (("Il6_trajectory", "png", tag), f"Il6_trajectory_{tag}.png"),
            (("diagnosis_results", "json", tag), f"diagnosis_results_{tag}.json"),
            (("diagnosis_results", "json", tag, "seed42"), f"diagnosis_results_{tag}_seed42.json"),
            (("benchmark_report", "md", tag), f"benchmark_report_{tag}.md"),
            (("benchmark_report", "md", tag, "seed42"), f"benchmark_report_{tag}_seed42.md"),
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                self.assertEqual(self.tagged_filename(*args), expected)

    def test_tag_with_full_allowed_charset_survives_unchanged(self) -> None:
        tag = "v1.2-3_4"
        self.assertEqual(
            self.tagged_filename("gap_decay_comparison", "png", tag),
            f"gap_decay_comparison_{tag}.png",
        )


class ValidateResultTagTest(unittest.TestCase):
    """validate_result_tag: same strictness as scripts/audit_freeze_integrity.py."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.validate_result_tag = staticmethod(load_pure_helpers()["validate_result_tag"])

    def test_empty_tag_is_allowed_and_returns_empty(self) -> None:
        self.assertEqual(self.validate_result_tag(""), "")

    def test_valid_tags_are_returned_unchanged(self) -> None:
        for tag in ["internal", "gse224518_3to5h", "v1.2-3_4", "A9"]:
            with self.subTest(tag=tag):
                self.assertEqual(self.validate_result_tag(tag), tag)

    def test_illegal_tags_are_rejected(self) -> None:
        for bad in ["../evil", "..", ".", "a/b", "a\\b", "a b", "tag;rm", "$(whoami)", "tag\n"]:
            with self.subTest(tag=bad):
                with self.assertRaises(ValueError):
                    self.validate_result_tag(bad)


class CheckpointFilenameTest(unittest.TestCase):
    """Tagged checkpoints are seed-isolated; legacy paths remain stable."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.checkpoint_filename = staticmethod(
            load_pure_helpers()["checkpoint_filename"]
        )

    def test_untagged_checkpoint_keeps_legacy_name(self) -> None:
        self.assertEqual(
            self.checkpoint_filename("ode", "", 2026),
            "best_ode.pt",
        )
        self.assertEqual(
            self.checkpoint_filename("gru", "", 123),
            "best_gru.pt",
        )

    def test_tagged_checkpoint_contains_tag_and_seed(self) -> None:
        self.assertEqual(
            self.checkpoint_filename(
                "ode_primary_internal_freeze", "primary_internal_freeze", 2026
            ),
            "best_ode_primary_internal_freeze_seed2026.pt",
        )


class InternalValidationSplitTest(unittest.TestCase):
    """The checkpoint split is deterministic, stratified, and train-time only."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.split = staticmethod(load_pure_helpers()["split_internal_validation"])

    def test_split_is_deterministic_disjoint_and_exhaustive(self) -> None:
        indices = list(range(12))
        timepoints = [0] * 4 + [1] * 4 + [4] * 4 + [36, 48]
        first = self.split(indices, timepoints, 0.25, 42)
        second = self.split(indices, timepoints, 0.25, 42)
        self.assertEqual(first, second)
        train, validation = first
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(train) | set(validation), set(indices))
        self.assertTrue((set(train) | set(validation)).isdisjoint({12, 13}))

    def test_each_represented_time_retains_training_cells(self) -> None:
        indices = list(range(9))
        timepoints = [0] * 3 + [1] * 3 + [4] * 3
        train, validation = self.split(indices, timepoints, 0.9, 7)
        self.assertEqual({timepoints[i] for i in train}, {0, 1, 4})
        self.assertEqual({timepoints[i] for i in validation}, {0, 1, 4})

    def test_singleton_timepoint_stays_in_training(self) -> None:
        train, validation = self.split([0, 1, 2], [0, 1, 1], 0.5, 7)
        self.assertIn(0, train)
        self.assertNotIn(0, validation)

    def test_invalid_fraction_is_rejected(self) -> None:
        for fraction in [0, 1, -0.1, 1.1]:
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                self.split([0, 1], [0, 0], fraction, 42)


class SidecarCallSiteTest(unittest.TestCase):
    """Static AST checks: sidecar writes must route through tagged_filename."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = _parse()

    def _tagged_filename_calls(self) -> list[ast.Call]:
        return [
            n
            for n in ast.walk(self.tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "tagged_filename"
        ]

    def test_required_sidecars_are_built_with_tagged_filename(self) -> None:
        calls = self._tagged_filename_calls()
        self.assertTrue(calls, "no tagged_filename() calls found in the training script")
        stems: set[str] = set()
        for call in calls:
            self.assertGreaterEqual(len(call.args), 1, "stem must be positional")
            first = call.args[0]
            if isinstance(first, ast.Constant):
                stems.add(first.value)
            elif isinstance(first, ast.JoinedStr):
                stems.add("<dynamic>")  # f'{gene}_trajectory'
            else:
                self.fail(f"unexpected stem expression at line {call.lineno}")
        expected = {
            "gap_decay_comparison",
            "diagnosis_results",
            "benchmark_report",
            "*_trajectory",  # report/print glob reference
            "<dynamic>",  # per-gene trajectory figures
        }
        self.assertTrue(expected <= stems, f"missing stems: {expected - stems}")

    def test_every_tagged_filename_call_uses_validated_result_tag(self) -> None:
        for call in self._tagged_filename_calls():
            tag_arg = None
            if len(call.args) >= 3:
                tag_arg = call.args[2]
            else:
                for kw in call.keywords:
                    if kw.arg == "tag":
                        tag_arg = kw.value
            self.assertIsInstance(
                tag_arg, ast.Name, f"tag must be a plain variable at line {call.lineno}"
            )
            self.assertEqual(tag_arg.id, "RESULT_TAG")

    def test_result_tag_is_validated_at_import_before_directory_creation(self) -> None:
        body = self.tree.body
        idx_tag = idx_cfg = None
        for i, node in enumerate(body):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "RESULT_TAG" for t in node.targets
            ):
                idx_tag = i
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "cfg" for t in node.targets
            ):
                idx_cfg = i
        self.assertIsNotNone(idx_tag, "RESULT_TAG assignment not found at module level")
        self.assertIsNotNone(idx_cfg, "cfg = Config() not found at module level")
        tag_value = body[idx_tag].value
        self.assertIsInstance(tag_value, ast.Call, "RESULT_TAG must be validated inline")
        func = tag_value.func
        self.assertIsInstance(func, ast.Name)
        self.assertEqual(func.id, "validate_result_tag")
        self.assertLess(
            idx_tag,
            idx_cfg,
            "tag validation must precede cfg = Config() (which creates artifact dirs)",
        )

    def test_no_hardcoded_legacy_sidecar_filenames_remain(self) -> None:
        constants = [
            n.value
            for n in ast.walk(self.tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        ]
        for legacy in ("gap_decay_comparison.png", "diagnosis_results.json", "benchmark_report.md"):
            self.assertNotIn(legacy, constants)
        for node in ast.walk(self.tree):
            if isinstance(node, ast.JoinedStr):
                text = "".join(v.value for v in node.values if isinstance(v, ast.Constant))
                for legacy in ("diagnosis_results_seed", "benchmark_report_seed", "_trajectory.png"):
                    self.assertNotIn(legacy, text)


class ProtocolAndProvenanceStaticTest(unittest.TestCase):
    """Static checks for final-only holdout and provenance propagation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = _parse()

    def _function(self, name: str) -> ast.FunctionDef:
        matches = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        self.assertEqual(len(matches), 1, f"expected one function named {name}")
        return matches[0]

    def test_holdout_indices_are_not_passed_to_dataloader_or_trainer(self) -> None:
        main = self._function("main")
        for call in (node for node in ast.walk(main) if isinstance(node, ast.Call)):
            if isinstance(call.func, ast.Name) and call.func.id in {"DataLoader", "Trainer", "Trainer_ODE"}:
                rendered = ast.unparse(call)
                self.assertNotIn("holdout_idx", rendered)
        split_calls = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "split_internal_validation"
        ]
        self.assertEqual(len(split_calls), 1)
        self.assertEqual(ast.unparse(split_calls[0].args[0]), "eligible_train_idx")

    def test_metadata_declares_final_only_holdout_and_required_hashes(self) -> None:
        build = self._function("build_run_metadata")
        source = ast.unparse(build)
        for required in (
            "holdout_used_for_selection",
            "gene_manifest_sha256",
            "metadata_sha256",
            "expression_matrix_sha256",
            "training_script_sha256",
            "result_tag",
            "seed",
        ):
            self.assertIn(required, source)
        false_values = [
            node
            for node in ast.walk(build)
            if isinstance(node, ast.Constant) and node.value is False
        ]
        self.assertTrue(false_values, "holdout_used_for_selection must be literal False")

    def test_validation_payload_receives_metadata_and_checkpoint_hash(self) -> None:
        init = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "__init__"
            and any(arg.arg == "checkpoint" for arg in node.args.args)
        ]
        self.assertEqual(len(init), 1)
        init_source = ast.unparse(init[0])
        self.assertIn("self.run_metadata['checkpoint']", init_source)
        self.assertIn("file_sha256(checkpoint)", init_source)
        run_all = self._function("run_all")
        self.assertIn("self.results['run_metadata']", ast.unparse(run_all))

    def test_training_history_keeps_legacy_keys_and_adds_metadata(self) -> None:
        trainer = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Trainer"
        )
        train = next(
            node
            for node in trainer.body
            if isinstance(node, ast.FunctionDef) and node.name == "train"
        )
        source = ast.unparse(train)
        self.assertIn("{**self.history, 'run_metadata': self.run_metadata}", source)
        self.assertNotIn("'history': self.history", source)

    def test_checkpoint_metadata_and_load_are_weights_only_safe(self) -> None:
        build = self._function("build_run_metadata")
        build_source = ast.unparse(build)
        self.assertIn("str(torch.__version__)", build_source)
        main = self._function("main")
        load_calls = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "load"
        ]
        self.assertEqual(len(load_calls), 2)
        for call in load_calls:
            weights_only = next(
                (keyword.value for keyword in call.keywords if keyword.arg == "weights_only"),
                None,
            )
            self.assertIsInstance(weights_only, ast.Constant)
            self.assertIs(weights_only.value, True)


if __name__ == "__main__":
    unittest.main()
