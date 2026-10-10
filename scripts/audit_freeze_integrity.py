#!/usr/bin/env python3
"""Read-only audit for the paper-v1-freeze provenance contract.

This script never loads arrays into memory, trains models, downloads data, or
rewrites artifacts. For tagged production runs it hashes the frozen input
files, training script, and per-seed checkpoints, then checks those bytes
against each validation record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULT_TAG_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
CURRENT_SOURCE_PATHS = {
    "dataset.gene_manifest_sha256": Path("data/leworld_input/gene_names.txt"),
    "dataset.metadata_sha256": Path("data/leworld_input/metadata.csv"),
    "dataset.expression_matrix_sha256": Path("data/leworld_input/expression_matrix.npy"),
    "source_hashes.training_script_sha256": Path(
        "ode_benchmark/train_ode_benchmark.py"
    ),
}


def resolve_result_tag(explicit: str | None = None) -> str | None:
    """Return the effective result tag from --result-tag or LEWORLD_RESULT_TAG.

    Returns None (legacy untagged filenames) when neither source is set.
    Rejects anything outside [A-Za-z0-9._-] plus '.'/'..' so a tag can never
    traverse out of the artifact directory.
    """
    raw = explicit if explicit is not None else os.environ.get("LEWORLD_RESULT_TAG", "")
    tag = raw.strip()
    if not tag:
        return None
    if not RESULT_TAG_PATTERN.fullmatch(tag) or tag in {".", ".."}:
        raise ValueError(
            f"invalid result tag {tag!r}: only letters, digits, '.', '_' and '-' "
            "are allowed; path separators and '..' are rejected"
        )
    return tag


def result_filename(model: str, seed: int, tag: str | None) -> str:
    if tag:
        return f"validation_results_{model}_{tag}_seed{seed}.json"
    return f"validation_results_{model}_seed{seed}.json"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_source_hashes(root: Path) -> tuple[dict[str, str], list[str]]:
    """Hash each current frozen source once and report missing inputs."""
    observed: dict[str, str] = {}
    failures: list[str] = []
    for label, relative_path in CURRENT_SOURCE_PATHS.items():
        path = root / relative_path
        if not path.is_file():
            failures.append(f"missing current source: {display_path(path)}")
            continue
        observed[label] = file_sha256(path)
    return observed, failures


def inspect_validation(path: Path) -> dict:
    payload = read_json(path)
    ex = payload.get("time_extrapolation", {})
    pairs = ex.get("pairs", {})
    return {
        "path": display_path(path),
        "mean": ex.get("mean_correlation"),
        "pairs": {
            str(k): {
                "from_time": v.get("from_time"),
                "corr": v.get("corr"),
            }
            for k, v in pairs.items()
            if isinstance(v, dict)
        },
        "run_metadata": payload.get("run_metadata"),
    }


def _sha256_is_present(value: object) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-fA-F]{64}", value))


def _metadata_failures(
    *,
    filename: str,
    metadata: object,
    manifest: dict,
    model: str,
    seed: int,
    tag: str,
    root: Path,
    observed_sources: dict[str, str],
) -> list[str]:
    """Validate the tagged production provenance contract."""
    prefix = f"{filename}:"
    if not isinstance(metadata, dict):
        return [f"{prefix} missing run_metadata object"]

    failures: list[str] = []
    expected_dataset = manifest["dataset"]
    dataset = metadata.get("dataset")
    protocol = metadata.get("protocol")
    sources = metadata.get("source_hashes")
    checkpoint = metadata.get("checkpoint")
    if not isinstance(dataset, dict):
        failures.append(f"{prefix} run_metadata.dataset is missing or invalid")
        dataset = {}
    if not isinstance(protocol, dict):
        failures.append(f"{prefix} run_metadata.protocol is missing or invalid")
        protocol = {}
    if not isinstance(sources, dict):
        failures.append(f"{prefix} run_metadata.source_hashes is missing or invalid")
        sources = {}
    if not isinstance(checkpoint, dict):
        failures.append(f"{prefix} run_metadata.checkpoint is missing or invalid")
        checkpoint = {}

    expected_fields = {
        "dataset_id": expected_dataset["dataset_id"],
        "source_accession": expected_dataset["source_accession"],
        "n_cells": expected_dataset.get("n_cells"),
        "n_genes": expected_dataset["n_genes"],
        "timepoints": expected_dataset.get("timepoints"),
    }
    for key, expected_value in expected_fields.items():
        if expected_value is not None and dataset.get(key) != expected_value:
            failures.append(
                f"{prefix} dataset.{key}={dataset.get(key)!r}, expected {expected_value!r}"
            )

    protocol_fields = {
        "protocol_id": manifest["protocol"].get("protocol_id"),
        "train_timepoints": expected_dataset["train_timepoints"],
        "holdout_timepoints": expected_dataset["holdout_timepoints"],
        "source_time_for_extrapolation": expected_dataset["source_time_for_extrapolation"],
        "primary_metric": manifest["protocol"].get("primary_metric"),
        "seed": seed,
        "result_tag": tag,
    }
    for key, expected_value in protocol_fields.items():
        if expected_value is not None and protocol.get(key) != expected_value:
            failures.append(
                f"{prefix} protocol.{key}={protocol.get(key)!r}, expected {expected_value!r}"
            )

    selection = protocol.get("checkpoint_selection")
    if not isinstance(selection, dict):
        failures.append(f"{prefix} protocol.checkpoint_selection is missing or invalid")
    elif selection.get("holdout_used_for_selection") is not False:
        failures.append(
            f"{prefix} checkpoint_selection.holdout_used_for_selection must be false"
        )

    required_hashes = {
        "dataset.gene_manifest_sha256": dataset.get("gene_manifest_sha256"),
        "dataset.metadata_sha256": dataset.get("metadata_sha256"),
        "dataset.expression_matrix_sha256": dataset.get("expression_matrix_sha256"),
        "source_hashes.training_script_sha256": sources.get("training_script_sha256"),
        "checkpoint.sha256": checkpoint.get("sha256"),
    }
    for label, value in required_hashes.items():
        if not _sha256_is_present(value):
            failures.append(f"{prefix} {label} is not a SHA-256 digest")

    for label, observed_digest in observed_sources.items():
        section, field = label.split(".", 1)
        recorded_digest = dataset.get(field) if section == "dataset" else sources.get(field)
        if _sha256_is_present(recorded_digest) and recorded_digest.lower() != observed_digest:
            failures.append(
                f"{prefix} {label} does not match current "
                f"{CURRENT_SOURCE_PATHS[label].as_posix()}"
            )

    expected_checkpoint_relative = Path(
        "artifacts/models/ode_benchmark"
    ) / f"best_{model}_{tag}_seed{seed}.pt"
    expected_checkpoint_text = expected_checkpoint_relative.as_posix()
    recorded_checkpoint_path = checkpoint.get("path")
    if recorded_checkpoint_path != expected_checkpoint_text:
        failures.append(
            f"{prefix} checkpoint.path={recorded_checkpoint_path!r}, "
            f"expected {expected_checkpoint_text!r}"
        )

    expected_checkpoint = root / expected_checkpoint_relative
    if not expected_checkpoint.is_file():
        failures.append(
            f"{prefix} checkpoint file is missing: {display_path(expected_checkpoint)}"
        )
    else:
        recorded_checkpoint_sha = checkpoint.get("sha256")
        if _sha256_is_present(recorded_checkpoint_sha):
            observed_checkpoint_sha = file_sha256(expected_checkpoint)
            if recorded_checkpoint_sha.lower() != observed_checkpoint_sha:
                failures.append(
                    f"{prefix} checkpoint.sha256 does not match "
                    f"{display_path(expected_checkpoint)}"
                )
    return failures


def audit_freeze(
    manifest: dict,
    result_dir: Path,
    tag: str | None,
    *,
    root: Path = ROOT,
) -> list[str]:
    """Check per-seed validation JSONs against the manifest; return failures."""
    expected = manifest["dataset"]
    failures: list[str] = []
    observed_sources: dict[str, str] = {}
    if tag:
        observed_sources, source_failures = current_source_hashes(root)
        failures.extend(source_failures)
    for model in ("ode", "gru"):
        for seed in manifest["protocol"]["seeds"]:
            path = result_dir / result_filename(model, seed, tag)
            if not path.exists():
                failures.append(f"missing: {display_path(path)}")
                continue
            info = inspect_validation(path)
            print(json.dumps(info, ensure_ascii=False))
            observed_targets = sorted(int(k) for k in info["pairs"])
            expected_targets = sorted(expected["holdout_timepoints"])
            if observed_targets != expected_targets:
                failures.append(
                    f"{path.name}: targets {observed_targets} != expected {expected_targets}"
                )
            for target, row in info["pairs"].items():
                if row["from_time"] != expected["source_time_for_extrapolation"]:
                    failures.append(
                        f"{path.name}: {target}h comes from {row['from_time']}h, "
                        f"expected {expected['source_time_for_extrapolation']}h"
                    )
                corr = row.get("corr")
                if not isinstance(corr, (int, float)) or not math.isfinite(corr):
                    failures.append(f"{path.name}: {target}h correlation is not finite")
            mean = info.get("mean")
            if not isinstance(mean, (int, float)) or not math.isfinite(mean):
                failures.append(f"{path.name}: mean correlation is not finite")
            if tag:
                failures.extend(
                    _metadata_failures(
                        filename=path.name,
                        metadata=info.get("run_metadata"),
                        manifest=manifest,
                        model=model,
                        seed=seed,
                        tag=tag,
                        root=root,
                        observed_sources=observed_sources,
                    )
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts" / "reports" / "primary_internal_freeze_manifest.json",
    )
    parser.add_argument(
        "--result-tag",
        default=None,
        help="Tagged result namespace; defaults to $LEWORLD_RESULT_TAG, "
        "else legacy untagged filenames.",
    )
    args = parser.parse_args()

    try:
        tag = resolve_result_tag(args.result_tag)
    except ValueError as exc:
        print("STATUS: CONFLICT")
        print(" -", exc)
        return 2

    manifest = read_json(args.manifest)
    expected = manifest["dataset"]
    locked = manifest["locked_claims"]
    result_dir = ROOT / "artifacts" / "results" / "ode_benchmark"

    print("manifest:", display_path(args.manifest))
    if tag:
        print("result tag:", tag)
    print("dataset:", expected["dataset_id"], "accession:", expected["source_accession"])
    print("expected holdout:", expected["holdout_timepoints"])
    print("locked ODE composite:", locked["ode_composite_mean"])
    print("locked GRU composite:", locked["gru_composite_mean"])

    failures = audit_freeze(manifest, result_dir, tag)

    if failures:
        print("STATUS: CONFLICT")
        for failure in failures:
            print(" -", failure)
        return 2
    if tag:
        print("STATUS: PASS (tagged provenance and result schema; no model rerun)")
    else:
        print("STATUS: PASS (legacy result schema only; no tagged provenance check)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
