#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import os
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data" / "leworld_input"
EXPECTED_SHAPE = (38439, 2991)
EXPECTED_TIME_COUNTS = {
    0: 4867,
    1: 5216,
    4: 5345,
    16: 4498,
    27: 4000,
    36: 6093,
    48: 8420,
}
EXPECTED_GENE_MANIFEST_SHA256 = (
    "741e496abe9e14fea59c6070b3b7da86b62fa59c2b982fa66e2ab547bcfbe5b2"
)
EXPECTED_METADATA_SHA256 = (
    "80223b5d11d743425550e06ce2413c6dc79a735b073b6ef0a82fcc5ed4ffe3c6"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_npy_header(path: Path) -> dict:
    """Read a NumPy .npy header without importing NumPy."""
    with path.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise ValueError(f"{path}: invalid NumPy magic")
        major, minor = handle.read(2)
        if major == 1:
            header_length = struct.unpack("<H", handle.read(2))[0]
            encoding = "latin1"
        elif major in {2, 3}:
            header_length = struct.unpack("<I", handle.read(4))[0]
            encoding = "utf-8" if major == 3 else "latin1"
        else:
            raise ValueError(f"{path}: unsupported NumPy format {major}.{minor}")
        header = ast.literal_eval(
            handle.read(header_length).decode(encoding).strip()
        )
    if not isinstance(header, dict):
        raise ValueError(f"{path}: invalid NumPy header")
    return header


def run_script(
    path: Path,
    extra_args: list[str] | None = None,
    seed: int | None = None,
    result_tag: str | None = None,
) -> None:
    cmd = [sys.executable, str(path)]
    if extra_args:
        cmd.extend(extra_args)
    env = os.environ.copy()
    if seed is not None:
        env["LEWORLD_SEED"] = str(seed)
    if result_tag:
        env["LEWORLD_RESULT_TAG"] = result_tag
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def resolve_data_dir(environ: dict[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    raw = env.get("LEWORLD_DATA_DIR", "").strip()
    if not raw:
        return DEFAULT_DATA_DIR
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def run_check(data_dir: Path | None = None) -> None:
    data_dir = resolve_data_dir() if data_dir is None else data_dir
    required = [
        data_dir / "expression_matrix.npy",
        data_dir / "metadata.csv",
        data_dir / "gene_names.txt",
    ]
    missing = [str(p) for p in required if not p.exists()]
    print("Bundle root:", ROOT)
    print("Data directory:", data_dir)
    if missing:
        print("Missing required data files:")
        for path in missing:
            print(" -", path)
        raise SystemExit(2)

    failures = []
    matrix_path = data_dir / "expression_matrix.npy"
    try:
        header = read_npy_header(matrix_path)
        shape = tuple(header.get("shape", ()))
        if shape != EXPECTED_SHAPE:
            failures.append(
                f"expression matrix shape {shape} != expected {EXPECTED_SHAPE}"
            )
        if header.get("fortran_order") is not False:
            failures.append("expression matrix must be C-contiguous")
        if header.get("descr") not in {"<f4", ">f4", "|f4", "=f4"}:
            failures.append(
                f"expression matrix dtype {header.get('descr')!r} is not float32"
            )
    except (OSError, ValueError, SyntaxError, struct.error) as exc:
        failures.append(f"cannot validate expression_matrix.npy: {exc}")

    gene_path = data_dir / "gene_names.txt"
    gene_names = [
        line.strip()
        for line in gene_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(gene_names) != EXPECTED_SHAPE[1]:
        failures.append(
            f"gene manifest contains {len(gene_names)} genes, expected {EXPECTED_SHAPE[1]}"
        )
    if len({name.upper() for name in gene_names}) != len(gene_names):
        failures.append("gene manifest contains case-insensitive duplicates")
    observed_gene_hash = file_sha256(gene_path)
    if observed_gene_hash != EXPECTED_GENE_MANIFEST_SHA256:
        failures.append(
            "gene manifest SHA-256 does not match the frozen 2,991-gene panel"
        )

    metadata_path = data_dir / "metadata.csv"
    time_counts: Counter[int] = Counter()
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "timepoint_numeric" not in (reader.fieldnames or []):
            failures.append("metadata.csv lacks timepoint_numeric")
        else:
            for row in reader:
                try:
                    time_counts[int(float(row["timepoint_numeric"]))] += 1
                except (TypeError, ValueError):
                    failures.append("metadata.csv contains an invalid timepoint")
                    break
    if dict(sorted(time_counts.items())) != EXPECTED_TIME_COUNTS:
        failures.append(
            "metadata time counts "
            f"{dict(sorted(time_counts.items()))} != expected {EXPECTED_TIME_COUNTS}"
        )
    observed_metadata_hash = file_sha256(metadata_path)
    if observed_metadata_hash != EXPECTED_METADATA_SHA256:
        failures.append(
            "metadata SHA-256 does not match the frozen 38,439-cell manifest"
        )

    summary_path = data_dir / "data_summary.json"
    if summary_path.exists():
        import json

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_summary = {
            "n_cells": EXPECTED_SHAPE[0],
            "n_genes": EXPECTED_SHAPE[1],
            "timepoints": sorted(EXPECTED_TIME_COUNTS),
        }
        for key, expected in expected_summary.items():
            if summary.get(key) != expected:
                failures.append(
                    f"data_summary.{key}={summary.get(key)!r}, expected {expected!r}"
                )

    if failures:
        print("Data contract failed:")
        for failure in failures:
            print(" -", failure)
        raise SystemExit(2)
    print(
        "Data check passed:",
        f"shape={EXPECTED_SHAPE}",
        f"timepoints={sorted(EXPECTED_TIME_COUNTS)}",
        f"gene_sha256={observed_gene_hash}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run publication bundle experiments.")
    parser.add_argument(
        "target",
        choices=["check", "audit_freeze", "preprocess", "ode", "baselines", "stats", "stats_extrapolation", "scvelo"],
        help="Experiment target to run.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Reserved for future seed override.")
    parser.add_argument(
        "--result-tag",
        default=None,
        help="Tagged result namespace; forwarded to audit_freeze as --result-tag "
        "and to ode runs via LEWORLD_RESULT_TAG.",
    )
    args = parser.parse_args()

    if args.target == "check":
        run_check()
        return
    if args.target == "audit_freeze":
        cmd = [sys.executable, str(ROOT / "scripts" / "audit_freeze_integrity.py")]
        if args.result_tag:
            cmd.extend(["--result-tag", args.result_tag])
        subprocess.run(cmd, cwd=ROOT, check=True)
        return
    mapping = {
        "preprocess": ROOT / "preprocess" / "preprocess_full_v7_1.4g_integrated_full.py",
        "ode": ROOT / "ode_benchmark" / "train_ode_benchmark.py",
        "baselines": ROOT / "baselines" / "baseline_comparison.py",
        "stats": ROOT / "baselines" / "statistical_significance.py",
        "stats_extrapolation": ROOT / "baselines" / "extrapolation_significance.py",
        "scvelo": ROOT / "baselines" / "run_scvelo_baseline.py",
    }
    run_script(
        mapping[args.target],
        seed=args.seed,
        result_tag=args.result_tag if args.target == "ode" else None,
    )


if __name__ == "__main__":
    main()
