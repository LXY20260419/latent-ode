#!/usr/bin/env python3
"""Pure, lightweight safety helpers for the GSE151658 preprocessing job."""
from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


EXPECTED_RDS_SHA256 = (
    "4f2847d9946dc06a31df956780dc26d3d76848ee8b7582888b5620871ef14c25"
)
EXPECTED_GENE_MANIFEST_SHA256 = (
    "741e496abe9e14fea59c6070b3b7da86b62fa59c2b982fa66e2ab547bcfbe5b2"
)
EXPECTED_GENE_COUNT = 2991


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_sha256(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    observed = file_sha256(path)
    if observed.lower() != expected.lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: observed {observed}, expected {expected}"
        )
    return observed


def parse_env_bool(name: str, default: bool, environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {raw!r}"
    )


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def resolve_preprocess_paths(
    base_dir: Path, environ: Mapping[str, str] | None = None
) -> tuple[Path, Path, Path]:
    """Resolve source, candidate output, and frozen manifest paths.

    ``LEWORLD_PREPROCESS_OUTPUT_DIR`` names the final ``leworld_input``
    candidate directory, not its parent.
    """
    env = os.environ if environ is None else environ
    base_dir = base_dir.resolve()
    rds_path = _resolve_path(
        base_dir,
        env.get(
            "LEWORLD_RDS_FILE",
            "data/raw/GSE151658_integrated.0h.1h_4h_16h_27h_36h_48h.rds.gz",
        ),
    )
    output_dir = _resolve_path(
        base_dir,
        env.get(
            "LEWORLD_PREPROCESS_OUTPUT_DIR",
            "data/integrated_processed/leworld_input",
        ),
    )
    frozen_manifest = _resolve_path(
        base_dir,
        env.get(
            "LEWORLD_FROZEN_GENE_MANIFEST",
            "data/leworld_input/gene_names.txt",
        ),
    )
    protected_dir = (base_dir / "data" / "leworld_input").resolve()
    if output_dir == protected_dir:
        raise ValueError(
            "refusing to preprocess directly into the frozen data/leworld_input directory"
        )
    return rds_path, output_dir, frozen_manifest


def load_frozen_gene_manifest(
    path: Path,
    expected_sha256: str = EXPECTED_GENE_MANIFEST_SHA256,
    expected_count: int = EXPECTED_GENE_COUNT,
) -> tuple[list[str], str]:
    observed_hash = verify_file_sha256(path, expected_sha256, "frozen gene manifest")
    genes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not gene for gene in genes):
        raise ValueError("frozen gene manifest contains blank gene names")
    if len(genes) != expected_count:
        raise ValueError(
            f"frozen gene manifest contains {len(genes)} genes, expected {expected_count}"
        )
    if len(set(genes)) != len(genes):
        raise ValueError("frozen gene manifest contains exact duplicate gene names")
    if len({gene.upper() for gene in genes}) != len(genes):
        raise ValueError("frozen gene manifest contains case-insensitive duplicates")
    return genes, observed_hash


def resolve_frozen_gene_indices(
    source_genes: Sequence[str], frozen_genes: Sequence[str]
) -> tuple[list[int], dict[str, str]]:
    """Resolve frozen genes against source names, preserving frozen order.

    Exact matches take precedence. Otherwise a case-insensitive match must be
    unique. Missing and ambiguous names are reported together.
    """
    exact: dict[str, list[int]] = defaultdict(list)
    folded: dict[str, list[int]] = defaultdict(list)
    source = [str(gene) for gene in source_genes]
    for index, gene in enumerate(source):
        exact[gene].append(index)
        folded[gene.upper()].append(index)

    indices: list[int] = []
    remapped_case: dict[str, str] = {}
    missing: list[str] = []
    ambiguous: dict[str, list[str]] = {}
    for frozen in frozen_genes:
        exact_matches = exact.get(frozen, [])
        if len(exact_matches) == 1:
            indices.append(exact_matches[0])
            continue
        if len(exact_matches) > 1:
            ambiguous[frozen] = [source[index] for index in exact_matches]
            continue
        folded_matches = folded.get(frozen.upper(), [])
        if not folded_matches:
            missing.append(frozen)
            continue
        if len(folded_matches) > 1:
            ambiguous[frozen] = [source[index] for index in folded_matches]
            continue
        index = folded_matches[0]
        indices.append(index)
        remapped_case[frozen] = source[index]

    if missing or ambiguous:
        parts = []
        if missing:
            preview = ", ".join(missing[:20])
            suffix = " ..." if len(missing) > 20 else ""
            parts.append(f"missing frozen genes ({len(missing)}): {preview}{suffix}")
        if ambiguous:
            preview_items = list(ambiguous.items())[:10]
            preview = "; ".join(
                f"{gene} -> {matches}" for gene, matches in preview_items
            )
            suffix = " ..." if len(ambiguous) > 10 else ""
            parts.append(
                f"ambiguous frozen genes ({len(ambiguous)}): {preview}{suffix}"
            )
        raise ValueError("; ".join(parts))
    if len(indices) != len(frozen_genes) or len(set(indices)) != len(indices):
        raise ValueError("frozen gene resolution did not produce a unique full panel")
    return indices, remapped_case
