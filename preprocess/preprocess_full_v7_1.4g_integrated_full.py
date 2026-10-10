#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild GSE151658 as an isolated candidate using frozen cell/gene manifests."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import rdata
import scanpy as sc
import scipy

from preprocess_contract import (
    EXPECTED_GENE_COUNT,
    EXPECTED_RDS_SHA256,
    file_sha256,
    load_frozen_gene_manifest,
    parse_env_bool,
    resolve_frozen_gene_indices,
    resolve_preprocess_paths,
    verify_file_sha256,
)


EXPECTED_METADATA_SHA256 = (
    "80223b5d11d743425550e06ce2413c6dc79a735b073b6ef0a82fcc5ed4ffe3c6"
)
EXPECTED_SUMMARY_SHA256 = (
    "7136c5f45bb0404f438ece0a2fe0b768eb8fd029dfbaa9481c14d81a8d6fe3c5"
)
EXPECTED_CELL_COUNT = 38439
EXPECTED_TIME_COUNTS = {
    0: 4867,
    1: 5216,
    4: 5345,
    16: 4498,
    27: 4000,
    36: 6093,
    48: 8420,
}


class Config:
    N_PCA = 50
    MIN_GENES = 200
    MAX_GENES = 6000
    MAX_MT = 25
    MIN_CELLS = 3
    TARGET_SUM = 1e4
    SCALE_MAX_VALUE = 10
    PROTECTED_GENES = [
        "Nfkbia", "Tnfaip3", "Il6", "Il1b", "Tnf", "Cxcl10",
        "Itgax", "Cd209a", "H2-Ab1", "H2-Eb1", "Flt3", "Zbtb46",
    ]

    def __init__(self) -> None:
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        (
            self.RDS_FILE,
            self.LEWORLD_DIR,
            self.FROZEN_GENE_MANIFEST,
        ) = resolve_preprocess_paths(self.BASE_DIR)
        frozen_metadata = Path(
            os.environ.get(
                "LEWORLD_FROZEN_METADATA", "data/leworld_input/metadata.csv"
            )
        ).expanduser()
        if not frozen_metadata.is_absolute():
            frozen_metadata = self.BASE_DIR / frozen_metadata
        self.FROZEN_METADATA = frozen_metadata.resolve()
        frozen_summary = Path(
            os.environ.get(
                "LEWORLD_FROZEN_SUMMARY", "data/leworld_input/data_summary.json"
            )
        ).expanduser()
        if not frozen_summary.is_absolute():
            frozen_summary = self.BASE_DIR / frozen_summary
        self.FROZEN_SUMMARY = frozen_summary.resolve()
        self.COMPUTE_PCA = parse_env_bool("LEWORLD_COMPUTE_PCA", False)
        if not parse_env_bool("LEWORLD_REQUIRE_FROZEN_PANEL", True):
            raise ValueError(
                "LEWORLD_REQUIRE_FROZEN_PANEL=0 is unsafe and unsupported"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-inputs-only",
        action="store_true",
        help="verify source/manifests without parsing the large RDS",
    )
    return parser.parse_args(argv)


def load_frozen_metadata(cfg: Config) -> tuple[pd.DataFrame, str]:
    observed_hash = verify_file_sha256(
        cfg.FROZEN_METADATA, EXPECTED_METADATA_SHA256, "frozen metadata"
    )
    metadata = pd.read_csv(cfg.FROZEN_METADATA, index_col=0)
    if len(metadata) != EXPECTED_CELL_COUNT:
        raise ValueError(
            f"frozen metadata contains {len(metadata)} rows, expected {EXPECTED_CELL_COUNT}"
        )
    if not metadata.index.is_unique:
        raise ValueError("frozen metadata cell index is not unique")
    if "timepoint_numeric" not in metadata.columns:
        raise ValueError("frozen metadata lacks timepoint_numeric")
    counts = Counter(int(value) for value in metadata["timepoint_numeric"])
    if dict(sorted(counts.items())) != EXPECTED_TIME_COUNTS:
        raise ValueError(
            f"frozen metadata time counts {dict(sorted(counts.items()))} "
            f"!= expected {EXPECTED_TIME_COUNTS}"
        )
    previous_source_index = -1
    for name in metadata.index.astype(str):
        match = re.fullmatch(r"Cell_(\d+)", name)
        if match is None:
            raise ValueError(f"invalid frozen cell identifier: {name!r}")
        source_index = int(match.group(1))
        if source_index <= previous_source_index:
            raise ValueError("frozen cell identifiers are not strictly increasing")
        previous_source_index = source_index
    return metadata, observed_hash


def verify_inputs(cfg: Config) -> dict:
    rds_hash = verify_file_sha256(cfg.RDS_FILE, EXPECTED_RDS_SHA256, "source RDS")
    genes, gene_hash = load_frozen_gene_manifest(cfg.FROZEN_GENE_MANIFEST)
    metadata, metadata_hash = load_frozen_metadata(cfg)
    summary_hash = verify_file_sha256(
        cfg.FROZEN_SUMMARY, EXPECTED_SUMMARY_SHA256, "frozen data summary"
    )
    summary = json.loads(cfg.FROZEN_SUMMARY.read_text(encoding="utf-8"))
    expected_summary = {
        "actual_input_dim": EXPECTED_GENE_COUNT,
        "n_cells": EXPECTED_CELL_COUNT,
        "n_genes": EXPECTED_GENE_COUNT,
        "protected_genes": Config.PROTECTED_GENES,
        "timepoints": sorted(EXPECTED_TIME_COUNTS),
    }
    if summary != expected_summary:
        raise ValueError("frozen data summary content does not match the expected contract")
    return {
        "rds_sha256": rds_hash,
        "rds_size_bytes": cfg.RDS_FILE.stat().st_size,
        "frozen_genes": genes,
        "frozen_gene_manifest_sha256": gene_hash,
        "frozen_metadata": metadata,
        "frozen_metadata_sha256": metadata_hash,
        "frozen_summary_sha256": summary_hash,
    }


def load_rds_safe(path: Path):
    print(f"📦 加载已校验 RDS: {path}")
    obj = rdata.read_rds(path)
    print(f"✅ RDS 加载成功 | 压缩文件: {path.stat().st_size / 1024**3:.2f} GiB")
    return obj


def parse_metadata(obj) -> pd.DataFrame:
    print("🔍 解析元数据...")
    meta = getattr(obj, "meta.data", None)
    if meta is None and hasattr(obj, "object"):
        meta = getattr(obj.object, "meta.data", None)
    if meta is None:
        raise ValueError("无法提取 meta.data")
    df = meta.to_dataframe() if hasattr(meta, "to_dataframe") else pd.DataFrame(meta)
    if not df.index.is_unique:
        raise ValueError("RDS meta.data cell index is not unique")

    time_col = None
    for column in df.columns:
        normalized = str(column).lower().strip()
        if normalized in {"hrs", "hr", "time", "hour", "timepoint"}:
            time_col = column
            break
    if time_col is None:
        for column in df.columns:
            normalized = str(column).lower()
            if "hr" in normalized and "mito" not in normalized:
                time_col = column
                break
    if time_col is None:
        raise ValueError("未找到有效时间列")
    parsed = pd.to_numeric(
        df[time_col].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    if parsed.isna().any():
        raise ValueError(f"时间列 {time_col!r} 有无法解析的值")
    df = df.copy()
    df["timepoint_numeric"] = parsed.astype(int)
    print(
        f"✅ 元数据: {df.shape} | 时间列: {time_col} | "
        f"分布: {df['timepoint_numeric'].value_counts().sort_index().to_dict()}"
    )
    return df


def _dimnames(raw) -> tuple[list[str], list[str]]:
    names = getattr(raw, "Dimnames", None)
    if names is None or len(names) < 2:
        raise ValueError("表达矩阵缺少基因/细胞 Dimnames，无法验证行列对齐")
    return [str(value) for value in names[0]], [str(value) for value in names[1]]


def _first_mismatch(left: list[str], right: list[str]) -> str:
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        if lhs != rhs:
            return f"index {index}: left={lhs!r}, right={rhs!r}"
    return f"lengths differ: left={len(left)}, right={len(right)}"


def extract_expression(obj, meta_df: pd.DataFrame) -> tuple[anndata.AnnData, dict]:
    print("🧬 提取表达矩阵...")
    assays = getattr(obj, "assays", None)
    if assays is None and hasattr(obj, "object"):
        assays = getattr(obj.object, "assays", None)
    if assays is None:
        raise ValueError("无法提取 assays")
    assay_name = "RNA" if "RNA" in assays else list(assays.keys())[0]
    assay = assays[assay_name]

    raw = None
    slot_name = None
    for candidate in ("counts", "data"):
        value = getattr(assay, candidate, None)
        needed = ("i", "p", "x", "Dim")
        if value is not None and all(hasattr(value, attr) for attr in needed):
            raw, slot_name = value, candidate
            break
    if raw is None or slot_name is None:
        raise ValueError("无法提取 counts/data 稀疏表达矩阵")

    from scipy.sparse import csc_matrix

    dimensions = tuple(int(value) for value in np.asarray(raw.Dim))
    matrix = csc_matrix(
        (np.asarray(raw.x), np.asarray(raw.i), np.asarray(raw.p)),
        shape=dimensions,
    ).tocsr()
    gene_names, cell_names = _dimnames(raw)
    if len(gene_names) != matrix.shape[0] or len(cell_names) != matrix.shape[1]:
        raise ValueError("表达矩阵 Dimnames 长度与矩阵形状不匹配")
    metadata_cells = [str(value) for value in meta_df.index]
    if cell_names != metadata_cells:
        raise ValueError(
            "表达矩阵细胞列与 meta.data 行顺序不一致; "
            + _first_mismatch(cell_names, metadata_cells)
        )

    adata = sc.AnnData(X=matrix.T.tocsr(), var=pd.DataFrame(index=gene_names))
    adata.obs = meta_df.reset_index(drop=True)
    adata.obs.index = [f"Cell_{index}" for index in range(adata.n_obs)]
    print(
        f"✅ assay={assay_name}, slot={slot_name}, AnnData={adata.shape}; "
        "表达列与元数据行已逐项匹配"
    )
    return adata, {
        "assay": assay_name,
        "slot": slot_name,
        "source_cells": int(adata.n_obs),
        "source_genes": int(adata.n_vars),
        "cell_dimnames_match_metadata_index": True,
    }


def run_pipeline(
    adata: anndata.AnnData,
    frozen_genes: list[str],
    frozen_metadata: pd.DataFrame,
    cfg: Config,
) -> tuple[anndata.AnnData, dict]:
    print("\n🚀 运行冻结面板预处理...")
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=True, inplace=True
    )
    mask = (
        (adata.obs["n_genes_by_counts"] >= cfg.MIN_GENES)
        & (adata.obs["n_genes_by_counts"] <= cfg.MAX_GENES)
        & (adata.obs["pct_counts_mt"] <= cfg.MAX_MT)
    )
    retained_cells = list(adata.obs_names[mask])
    expected_cells = [str(value) for value in frozen_metadata.index]
    if retained_cells != expected_cells:
        raise ValueError(
            "当前 QC 细胞集合/顺序与冻结 metadata 不一致; "
            + _first_mismatch(retained_cells, expected_cells)
        )
    adata = adata[mask].copy()
    if list(adata.obs["timepoint_numeric"].astype(int)) != list(
        frozen_metadata["timepoint_numeric"].astype(int)
    ):
        raise ValueError("QC 后时间标签与冻结 metadata 不一致")
    sc.pp.filter_genes(adata, min_cells=cfg.MIN_CELLS)
    print(f"✅ QC 后细胞顺序与冻结 manifest 一致: {adata.shape}")

    sc.pp.normalize_total(adata, target_sum=cfg.TARGET_SUM)
    sc.pp.log1p(adata)
    indices, remapped_case = resolve_frozen_gene_indices(
        list(adata.var_names), frozen_genes
    )
    adata = adata[:, indices].copy()
    adata.var_names = frozen_genes
    if list(adata.var_names) != frozen_genes:
        raise ValueError("冻结基因面板选择后顺序不一致")
    sc.pp.scale(adata, max_value=cfg.SCALE_MAX_VALUE)

    if cfg.COMPUTE_PCA:
        n_components = min(cfg.N_PCA, adata.n_obs - 1, adata.n_vars - 1)
        sc.tl.pca(adata, n_comps=n_components)
        print(f"✅ PCA 已计算: {n_components} components")
    else:
        print("ℹ️  PCA 已关闭（训练输入不消费 PCA）")
    gc.collect()
    return adata, {
        "qc_retained_cells_match_frozen_metadata": True,
        "frozen_gene_order_match": True,
        "case_insensitive_gene_remaps": remapped_case,
    }


def _software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "anndata": importlib.metadata.version("anndata"),
        "scanpy": importlib.metadata.version("scanpy"),
        "rdata": importlib.metadata.version("rdata"),
    }


def save_candidate(
    adata: anndata.AnnData,
    cfg: Config,
    verified: dict,
    extraction: dict,
    pipeline: dict,
) -> Path:
    output_dir = cfg.LEWORLD_DIR
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(
            f"candidate output already exists; refusing to overwrite: {output_dir}"
        )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        matrix = np.asarray(adata.X, dtype=np.float32, order="C")
        if matrix.shape != (EXPECTED_CELL_COUNT, EXPECTED_GENE_COUNT):
            raise ValueError(f"unexpected candidate matrix shape: {matrix.shape}")
        np.save(staging / "expression_matrix.npy", matrix, allow_pickle=False)
        shutil.copyfile(cfg.FROZEN_METADATA, staging / "metadata.csv")
        (staging / "gene_names.txt").write_text(
            "\n".join(str(value) for value in adata.var_names) + "\n",
            encoding="utf-8",
        )
        time_counts = dict(
            sorted(Counter(int(v) for v in verified["frozen_metadata"]["timepoint_numeric"]).items())
        )
        shutil.copyfile(cfg.FROZEN_SUMMARY, staging / "data_summary.json")
        file_names = (
            "expression_matrix.npy", "metadata.csv", "gene_names.txt",
            "data_summary.json",
        )
        output_files = {
            name: {
                "sha256": file_sha256(staging / name),
                "size_bytes": (staging / name).stat().st_size,
            }
            for name in file_names
        }
        provenance = {
            "status": "completed_isolated_candidate",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "accession": "GSE151658",
                "rds_path": str(cfg.RDS_FILE),
                "rds_size_bytes": verified["rds_size_bytes"],
                "rds_sha256": verified["rds_sha256"],
                "expected_rds_sha256": EXPECTED_RDS_SHA256,
            },
            "frozen_manifests": {
                "gene_path": str(cfg.FROZEN_GENE_MANIFEST),
                "gene_sha256": verified["frozen_gene_manifest_sha256"],
                "metadata_path": str(cfg.FROZEN_METADATA),
                "metadata_sha256": verified["frozen_metadata_sha256"],
                "metadata_output_mode": "byte_for_byte_authoritative_copy_after_alignment",
                "summary_path": str(cfg.FROZEN_SUMMARY),
                "summary_sha256": verified["frozen_summary_sha256"],
                "summary_output_mode": "byte_for_byte_authoritative_copy",
            },
            "configuration": {
                "min_genes": cfg.MIN_GENES,
                "max_genes": cfg.MAX_GENES,
                "max_mito_percent": cfg.MAX_MT,
                "min_cells_per_gene": cfg.MIN_CELLS,
                "normalize_target_sum": cfg.TARGET_SUM,
                "scale_max_value": cfg.SCALE_MAX_VALUE,
                "compute_pca": cfg.COMPUTE_PCA,
                "n_pca": cfg.N_PCA if cfg.COMPUTE_PCA else 0,
            },
            "alignment": {**extraction, **pipeline},
            "candidate": {
                "shape": [EXPECTED_CELL_COUNT, EXPECTED_GENE_COUNT],
                "dtype": "float32",
                "c_contiguous": bool(matrix.flags.c_contiguous),
                "time_counts": {str(key): value for key, value in time_counts.items()},
                "files": output_files,
            },
            "software_versions": _software_versions(),
            "source_hashes": {
                "preprocess_script_sha256": file_sha256(Path(__file__).resolve()),
                "preprocess_contract_sha256": file_sha256(
                    Path(__file__).resolve().with_name("preprocess_contract.py")
                ),
            },
            "warning": (
                "Cell/feature manifests are exact, but floating-point matrix bytes "
                "are not assumed reproducible across numerical stacks."
            ),
        }
        (staging / "preprocess_provenance.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_dir)
        print(f"✅ 隔离候选已原子发布: {output_dir}")
        return output_dir
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Config()
    print("=" * 72)
    print("🔬 GSE151658 冻结面板候选重建")
    print("=" * 72)
    print(f"RDS: {cfg.RDS_FILE}")
    print(f"candidate: {cfg.LEWORLD_DIR}")
    print(f"PCA: {'on' if cfg.COMPUTE_PCA else 'off'}")
    verified = verify_inputs(cfg)
    print(
        "✅ 输入校验通过 | "
        f"RDS sha256={verified['rds_sha256']} | "
        f"genes={len(verified['frozen_genes'])} | "
        f"cells={len(verified['frozen_metadata'])}"
    )
    if args.verify_inputs_only:
        return 0

    obj = load_rds_safe(cfg.RDS_FILE)
    metadata = parse_metadata(obj)
    adata, extraction = extract_expression(obj, metadata)
    del obj
    gc.collect()
    print("✅ 已释放展开的 Seurat/RDS 对象")
    adata, pipeline = run_pipeline(
        adata, verified["frozen_genes"], verified["frozen_metadata"], cfg
    )
    save_candidate(adata, cfg, verified, extraction, pipeline)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
