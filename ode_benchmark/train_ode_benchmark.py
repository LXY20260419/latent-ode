#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
LEWorldModel-ODE Benchmark v2.0 (Production Ready)
基于 torchdiffeq 的最佳实践单细胞时序连续动力学建模
✅ 公平对比：分别保存 GRU/ODE 最佳权重
✅ Figure 3: 自动生成 Gap 衰减对比图
✅ Figure 4: 自动生成关键基因轨迹可视化
✅ 完整诊断：集成 v8 的 DiagnosticAnalyzer + TripleValidation
=============================================================================
"""
import os, re, sys, json, csv, hashlib, random, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
from urllib.parse import quote_plus
from datetime import datetime
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import pearsonr, spearmanr, hypergeom
try:
    import torchdiffeq
except ImportError:
    raise ImportError("请安装 torchdiffeq: pip install torchdiffeq==0.2.3")
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# =============================================================================
# 结果 tag 安全校验与命名（纯函数，无 torch/numpy 依赖）
# =============================================================================
RESULT_TAG_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_result_tag(tag: str) -> str:
    """Validate LEWORLD_RESULT_TAG with audit_freeze-level strictness.

    Returns the tag unchanged (possibly empty).  Rejects path separators,
    whitespace, '..', '.' and any character outside [A-Za-z0-9._-] so a tag
    can never traverse out of the artifact directory.  Runs at import time,
    before any data loading or artifact directory creation (fail closed).
    """
    if tag and (not RESULT_TAG_PATTERN.fullmatch(tag) or tag in {".", ".."}):
        raise ValueError(
            f"invalid LEWORLD_RESULT_TAG {tag!r}: only letters, digits, '.', '_' and '-' "
            "are allowed; path separators and '..' are rejected"
        )
    return tag


def tagged_filename(stem: str, ext: str, tag: str, suffix: str = "") -> str:
    """Build ``stem[_tag][_suffix].ext`` for sidecar artifacts.

    The tag always sits directly after the stem, before an optional seed
    suffix and the extension, matching the validation_results naming
    convention.  An empty tag keeps the legacy filename untouched.
    """
    parts = [stem]
    if tag:
        parts.append(tag)
    if suffix:
        parts.append(suffix)
    return "_".join(parts) + "." + ext


def checkpoint_filename(result_stem: str, tag: str, seed: int) -> str:
    """Build a checkpoint filename without breaking legacy untagged runs.

    Tagged runs are production/reviewer namespaces and may execute several
    random seeds, so the seed must be part of the durable checkpoint name.
    Untagged runs retain the historical ``best_{model}.pt`` contract.
    """
    if tag:
        return f"best_{result_stem}_seed{int(seed)}.pt"
    return f"best_{result_stem}.pt"


def split_internal_validation(indices, timepoints, validation_fraction, seed):
    """Deterministically split training-time cells by timepoint."""
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("internal validation fraction must be between 0 and 1")
    by_time = {}
    for idx in indices:
        by_time.setdefault(int(timepoints[idx]), []).append(int(idx))
    rng = random.Random(int(seed))
    train_indices, validation_indices = [], []
    for timepoint in sorted(by_time):
        group = sorted(by_time[timepoint])
        rng.shuffle(group)
        if len(group) < 2:
            train_indices.extend(group)
            continue
        n_validation = max(1, int(round(len(group) * float(validation_fraction))))
        n_validation = min(n_validation, len(group) - 1)
        validation_indices.extend(group[:n_validation])
        train_indices.extend(group[n_validation:])
    return sorted(train_indices), sorted(validation_indices)


def set_seed(seed=42):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
RUN_SEED = int(os.getenv("LEWORLD_SEED", "42"))
RESULT_TAG = validate_result_tag(os.getenv("LEWORLD_RESULT_TAG", "").strip())
set_seed(RUN_SEED)

# =============================================================================
# 工具函数
# =============================================================================
def convert_to_native(obj):
    if isinstance(obj, dict):
        return { (int(k) if isinstance(k, np.integer) else k): convert_to_native(v) for k, v in obj.items() }
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    elif isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, np.bool_): return bool(obj)
    else: return obj

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

def load_biology_priors(config_path):
    if not config_path.exists():
        print(f"⚠️ biology priors not found: {config_path}")
        return {}
    with open(config_path, 'r') as f:
        payload = json.load(f)
    print(f"✅ loaded biology priors: {config_path}")
    return payload

def load_gene_sets_from_gmt(gmt_path):
    gene_sets = {}
    if not gmt_path or not Path(gmt_path).exists():
        return gene_sets
    with open(gmt_path, 'r') as f:
        for raw in f:
            parts = raw.strip().split('\t')
            if len(parts) < 3:
                continue
            set_name = parts[0].strip()
            genes = [g.strip() for g in parts[2:] if g.strip()]
            if set_name and genes:
                gene_sets[set_name] = genes
    print(f"✅ loaded GMT pathways: {gmt_path} ({len(gene_sets)} sets)")
    return gene_sets

def load_tf_targets_from_csv(csv_path):
    tf_targets = {}
    if not csv_path or not Path(csv_path).exists():
        return tf_targets
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        required = {'tf', 'target'}
        if not required.issubset(set(reader.fieldnames or [])):
            print(f"⚠️ TF CSV missing required columns tf,target: {csv_path}")
            return tf_targets
        for row in reader:
            tf = (row.get('tf') or '').strip()
            target = (row.get('target') or '').strip()
            if not tf or not target:
                continue
            tf_targets.setdefault(tf, []).append(target)
    print(f"✅ loaded TF targets CSV: {csv_path} ({len(tf_targets)} TFs)")
    return tf_targets

def merge_priors(base_priors, extra_pathways=None, extra_tf_targets=None):
    merged = {
        'pathways': dict(base_priors.get('pathways', {})),
        'tf_targets': dict(base_priors.get('tf_targets', {})),
        'marker_sets': dict(base_priors.get('marker_sets', {})),
        'literature_terms': dict(base_priors.get('literature_terms', {})),
        'source': {
            'base_json': str(base_priors.get('source', {}).get('base_json', 'default')),
            'external_gmt': None,
            'external_tf_csv': None
        }
    }
    for k, v in (extra_pathways or {}).items():
        merged['pathways'][k] = v
    for tf, genes in (extra_tf_targets or {}).items():
        uniq = list(dict.fromkeys(genes))
        merged['tf_targets'][tf] = uniq
    return merged

def match_gene_sets(gene_names, priors, min_genes=2):
    gene_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    context = {
        'pathways': {},
        'tf_targets': {},
        'marker_sets': {},
        'literature_terms': priors.get('literature_terms', {})
    }
    for group_key in ['pathways', 'tf_targets']:
        for name, genes in priors.get(group_key, {}).items():
            idxs = [gene_to_idx[g.upper()] for g in genes if g.upper() in gene_to_idx]
            if len(idxs) >= min_genes:
                context[group_key][name] = sorted(set(idxs))
            print(f"   {group_key}:{name} matched {len(idxs)}/{len(genes)} genes")
    for name, genes in priors.get('marker_sets', {}).items():
        idxs = [gene_to_idx[g.upper()] for g in genes if g.upper() in gene_to_idx]
        if idxs:
            context['marker_sets'][name] = sorted(set(idxs))
        print(f"   marker_sets:{name} matched {len(idxs)}/{len(genes)} genes")
    return context

def compute_pathway_scores_tensor(x, pathway_map):
    if not pathway_map:
        return None
    cols = []
    for idxs in pathway_map.values():
        cols.append(x[:, idxs].mean(dim=1, keepdim=True))
    return torch.cat(cols, dim=1) if cols else None

def pick_next_times(times):
    ordered = sorted(times)
    next_map = {}
    for i, t in enumerate(ordered):
        next_map[t] = ordered[i + 1] if i + 1 < len(ordered) else t
    return next_map

def build_future_targets(x, t, next_time_map, time_means, device):
    targets = []
    for t_raw in t.detach().cpu().numpy().tolist():
        t_key = int(round(float(t_raw)))
        t_next = next_time_map.get(t_key, t_key)
        if t_next in time_means:
            targets.append(time_means[t_next])
        elif t_key in time_means:
            targets.append(time_means[t_key])
        else:
            targets.append(np.zeros(x.shape[1], dtype=np.float32))
    arr = np.stack(targets, axis=0).astype(np.float32)
    return torch.FloatTensor(arr).to(device)

def safe_corr(a, b):
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return None
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    corr, _ = pearsonr(a, b)
    return float(corr)

# =============================================================================
# 配置
# =============================================================================
def _parse_int_list(env_name: str, default: list[int]) -> list[int]:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = Path(os.getenv("LEWORLD_DATA_DIR", str(BASE_DIR / "data" / "leworld_input")))
    MODEL_DIR = BASE_DIR / "artifacts" / "models" / "ode_benchmark"
    RESULT_DIR = BASE_DIR / "artifacts" / "results" / "ode_benchmark"
    INPUT_DIM = 2991
    LATENT_DIM = 32
    HIDDEN_DIM = 64
    DROPOUT = 0.3
    BATCH_SIZE = 32
    N_EPOCHS = int(os.getenv("LEWORLD_EPOCHS", "100"))
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-3
    EARLY_STOPPING_PATIENCE = 15
    MIN_EPOCHS = 20
    INTERNAL_VAL_FRACTION = float(os.getenv("LEWORLD_INTERNAL_VAL_FRACTION", "0.1"))
    HOLDOUT_TIMEPOINTS = _parse_int_list("LEWORLD_HOLDOUT_TIMEPOINTS", [36, 48])
    TRAIN_TIMEPOINTS = _parse_int_list("LEWORLD_TRAIN_TIMEPOINTS", [0, 1, 4, 16, 27])
    LPS_GENES = ['Nfkbia', 'Tnfaip3', 'Il6', 'Il1b', 'Tnf', 'Cxcl10']
    BIOLOGY_PRIORS_PATH = Path(
        os.getenv(
            "LEWORLD_BIOLOGY_PRIORS_PATH",
            str(BASE_DIR / "configs" / "biology_priors.json"),
        )
    )
    EXTERNAL_PATHWAY_GMT = os.getenv("LEWORLD_EXTERNAL_PATHWAY_GMT", str(BASE_DIR / "configs" / "external_pathways.gmt"))
    EXTERNAL_TF_TARGET_CSV = os.getenv("LEWORLD_EXTERNAL_TF_TARGET_CSV", str(BASE_DIR / "configs" / "external_tf_targets.csv"))
    PATHWAY_WEIGHT = float(os.getenv("LEWORLD_PATHWAY_WEIGHT", "0.2"))
    PATHWAY_DELTA_WEIGHT = float(os.getenv("LEWORLD_PATHWAY_DELTA_WEIGHT", "0.1"))
    PATHWAY_MIN_GENES = int(os.getenv("LEWORLD_PATHWAY_MIN_GENES", "2"))
    TOP_K_BIO_GENES = int(os.getenv("LEWORLD_TOP_K_BIO_GENES", "25"))
    TRAIN_MAX_PREDICT_STEPS = int(os.getenv("LEWORLD_TRAIN_MAX_PREDICT_STEPS", "8"))
    # ODE 专属参数
    MAX_TIME = float(os.getenv("LEWORLD_MAX_TIME", "48.0"))
    ODE_DELTA = 1.0
    ODE_METHOD = 'dopri5'
    ODE_RTOL = 1e-3
    ODE_ATOL = 1e-4
    LATENT_L2_REG = 0.01
    GRAD_CLIP = 1.0
    def __init__(self):
        for d in [self.MODEL_DIR, self.RESULT_DIR]: d.mkdir(parents=True, exist_ok=True)
cfg = Config()
EVAL_MAX_CELLS = int(os.getenv("LEWORLD_EVAL_MAX_CELLS", "0"))
MAX_CELLS = int(os.getenv("LEWORLD_MAX_CELLS", "0"))
BALANCED_TRAIN = os.getenv("LEWORLD_BALANCED_TRAIN", "").strip().lower() in {"1", "true", "yes"}
BALANCED_CELLS_PER_TIMEPOINT = int(os.getenv("LEWORLD_BALANCED_CELLS_PER_TIMEPOINT", "0"))


def result_suffix(model_name: str) -> str:
    """Return a collision-safe artifact stem.

    The historical workflow allowed a non-primary dataset to write to the
    untagged ``validation_results_{model}.json`` path.  That made it possible
    for an external run to silently overwrite the frozen internal artifacts.
    Keep the legacy internal filenames for the canonical ``leworld_input``
    dataset, but require an explicit tag for every other dataset.
    """
    dataset_name = cfg.DATA_DIR.name
    if dataset_name != "leworld_input" and not RESULT_TAG:
        raise RuntimeError(
            "Refusing to write untagged artifacts for a non-primary dataset. "
            "Set LEWORLD_RESULT_TAG (for example gse224518_3to5h) before running."
        )
    return f"{model_name}_{RESULT_TAG}" if RESULT_TAG else model_name


def checkpoint_path(model_name: str) -> Path:
    """Return a collision-safe checkpoint path for this run and seed."""
    return cfg.MODEL_DIR / checkpoint_filename(
        result_suffix(model_name), RESULT_TAG, RUN_SEED
    )


def artifact_tag() -> str:
    """Human-readable tag for figures and diagnostic sidecars."""
    if RESULT_TAG:
        return RESULT_TAG
    return "internal" if cfg.DATA_DIR.name == "leworld_input" else cfg.DATA_DIR.name


def file_sha256(path: Path) -> str | None:
    """Hash small manifest files without loading expression matrices."""
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def portable_path(path: str | Path | None) -> str | None:
    """Serialize paths relative to the bundle when possible."""
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(Config.BASE_DIR.resolve()))
    except (ValueError, OSError):
        return str(p)


def build_run_metadata(dataset, gene_names, train_times, internal_train_idx, internal_val_idx):
    """Build the provenance record embedded in run artifacts."""
    checkpoint_selection = {
        "strategy": "stratified_internal_validation",
        "validation_fraction": cfg.INTERNAL_VAL_FRACTION,
        "selection_timepoints": list(map(int, train_times)),
        "n_internal_train": len(internal_train_idx),
        "n_internal_validation": len(internal_val_idx),
        "holdout_used_for_selection": False,
    }
    return {
        "status": "completed_run_metadata",
        "dataset": {
            "dataset_id": os.getenv(
                "LEWORLD_DATASET_ID",
                "internal_lps_primary" if cfg.DATA_DIR.name == "leworld_input" else cfg.DATA_DIR.name,
            ),
            "source_accession": os.getenv(
                "LEWORLD_SOURCE_ACCESSION",
                "GSE151658" if cfg.DATA_DIR.name == "leworld_input" else "",
            ) or None,
            "data_dir": portable_path(cfg.DATA_DIR),
            "n_cells": int(len(dataset)),
            "n_genes": int(len(gene_names)),
            "timepoints": sorted(int(t) for t in dataset.time_to_cells),
            "gene_manifest_sha256": file_sha256(cfg.DATA_DIR / "gene_names.txt"),
            "metadata_sha256": file_sha256(cfg.DATA_DIR / "metadata.csv"),
            "expression_matrix_sha256": file_sha256(
                cfg.DATA_DIR / "expression_matrix.npy"
            ),
        },
        "protocol": {
            "protocol_id": os.getenv(
                "LEWORLD_PROTOCOL_ID",
                "paper-v1-freeze-internal-population-extrapolation",
            ),
            "train_timepoints": list(map(int, train_times)),
            "holdout_timepoints": list(map(int, cfg.HOLDOUT_TIMEPOINTS)),
            "source_time_for_extrapolation": int(max(train_times)),
            "primary_metric": "population_mean_across_gene_pearson_r",
            "seed": int(RUN_SEED),
            "result_tag": RESULT_TAG or None,
            "checkpoint_selection": checkpoint_selection,
        },
        "config": {
            "latent_dim": int(cfg.LATENT_DIM),
            "hidden_dim": int(cfg.HIDDEN_DIM),
            "dropout": float(cfg.DROPOUT),
            "batch_size": int(cfg.BATCH_SIZE),
            "epochs": int(cfg.N_EPOCHS),
            "learning_rate": float(cfg.LEARNING_RATE),
            "weight_decay": float(cfg.WEIGHT_DECAY),
            "pathway_weight": float(cfg.PATHWAY_WEIGHT),
            "pathway_delta_weight": float(cfg.PATHWAY_DELTA_WEIGHT),
            "train_max_predict_steps": int(cfg.TRAIN_MAX_PREDICT_STEPS),
            "ode_delta": float(cfg.ODE_DELTA),
            "ode_method": cfg.ODE_METHOD,
            "ode_rtol": float(cfg.ODE_RTOL),
            "ode_atol": float(cfg.ODE_ATOL),
        },
        "source_hashes": {
            "training_script_sha256": file_sha256(Path(__file__)),
            "biology_priors_sha256": file_sha256(cfg.BIOLOGY_PRIORS_PATH),
            "external_pathways_sha256": file_sha256(Path(cfg.EXTERNAL_PATHWAY_GMT)),
            "external_tf_targets_sha256": file_sha256(Path(cfg.EXTERNAL_TF_TARGET_CSV)),
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            # torch.__version__ is a TorchVersion subclass in recent PyTorch.
            # Persist a plain string so checkpoints remain compatible with the
            # safe weights_only=True loader introduced as the default in 2.6.
            "torch": str(torch.__version__),
        },
    }

# =============================================================================
# 数据集
# =============================================================================
class SCTimeSeriesDataset(Dataset):
    def __init__(self, expression_matrix, metadata, gene_names, time_col='timepoint_numeric'):
        self.X = expression_matrix.astype(np.float32)
        self.metadata = metadata
        self.gene_names = gene_names
        if time_col in metadata.columns:
            self.timepoints = metadata[time_col].values
        else:
            self.timepoints = np.zeros(len(metadata))
        self.time_to_cells = {}
        for i, t in enumerate(self.timepoints):
            if t not in self.time_to_cells: self.time_to_cells[t] = []
            self.time_to_cells[t].append(i)
        print(f"📊 数据集统计：细胞数={len(self.X)}, 基因数={self.X.shape[1]}, 时间点={sorted(self.time_to_cells.keys())}")
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return {'expression': self.X[idx], 'timepoint': self.timepoints[idx], 'cell_idx': idx}
    def get_cells_by_timepoint(self, timepoint):
        return self.time_to_cells.get(timepoint, [])


def build_train_indices(dataset: SCTimeSeriesDataset, train_times: list[int]) -> list[int]:
    per_time = {int(t): list(dataset.get_cells_by_timepoint(t)) for t in train_times}
    counts = {t: len(idxs) for t, idxs in per_time.items() if idxs}
    if not counts:
        return []

    if BALANCED_TRAIN:
        cap = BALANCED_CELLS_PER_TIMEPOINT or min(counts.values())
        rng = np.random.default_rng(RUN_SEED)
        train_idx = []
        for t in sorted(counts):
            idxs = per_time[t]
            if len(idxs) > cap:
                chosen = rng.choice(idxs, size=cap, replace=False).tolist()
            else:
                chosen = list(idxs)
            train_idx.extend(chosen)
            print(f"   balanced train t={t}h: {len(chosen)} cells (pool={len(idxs)}, cap={cap})")
        print(f"⚖️  Balanced training set: {len(train_idx)} cells across {len(counts)} timepoints")
        return train_idx

    train_idx = [i for t in train_times for i in per_time.get(int(t), [])]
    for t, n in sorted(counts.items()):
        print(f"   train t={t}h: {n} cells")
    return train_idx

# =============================================================================
# 模型基础组件
# =============================================================================
class SparseAwareEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim), nn.LayerNorm(latent_dim)
        )
    def forward(self, x): return self.network(x)

class TimeEncoder(nn.Module):
    def __init__(self, latent_dim, max_time=100):
        super().__init__()
        self.latent_dim, self.max_time = latent_dim, max_time
        position = torch.arange(max_time).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, latent_dim, 2) * (-np.log(10000.0) / latent_dim))
        pe = torch.zeros(max_time, latent_dim)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div_term), torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, t):
        t = t.long().clamp(0, self.max_time - 1)
        return self.pe[t]

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, z): return self.network(z)

class WorldModelLoss(nn.Module):
    def __init__(self, recon_weight=1.0, pred_weight=1.0, pathway_weight=0.0, pathway_delta_weight=0.0, pathway_map=None):
        super().__init__()
        self.recon_weight, self.pred_weight = recon_weight, pred_weight
        self.pathway_weight = pathway_weight
        self.pathway_delta_weight = pathway_delta_weight
        self.pathway_map = pathway_map or {}
        self.mse = nn.MSELoss(reduction='mean')
    def forward(self, outputs, x_true, x_future_true=None):
        losses = {}
        losses['recon'] = self.mse(outputs['x_recon'], x_true)
        if x_future_true is not None and len(outputs['x_predictions']) > 0:
            # Train to match the target horizon state (last rollout step), not per-step broadcasted rows.
            losses['prediction'] = self.mse(outputs['x_predictions'][-1], x_future_true)
        else:
            losses['prediction'] = torch.tensor(0.0, device=x_true.device)
        losses['pathway_score'] = torch.tensor(0.0, device=x_true.device)
        losses['pathway_delta'] = torch.tensor(0.0, device=x_true.device)
        if self.pathway_map:
            true_scores = compute_pathway_scores_tensor(x_true, self.pathway_map)
            recon_scores = compute_pathway_scores_tensor(outputs['x_recon'], self.pathway_map)
            if true_scores is not None and recon_scores is not None:
                losses['pathway_score'] = self.mse(recon_scores, true_scores)
            if x_future_true is not None and len(outputs['x_predictions']) > 0:
                pred_scores = compute_pathway_scores_tensor(outputs['x_predictions'][-1], self.pathway_map)
                future_scores = compute_pathway_scores_tensor(x_future_true, self.pathway_map)
                if pred_scores is not None and future_scores is not None:
                    pred_delta = pred_scores - true_scores
                    true_delta = future_scores - true_scores
                    losses['pathway_delta'] = self.mse(pred_delta, true_delta)
        losses['total'] = (
            self.recon_weight * losses['recon']
            + self.pred_weight * losses['prediction']
            + self.pathway_weight * losses['pathway_score']
            + self.pathway_delta_weight * losses['pathway_delta']
        )
        return losses

# =============================================================================
# GRU 模型 (基线)
# =============================================================================
class DynamicsPredictor_GRU(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=latent_dim * 2, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
    def forward(self, z, t_embed, predict_steps=1):
        gru_input = torch.cat([z.unsqueeze(1), t_embed.unsqueeze(1)], dim=-1)
        outputs, h = [], None
        current_input = gru_input
        for step in range(predict_steps):
            output, h = self.gru(current_input, h)
            z_next = self.output_proj(output.squeeze(1))
            outputs.append(z_next)
            if step < predict_steps - 1:
                current_input = torch.cat([z_next.unsqueeze(1), t_embed.unsqueeze(1)], dim=-1)
        return outputs

class LEWorldModel_GRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = SparseAwareEncoder(config.INPUT_DIM, config.HIDDEN_DIM, config.LATENT_DIM, config.DROPOUT)
        self.time_encoder = TimeEncoder(latent_dim=config.LATENT_DIM)
        self.dynamics = DynamicsPredictor_GRU(config.LATENT_DIM, config.HIDDEN_DIM, n_layers=1, dropout=config.DROPOUT)
        self.decoder = Decoder(config.LATENT_DIM, config.INPUT_DIM, config.HIDDEN_DIM)
    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)
    def predict(self, z, t, predict_steps=1):
        t_embed = self.time_encoder(t)
        return self.dynamics(z, t_embed, predict_steps)
    def forward(self, x, t, predict_steps=1):
        z = self.encode(x)
        z_predictions = self.predict(z, t, predict_steps)
        x_recon = self.decode(z)
        x_predictions = [self.decode(z_pred) for z_pred in z_predictions]
        return {'z': z, 'x_recon': x_recon, 'z_predictions': z_predictions, 'x_predictions': x_predictions}

# =============================================================================
# Neural ODE 模型 (核心升级)
# =============================================================================
class NeuralODEDynamics(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
    def forward(self, t, z):
        if isinstance(t, torch.Tensor) and t.ndim == 0: t_val = t.item()
        elif isinstance(t, (int, float)): t_val = t
        else: raise ValueError(f"Expected scalar t, got {type(t)}")
        t_broadcast = torch.full((z.shape[0], 1), t_val, device=z.device, dtype=z.dtype)
        return self.net(torch.cat([z, t_broadcast], dim=-1))
    def solve(self, z0, t_start, t_end):
        t_span = torch.tensor([t_start, t_end], device=z0.device)
        sol = torchdiffeq.odeint(self.forward, z0, t_span, method=cfg.ODE_METHOD, rtol=cfg.ODE_RTOL, atol=cfg.ODE_ATOL, options={'max_num_steps': 2000})
        return sol[-1]

class LEWorldModel_ODE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = SparseAwareEncoder(config.INPUT_DIM, config.HIDDEN_DIM, config.LATENT_DIM, config.DROPOUT)
        self.time_encoder = TimeEncoder(latent_dim=config.LATENT_DIM)
        self.dynamics = NeuralODEDynamics(config.LATENT_DIM, config.HIDDEN_DIM)
        self.decoder = Decoder(config.LATENT_DIM, config.INPUT_DIM, config.HIDDEN_DIM)
        self.max_time = config.MAX_TIME
        self.delta = config.ODE_DELTA
    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)
    def predict(self, z, t, predict_steps=1):
        t_norm = t.float() / self.max_time
        delta_norm = self.delta / self.max_time
        z_preds, current_z, current_t = [], z, t_norm
        for _ in range(predict_steps):
            next_t_val = torch.clamp(current_t + delta_norm, 0.0, 1.0)
            unique_starts = torch.unique(current_t, sorted=True)
            z_next_batch = torch.empty_like(current_z)
            for t_s in unique_starts:
                mask = (current_t == t_s)
                if mask.any():
                    t_start, t_end = t_s.item(), (t_s + delta_norm).clamp(0.0, 1.0).item()
                    if t_end <= t_start + 1e-7:
                        z_next_batch[mask] = current_z[mask]
                    else:
                        z_next_batch[mask] = self.dynamics.solve(current_z[mask], t_start, t_end)
            z_preds.append(z_next_batch); current_z, current_t = z_next_batch, next_t_val
        return z_preds
    def forward(self, x, t, predict_steps=1):
        z = self.encode(x)
        z_predictions = self.predict(z, t, predict_steps)
        x_recon = self.decode(z)
        x_predictions = [self.decode(z_p) for z_p in z_predictions]
        return {'z': z, 'x_recon': x_recon, 'z_predictions': z_predictions, 'x_predictions': x_predictions}
    def latent_regularization(self):
        reg = sum(torch.norm(p, p=2) for p in self.encoder.parameters())
        return cfg.LATENT_L2_REG * reg

# =============================================================================
# 训练器
# =============================================================================
class Trainer:
    def __init__(self, model, config, biology_context=None, train_time_means=None, train_times=None, run_metadata=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device); self.config = config; self.device = device
        self.optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.N_EPOCHS)
        pathway_map = (biology_context or {}).get('pathways', {})
        self.loss_fn = WorldModelLoss(
            pathway_weight=config.PATHWAY_WEIGHT,
            pathway_delta_weight=config.PATHWAY_DELTA_WEIGHT,
            pathway_map=pathway_map
        )
        self.history = {'train': [], 'val': [], 'train_terms': [], 'val_terms': []}
        self.train_time_means = train_time_means or {}
        self.next_time_map = pick_next_times(train_times or [])
        self.run_metadata = run_metadata or {}
        self.early_stopping_patience = config.EARLY_STOPPING_PATIENCE
        self.early_stopping_counter = 0; self.best_val_loss = float('inf')
    def _loss_dict_to_float(self, losses):
        out = {}
        for k, v in losses.items():
            out[k] = float(v.item()) if hasattr(v, 'item') else float(v)
        return out
    def _predict_steps_for_gap(self, gap):
        delta = max(float(getattr(self.config, 'ODE_DELTA', 1.0)), 1e-6)
        raw = max(1, int(np.ceil(max(0.0, float(gap)) / delta)))
        return min(raw, max(1, int(getattr(self.config, 'TRAIN_MAX_PREDICT_STEPS', 8))))
    def _build_future_target_tensor(self, source_time, n_rows):
        source_time = int(source_time)
        target_time = int(self.next_time_map.get(source_time, source_time))
        target = self.train_time_means.get(target_time)
        if target is None:
            target = self.train_time_means.get(source_time)
        if target is None:
            target = np.zeros(cfg.INPUT_DIM, dtype=np.float32)
        tiled = np.repeat(target[None, :], n_rows, axis=0).astype(np.float32)
        return torch.FloatTensor(tiled).to(self.device), target_time
    def _batch_losses_with_gap(self, x, t):
        t_int = t.detach().cpu().numpy().astype(int)
        unique_times = sorted(set(t_int.tolist()))
        weighted_losses = None
        weighted_steps = 0.0
        batch_n = max(1, x.shape[0])
        for t_val in unique_times:
            mask = (t_int == t_val)
            x_sub = x[mask]
            t_sub = t[mask]
            if x_sub.shape[0] == 0:
                continue
            x_future_true, target_time = self._build_future_target_tensor(t_val, x_sub.shape[0])
            predict_steps = self._predict_steps_for_gap(target_time - int(t_val))
            outputs = self.model(x_sub, t_sub, predict_steps=predict_steps)
            sub_losses = self.loss_fn(outputs, x_sub, x_future_true=x_future_true)
            weight = x_sub.shape[0] / batch_n
            if weighted_losses is None:
                weighted_losses = {k: v * 0.0 for k, v in sub_losses.items()}
            for k, v in sub_losses.items():
                weighted_losses[k] = weighted_losses[k] + v * weight
            weighted_steps += predict_steps * weight
        if weighted_losses is None:
            weighted_losses = {
                'recon': torch.tensor(0.0, device=self.device),
                'prediction': torch.tensor(0.0, device=self.device),
                'pathway_score': torch.tensor(0.0, device=self.device),
                'pathway_delta': torch.tensor(0.0, device=self.device),
                'total': torch.tensor(0.0, device=self.device),
            }
        weighted_losses['avg_predict_steps'] = torch.tensor(weighted_steps, device=self.device)
        return weighted_losses
    def train_epoch(self, dataloader):
        self.model.train(); loss_acc = {}
        for batch in dataloader:
            x, t = batch['expression'].to(self.device), batch['timepoint'].to(self.device)
            losses = self._batch_losses_with_gap(x, t)
            self.optimizer.zero_grad(); losses['total'].backward()
            max_norm = cfg.GRAD_CLIP if hasattr(self.model, 'latent_regularization') else 1.0
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            self.optimizer.step()
            for k, v in self._loss_dict_to_float(losses).items():
                loss_acc[k] = loss_acc.get(k, 0.0) + v
        return {k: v / len(dataloader) for k, v in loss_acc.items()}
    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval(); loss_acc = {}
        for batch in dataloader:
            x, t = batch['expression'].to(self.device), batch['timepoint'].to(self.device)
            losses = self._batch_losses_with_gap(x, t)
            for k, v in self._loss_dict_to_float(losses).items():
                loss_acc[k] = loss_acc.get(k, 0.0) + v
        return {k: v / len(dataloader) for k, v in loss_acc.items()}
    def train(self, train_loader, val_loader=None, model_name='model'):
        print(f"\n🚀 开始训练 {model_name} | 设备：{self.device}")
        for epoch in range(self.config.N_EPOCHS):
            train_terms = self.train_epoch(train_loader)
            train_loss = train_terms['total']
            self.history['train'].append(train_loss)
            self.history['train_terms'].append(train_terms)
            if val_loader is not None:
                val_terms = self.validate(val_loader)
                val_loss = val_terms['total']
                self.history['val'].append(val_loss)
                self.history['val_terms'].append(val_terms)
                if val_loss < self.best_val_loss - 1e-4:
                    self.best_val_loss = val_loss; self.early_stopping_counter = 0
                    torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(), 'val_loss': val_loss,
                                'run_metadata': self.run_metadata},
                               checkpoint_path(model_name))
                else:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.early_stopping_patience and epoch >= self.config.MIN_EPOCHS:
                        print(f"\n⏹️ 早停触发"); break
            self.scheduler.step()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_str = f"Val: {val_loss:.4f}" if val_loader else ""
                print(
                    f"Epoch {epoch+1:3d}/{self.config.N_EPOCHS} | "
                    f"Train: {train_loss:.4f} (recon={train_terms.get('recon', 0.0):.4f}, "
                    f"pred={train_terms.get('prediction', 0.0):.4f}, "
                    f"path={train_terms.get('pathway_score', 0.0):.4f}, "
                    f"steps={train_terms.get('avg_predict_steps', 0.0):.2f}) | {val_str}"
                )
        history_stem = result_suffix(model_name)
        # Preserve the legacy top-level train/val keys for downstream notebooks
        # while adding the provenance record required by tagged production runs.
        history_payload = {**self.history, "run_metadata": self.run_metadata}
        with open(cfg.RESULT_DIR / f'training_history_{history_stem}.json', 'w') as f:
            json.dump(convert_to_native(history_payload), f, indent=2)
        with open(cfg.RESULT_DIR / f'training_history_{history_stem}_seed{RUN_SEED}.json', 'w') as f:
            json.dump(convert_to_native(history_payload), f, indent=2)
        print(f"✅ {model_name} 训练完成！最佳验证损失：{self.best_val_loss:.4f}")

class Trainer_ODE(Trainer):
    def train_epoch(self, dataloader):
        self.model.train(); loss_acc = {}
        for batch in dataloader:
            x, t = batch['expression'].to(self.device), batch['timepoint'].to(self.device)
            losses = self._batch_losses_with_gap(x, t)
            losses['latent_reg'] = self.model.latent_regularization()
            losses['total'] = losses['total'] + losses['latent_reg']
            self.optimizer.zero_grad(); losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.GRAD_CLIP)
            self.optimizer.step()
            for k, v in self._loss_dict_to_float(losses).items():
                loss_acc[k] = loss_acc.get(k, 0.0) + v
        return {k: v / len(dataloader) for k, v in loss_acc.items()}
    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval(); loss_acc = {}
        for batch in dataloader:
            x, t = batch['expression'].to(self.device), batch['timepoint'].to(self.device)
            losses = self._batch_losses_with_gap(x, t)
            losses['latent_reg'] = self.model.latent_regularization()
            losses['total'] = losses['total'] + losses['latent_reg']
            for k, v in self._loss_dict_to_float(losses).items():
                loss_acc[k] = loss_acc.get(k, 0.0) + v
        return {k: v / len(dataloader) for k, v in loss_acc.items()}

# =============================================================================
# 🆕 Figure 3: Gap 衰减对比图生成器
# =============================================================================
def plot_gap_decay_comparison(gru_model, ode_model, dataset, gene_names, config, device, result_dir):
    """生成 GRU vs ODE 的时间外推性能衰减曲线 (论文 Figure 3)"""
    print("\n📈 生成 Gap 衰减对比图 (Figure 3)...")
    all_times = sorted(dataset.time_to_cells.keys())
    gap_corr_gru, gap_corr_ode = [], []
    
    for model, label, corr_list in [(gru_model, 'GRU', gap_corr_gru), (ode_model, 'ODE', gap_corr_ode)]:
        model.eval()
        with torch.no_grad():
            for i, t in enumerate(all_times):
                if i == 0: gap = 0
                else: gap = t - all_times[i-1]
                indices = dataset.get_cells_by_timepoint(t)
                if not indices: continue
                if EVAL_MAX_CELLS > 0 and len(indices) > EVAL_MAX_CELLS:
                    rng = np.random.default_rng(int(os.getenv("LEWORLD_SEED", "42")))
                    indices = rng.choice(indices, size=EVAL_MAX_CELLS, replace=False).tolist()
                corrs = []
                for idx in indices:
                    x = torch.FloatTensor(dataset[idx]['expression']).unsqueeze(0).to(device)
                    t_tensor = torch.LongTensor([dataset[idx]['timepoint']]).to(device)
                    out = model(x, t_tensor)
                    recon = out['x_recon'].cpu().numpy().flatten()
                    true = x.cpu().numpy().flatten()
                    c, _ = pearsonr(true, recon); corrs.append(c)
                corr_list.append((gap, float(np.mean(corrs))))
    
    # 绘图
    plt.figure(figsize=(8, 5))
    gaps_gru, corrs_gru = zip(*gap_corr_gru) if gap_corr_gru else ([], [])
    gaps_ode, corrs_ode = zip(*gap_corr_ode) if gap_corr_ode else ([], [])
    
    plt.scatter(gaps_gru, corrs_gru, s=80, alpha=0.7, label='GRU (Discrete)', color='#2E86AB', edgecolors='black', zorder=3)
    plt.scatter(gaps_ode, corrs_ode, s=80, alpha=0.7, label='ODE (Continuous)', color='#A23B72', edgecolors='black', zorder=3)
    
    # 拟合衰减曲线
    if len(gaps_gru) > 2:
        from scipy.optimize import curve_fit
        def decay_func(x, a, b, c): return a * np.exp(-b * x) + c
        try:
            popt_gru, _ = curve_fit(decay_func, gaps_gru, corrs_gru, p0=[0.8, 0.05, 0.1], maxfev=5000)
            x_fit = np.linspace(0, max(gaps_gru), 100)
            plt.plot(x_fit, decay_func(x_fit, *popt_gru), '--', color='#2E86AB', alpha=0.5, label='GRU fit')
        except: pass
        try:
            popt_ode, _ = curve_fit(decay_func, gaps_ode, corrs_ode, p0=[0.8, 0.03, 0.1], maxfev=5000)
            plt.plot(x_fit, decay_func(x_fit, *popt_ode), '--', color='#A23B72', alpha=0.5, label='ODE fit')
        except: pass
    
    plt.axvline(x=12, color='gray', linestyle=':', alpha=0.5, label='12h threshold')
    plt.xlabel('Time Gap from Previous Timepoint (hours)')
    plt.ylabel('Reconstruction Pearson Correlation')
    plt.title('Model Performance vs. Prediction Time Gap')
    plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    gap_png = tagged_filename('gap_decay_comparison', 'png', RESULT_TAG)
    plt.savefig(result_dir / gap_png, dpi=300)
    plt.close()
    
    # 打印统计
    short_gru = [c for g, c in gap_corr_gru if g <= 12]
    long_gru = [c for g, c in gap_corr_gru if g > 12]
    short_ode = [c for g, c in gap_corr_ode if g <= 12]
    long_ode = [c for g, c in gap_corr_ode if g > 12]
    print(f"   📊 Gap≤12h: GRU={np.mean(short_gru):.3f}±{np.std(short_gru):.3f}, ODE={np.mean(short_ode):.3f}±{np.std(short_ode):.3f}")
    print(f"   📊 Gap>12h:  GRU={np.mean(long_gru):.3f}±{np.std(long_gru):.3f}, ODE={np.mean(long_ode):.3f}±{np.std(long_ode):.3f}")
    print(f"   ✅ 图已保存: {result_dir / gap_png}")
    return {'gap_corr_gru': gap_corr_gru, 'gap_corr_ode': gap_corr_ode}

# =============================================================================
# 🆕 Figure 4: 关键基因轨迹可视化
# =============================================================================
def plot_gene_trajectory(model, dataset, gene_names, config, device, result_dir, gene_list=None):
    """生成关键基因的真实值 vs 模型预测轨迹 (论文 Figure 4)"""
    if gene_list is None: gene_list = config.LPS_GENES
    print(f"\n🧬 生成关键基因轨迹图 (Figure 4): {gene_list}")
    
    model.eval()
    timepoints = sorted(dataset.time_to_cells.keys())
    
    for gene in gene_list:
        # 查找基因索引
        idx = None
        for i, g in enumerate(gene_names):
            if g.upper() == gene.upper() or g.lower() == gene.lower():
                idx = i; break
        if idx is None: print(f"   ⚠️ 未找到基因: {gene}"); continue
        
        # 提取真实轨迹 (按时间平均)
        true_traj, pred_traj_gru, pred_traj_ode = [], [], []
        with torch.no_grad():
            for t in timepoints:
                cells = dataset.get_cells_by_timepoint(t)
                if not cells: continue
                # 真实值
                vals = [dataset.X[i, idx] for i in cells]
                true_traj.append(np.mean(vals))
                # 预测值 (取一个代表性细胞)
                x = torch.FloatTensor(dataset[cells[0]]['expression']).unsqueeze(0).to(device)
                t_tensor = torch.LongTensor([t]).to(device)
                out = model(x, t_tensor)
                pred = out['x_recon'].cpu().numpy().flatten()[idx]
                pred_traj_ode.append(pred)
        
        # 绘图
        plt.figure(figsize=(6, 4))
        plt.plot(timepoints, true_traj, 'ko-', label='True (mean)', linewidth=2, markersize=6)
        plt.plot(timepoints, pred_traj_ode, 'r--s', label='ODE Prediction', linewidth=1.5, markersize=4)
        plt.xlabel('Time (hours)'); plt.ylabel('Expression (log1p normalized)')
        plt.title(f'{gene} Dynamics'); plt.legend(fontsize=9); plt.grid(alpha=0.3)
        plt.tight_layout()
        traj_png = tagged_filename(f'{gene}_trajectory', 'png', RESULT_TAG)
        plt.savefig(result_dir / traj_png, dpi=300)
        plt.close()
        print(f"   ✅ {gene}: {result_dir / traj_png}")

# =============================================================================
# 诊断模块 (完整集成 v8)
# =============================================================================
class DiagnosticAnalyzer:
    def __init__(self, model, dataset, gene_names, config, device='cpu'):
        self.model = model.to(device); self.dataset = dataset; self.gene_names = gene_names
        self.config = config; self.device = device; self.results = {}
    def _get_gene_index(self, gene_name):
        target = gene_name.upper()
        for i, g in enumerate(self.gene_names):
            if g.upper() == target: return i
        return None
    def analyze_timepoint_distribution(self):
        print("\n" + "="*60); print("📊 诊断 1: 时间点细胞分布与采样密度"); print("="*60)
        times = sorted(self.dataset.time_to_cells.keys())
        counts = [len(self.dataset.time_to_cells[t]) for t in times]
        for t, c in zip(times, counts): print(f"   {t}h: {c} 细胞 ({c/len(self.dataset)*100:.1f}%)")
        if max(counts) > 3 * min(counts): print("   ⚠️  时间点分布高度不均")
        else: print("   ✅ 时间点分布相对均衡")
        gaps = np.diff(times); print(f"   ⏱️  时间间隔: {gaps.tolist()}")
        if any(g > 10 for g in gaps): print("   ⚠️  存在 >10h 的采样断层")
        self.results['time_dist'] = {'times': times, 'counts': counts, 'gaps': gaps.tolist()}
    def analyze_gene_sparsity(self):
        print("\n" + "="*60); print("🕸️  诊断 2: 关键炎症基因稀疏性"); print("="*60)
        target_genes = ['Il6', 'Tnf', 'Il1b', 'Cxcl10', 'Nfkbia', 'Tnfaip3']
        X = self.dataset.X
        for gene in target_genes:
            idx = self._get_gene_index(gene)
            if idx is not None:
                dropout = np.mean(X[:, idx] == 0) * 100
                status = "🔴 极高稀疏" if dropout > 90 else ("🟡 高稀疏" if dropout > 80 else "🟢 可接受")
                print(f"   {gene}: {dropout:.1f}% 零值 {status}")
            else: print(f"   {gene}: ❌ 未找到")
        self.results['sparsity'] = "已检查"
    def evaluate_recon_gap(self):
        print("\n" + "="*60); print("🔍 诊断 3: 训练集 vs 测试集重建性能对比"); print("="*60)
        all_times = sorted(self.dataset.time_to_cells.keys())
        train_times = [t for t in all_times if t not in self.config.HOLDOUT_TIMEPOINTS]
        test_times = self.config.HOLDOUT_TIMEPOINTS
        def calc_mean_corr(time_list):
            indices = [i for t in time_list for i in self.dataset.get_cells_by_timepoint(t)]
            if EVAL_MAX_CELLS > 0 and len(indices) > EVAL_MAX_CELLS:
                rng = np.random.default_rng(int(os.getenv("LEWORLD_SEED", "42")))
                indices = rng.choice(indices, size=EVAL_MAX_CELLS, replace=False).tolist()
            if not indices: return 0.0
            corrs = []
            self.model.eval()
            with torch.no_grad():
                for idx in indices:
                    x = torch.FloatTensor(self.dataset[idx]['expression']).unsqueeze(0).to(self.device)
                    t = torch.LongTensor([self.dataset[idx]['timepoint']]).to(self.device)
                    out = self.model(x, t); recon = out['x_recon'].cpu().numpy().flatten(); true = x.cpu().numpy().flatten()
                    c, _ = pearsonr(true, recon); corrs.append(c)
            return np.mean(corrs)
        train_corr = calc_mean_corr(train_times); test_corr = calc_mean_corr(test_times)
        np.random.seed(42); rand_corrs = []
        for _ in range(200):
            idx = np.random.choice(len(self.dataset))
            x = torch.FloatTensor(self.dataset[idx]['expression']).unsqueeze(0).to(self.device)
            t = torch.LongTensor([self.dataset[idx]['timepoint']]).to(self.device)
            self.model.eval()
            with torch.no_grad():
                out = self.model(x, t); recon = out['x_recon'].cpu().numpy().flatten()
                rand_idx = np.random.choice(len(self.dataset))
                rand_x = torch.FloatTensor(self.dataset[rand_idx]['expression']).numpy().flatten()
                c, _ = pearsonr(recon, rand_x); rand_corrs.append(c)
        rand_base = np.mean(rand_corrs)
        print(f"   训练集时间点重建相关性: {train_corr:.3f}")
        print(f"   测试集时间点重建相关性: {test_corr:.3f}")
        print(f"   随机基线相关性        : {rand_base:.3f} ± {np.std(rand_corrs):.3f}")
        decay_rate = (train_corr - test_corr) / max(train_corr, 1e-5)
        print(f"   性能衰减率            : {decay_rate*100:.1f}%")
        self.results['recon_gap'] = {'train': train_corr, 'test': test_corr, 'rand': rand_base, 'decay': decay_rate}
        return train_corr, test_corr, rand_base, decay_rate
    def run_diagnosis(self):
        self.analyze_timepoint_distribution(); self.analyze_gene_sparsity()
        train_c, test_c, rand_c, decay = self.evaluate_recon_gap()
        print("\n" + "="*60); print("🎯 综合诊断结论"); print("="*60)
        conclusions = []
        if test_c > rand_c + 0.05: conclusions.append("✅ 模型学到真实生物学信号")
        else: conclusions.append("⚠️ 模型接近随机猜测")
        if decay > 0.5: conclusions.append("📉 训练→测试性能断崖式下降 → 【数据瓶颈主导】")
        elif train_c < 0.4 and test_c < 0.2: conclusions.append("📉 训练/测试均偏低 → 【模型瓶颈主导】")
        else: conclusions.append("📈 表现符合预期单细胞时序建模范式")
        for line in conclusions: print(f"   {line}")
        with open(cfg.RESULT_DIR / tagged_filename('diagnosis_results', 'json', RESULT_TAG), 'w') as f:
            json.dump(convert_to_native(self.results), f, indent=2, cls=NumpyEncoder)
        return self.results

# =============================================================================
# 验证模块 (完整集成)
# =============================================================================
class TripleValidation:
    def __init__(self, model, dataset, gene_names, config, biology_context=None, run_metadata=None, checkpoint=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device); self.dataset = dataset; self.gene_names = gene_names
        self.config = config; self.device = device; self.results = {}
        self.biology_context = biology_context or {}
        self.run_metadata = dict(run_metadata or {})
        if checkpoint is not None:
            self.run_metadata["checkpoint"] = {
                "path": portable_path(checkpoint),
                "sha256": file_sha256(checkpoint),
            }
        self.gene_to_idx = {g.upper(): i for i, g in enumerate(self.gene_names)}
        self._extrapolation_details = {}
    def _predict_mean_to_target(self, source_cells, source_time, target_time):
        x_prev = torch.FloatTensor(self.dataset.X[source_cells]).to(self.device)
        t_prev_tensor = torch.LongTensor([source_time] * len(source_cells)).to(self.device)
        gap = max(0.0, float(target_time - source_time))
        delta = max(float(getattr(self.config, 'ODE_DELTA', 1.0)), 1e-6)
        # Align prediction horizon with real temporal gap instead of fixed one-step rollout.
        predict_steps = max(1, int(np.ceil(gap / delta)))
        out = self.model(x_prev, t_prev_tensor, predict_steps=predict_steps)
        if out['x_predictions']:
            pred = out['x_predictions'][-1].cpu().numpy()
        else:
            pred = out['x_recon'].cpu().numpy()
        return pred.mean(axis=0), predict_steps
    def _extract_extrapolation_details(self):
        all_times = sorted(self.dataset.time_to_cells.keys())
        details = {}
        self.model.eval()
        with torch.no_grad():
            for t in self.config.HOLDOUT_TIMEPOINTS:
                prev_times = [pt for pt in all_times if pt < t and pt in self.config.TRAIN_TIMEPOINTS]
                if not prev_times:
                    continue
                t_prev = max(prev_times)
                prev_cells = self.dataset.get_cells_by_timepoint(t_prev)
                target_cells = self.dataset.get_cells_by_timepoint(t)
                if not prev_cells or not target_cells:
                    continue
                pred_mean, predict_steps = self._predict_mean_to_target(prev_cells, t_prev, t)
                prev_mean = self.dataset.X[prev_cells].mean(axis=0)
                true_mean = self.dataset.X[target_cells].mean(axis=0)
                corr = safe_corr(true_mean, pred_mean)
                details[int(t)] = {
                    'from_time': int(t_prev),
                    'corr': corr if corr is not None else 0.0,
                    'pred_mean': pred_mean,
                    'true_mean': true_mean,
                    'prev_mean': prev_mean,
                    'n_prev': len(prev_cells),
                    'n_target': len(target_cells),
                    'predict_steps': int(predict_steps),
                }
        self._extrapolation_details = details
        return details
    def _pathway_scores_numpy(self, vector):
        out = {}
        for name, idxs in self.biology_context.get('pathways', {}).items():
            out[name] = float(np.mean(vector[idxs])) if idxs else 0.0
        return out
    def _pubmed_url(self, query):
        return f"https://pubmed.ncbi.nlm.nih.gov/?term={quote_plus(query)}"
    def validate_time_holdout(self):
        print("\n" + "="*60); print("📅 验证 1: 时间 Hold-out"); print("="*60)
        test_times = self.config.HOLDOUT_TIMEPOINTS
        test_indices = [i for t in test_times for i in self.dataset.get_cells_by_timepoint(t)]
        if EVAL_MAX_CELLS > 0 and len(test_indices) > EVAL_MAX_CELLS:
            rng = np.random.default_rng(int(os.getenv("LEWORLD_SEED", "42")))
            test_indices = rng.choice(test_indices, size=EVAL_MAX_CELLS, replace=False).tolist()
        if not test_indices: print("⚠️  测试集为空"); return None
        self.model.eval(); corrs = []
        with torch.no_grad():
            for idx in test_indices:
                x = torch.FloatTensor(self.dataset[idx]['expression']).unsqueeze(0).to(self.device)
                t = torch.LongTensor([self.dataset[idx]['timepoint']]).to(self.device)
                out = self.model(x, t); recon = out['x_recon'].cpu().numpy().flatten(); true = x.cpu().numpy().flatten()
                c, _ = pearsonr(true, recon); corrs.append(c)
        mean_corr = np.mean(corrs); std_corr = np.std(corrs)
        print(f"   ✅ Hold-out Corr: {mean_corr:.3f} ± {std_corr:.3f} (n={len(test_indices)})")
        self.results['time_holdout'] = {'mean_correlation': float(mean_corr), 'std': float(std_corr)}
        return self.results['time_holdout']
    def validate_time_extrapolation(self):
        print("\n" + "="*60); print("🧭 验证 2: 真实时间外推 (population-level)"); print("="*60)
        details = self._extract_extrapolation_details()
        extrapolation = {}
        for t, payload in details.items():
            extrapolation[t] = {'from_time': payload['from_time'], 'corr': payload['corr'], 'predict_steps': payload.get('predict_steps', 1)}
            print(
                f"   {payload['from_time']}h -> {t}h: corr={payload['corr']:.3f} "
                f"(steps={payload.get('predict_steps', 1)}, n_prev={payload['n_prev']}, n_target={payload['n_target']})"
            )
        if extrapolation:
            vals = [v['corr'] for v in extrapolation.values() if v['corr'] is not None]
            summary = {'mean_correlation': float(np.mean(vals)), 'std': float(np.std(vals)), 'pairs': extrapolation} if vals else {'mean_correlation': 0.0, 'std': 0.0, 'pairs': extrapolation}
            print(f"   ✅ 外推平均相关性: {summary['mean_correlation']:.3f} ± {summary['std']:.3f}")
            self.results['time_extrapolation'] = summary
            return summary
        print("   ⚠️ 未形成有效外推对")
        return None
    def validate_tf_direction_consistency(self):
        print("\n" + "="*60); print("🧬 验证 3: TF target 方向一致率"); print("="*60)
        details = self._extrapolation_details or self._extract_extrapolation_details()
        tf_map = self.biology_context.get('tf_targets', {})
        if not tf_map or not details:
            print("   ⚠️ TF targets 或外推结果缺失，跳过")
            return None
        per_tf = {}
        for tf_name, idxs in tf_map.items():
            agreements = []
            for payload in details.values():
                pred_delta = payload['pred_mean'][idxs] - payload['prev_mean'][idxs]
                true_delta = payload['true_mean'][idxs] - payload['prev_mean'][idxs]
                valid = np.abs(true_delta) > 1e-6
                if np.any(valid):
                    agree = np.mean(np.sign(pred_delta[valid]) == np.sign(true_delta[valid]))
                    agreements.append(float(agree))
            if agreements:
                per_tf[tf_name] = float(np.mean(agreements))
        overall = float(np.mean(list(per_tf.values()))) if per_tf else None
        if overall is not None:
            print(f"   ✅ TF direction consistency: {overall:.3f}")
        self.results['tf_direction_consistency'] = {'overall': overall, 'per_tf': per_tf}
        return self.results['tf_direction_consistency']
    def validate_pathway_activation_order(self):
        print("\n" + "="*60); print("🧭 验证 4: 通路激活顺序正确率"); print("="*60)
        details = self._extrapolation_details or self._extract_extrapolation_details()
        pathways = self.biology_context.get('pathways', {})
        if len(details) < 2 or not pathways:
            print("   ⚠️ 通路或外推时间点不足，跳过")
            return None
        sorted_targets = sorted(details.keys())
        per_pathway = {}
        for p_name, idxs in pathways.items():
            true_scores = [float(np.mean(details[t]['true_mean'][idxs])) for t in sorted_targets]
            pred_scores = [float(np.mean(details[t]['pred_mean'][idxs])) for t in sorted_targets]
            true_order = np.sign(np.diff(true_scores))
            pred_order = np.sign(np.diff(pred_scores))
            if len(true_order) == 0:
                continue
            acc = float(np.mean(true_order == pred_order))
            per_pathway[p_name] = acc
        overall = float(np.mean(list(per_pathway.values()))) if per_pathway else None
        if overall is not None:
            print(f"   ✅ Pathway activation order accuracy: {overall:.3f}")
        self.results['pathway_activation_order'] = {'overall': overall, 'per_pathway': per_pathway}
        return self.results['pathway_activation_order']
    def validate_marker_rank_spearman(self):
        print("\n" + "="*60); print("📏 验证 5: marker rank Spearman"); print("="*60)
        details = self._extrapolation_details or self._extract_extrapolation_details()
        marker_sets = self.biology_context.get('marker_sets', {})
        if not marker_sets or not details:
            print("   ⚠️ marker 集合或外推结果缺失，跳过")
            return None
        per_set = {}
        for set_name, idxs in marker_sets.items():
            scores = []
            for payload in details.values():
                pred_delta = payload['pred_mean'][idxs] - payload['prev_mean'][idxs]
                true_delta = payload['true_mean'][idxs] - payload['prev_mean'][idxs]
                if len(idxs) >= 2 and np.std(pred_delta) > 1e-12 and np.std(true_delta) > 1e-12:
                    corr, _ = spearmanr(pred_delta, true_delta)
                    if not np.isnan(corr):
                        scores.append(float(corr))
            if scores:
                per_set[set_name] = float(np.mean(scores))
        overall = float(np.mean(list(per_set.values()))) if per_set else None
        if overall is not None:
            print(f"   ✅ Marker rank Spearman: {overall:.3f}")
        self.results['marker_rank_spearman'] = {'overall': overall, 'per_set': per_set}
        return self.results['marker_rank_spearman']
    def validate_literature(self):
        print("\n" + "="*60); print("📚 验证 6: 文献佐证"); print("="*60)
        available_genes = []
        gene_name_map = self.gene_to_idx
        for gene in self.config.LPS_GENES:
            if gene.upper() in gene_name_map:
                available_genes.append(self.gene_names[gene_name_map[gene.upper()]])
        print(f"   ✅ 可用 LPS 基因: {available_genes}")
        if not available_genes: return None
        timepoints = sorted(self.dataset.time_to_cells.keys())
        gene_timecourse = {}
        for gene in available_genes:
            idx = gene_name_map[gene.upper()]
            gene_timecourse[gene] = [np.mean(self.dataset.X[self.dataset.get_cells_by_timepoint(t), idx]) for t in timepoints]
        for gene, values in gene_timecourse.items():
            trend = "↑" if values[-1] > values[0] else ("↓" if values[-1] < values[0] else "→")
            print(f"   {gene}: {values[0]:.3f} → {values[-1]:.3f} {trend}")
        context = self.biology_context.get('literature_terms', {}).get('default_context', "dendritic cell LPS time series")
        links = {gene: self._pubmed_url(f"{gene} {context}") for gene in available_genes}
        self.results['literature'] = {'genes': available_genes, 'timecourse': gene_timecourse, 'pubmed_links': links}
        return self.results['literature']
    def export_biological_conclusions(self, result_suffix):
        details = self._extrapolation_details or self._extract_extrapolation_details()
        if not details:
            return None
        top_k = self.config.TOP_K_BIO_GENES
        pathway_map = self.biology_context.get('pathways', {})
        gene_records = []
        pathway_records = []
        conclusion_payload = {'pairs': {}, 'pubmed_links': {}}
        for t, payload in details.items():
            pred_delta = payload['pred_mean'] - payload['prev_mean']
            true_delta = payload['true_mean'] - payload['prev_mean']
            top_up_idx = np.argsort(pred_delta)[-top_k:][::-1]
            top_down_idx = np.argsort(pred_delta)[:top_k]
            top_up = [self.gene_names[i] for i in top_up_idx]
            top_down = [self.gene_names[i] for i in top_down_idx]
            conclusion_payload['pairs'][str(t)] = {
                'from_time': payload['from_time'],
                'top_up_genes': top_up,
                'top_down_genes': top_down
            }
            for idx in list(top_up_idx) + list(top_down_idx):
                gene_records.append({
                    'target_time': t,
                    'from_time': payload['from_time'],
                    'gene': self.gene_names[idx],
                    'pred_delta': float(pred_delta[idx]),
                    'true_delta': float(true_delta[idx]),
                    'direction_match': bool(np.sign(pred_delta[idx]) == np.sign(true_delta[idx]))
                })
            selected = set(top_up) | set(top_down)
            M = len(self.gene_names)
            n = len(selected)
            for pathway_name, idxs in pathway_map.items():
                pathway_genes = {self.gene_names[i] for i in idxs}
                K = len(pathway_genes)
                k = len(pathway_genes & selected)
                if K == 0 or n == 0:
                    continue
                p_value = float(hypergeom.sf(k - 1, M, K, n))
                term = self.biology_context.get('literature_terms', {}).get('pathway_terms', {}).get(pathway_name, pathway_name)
                conclusion_payload['pubmed_links'][pathway_name] = self._pubmed_url(f"{term} dendritic cell LPS")
                pathway_records.append({
                    'target_time': t,
                    'from_time': payload['from_time'],
                    'pathway': pathway_name,
                    'overlap': int(k),
                    'pathway_size': int(K),
                    'selected_size': int(n),
                    'p_value': p_value
                })
        genes_df = pd.DataFrame(gene_records)
        pathways_df = pd.DataFrame(pathway_records)
        top_genes_path = cfg.RESULT_DIR / f'top_genes_{result_suffix}.csv'
        enrich_path = cfg.RESULT_DIR / f'pathway_enrichment_{result_suffix}.json'
        concl_path = cfg.RESULT_DIR / f'biological_conclusions_{result_suffix}.json'
        genes_df.to_csv(top_genes_path, index=False)
        with open(enrich_path, 'w') as f:
            json.dump(convert_to_native(pathway_records), f, indent=2, cls=NumpyEncoder)
        with open(concl_path, 'w') as f:
            json.dump(convert_to_native(conclusion_payload), f, indent=2, cls=NumpyEncoder)
        print(f"   ✅ biological conclusions exported: {concl_path.name}, {top_genes_path.name}, {enrich_path.name}")
        self.results['biological_conclusions'] = {
            'top_genes_csv': top_genes_path.name,
            'pathway_enrichment_json': enrich_path.name,
            'conclusions_json': concl_path.name
        }
        return self.results['biological_conclusions']
    def run_all(self, result_suffix_name):
        print("\n" + "="*60); print("🎯 运行验证"); print("="*60)
        self.results['run_metadata'] = self.run_metadata
        self.results['biology_prior_source'] = self.biology_context.get('source', {})
        self.validate_time_holdout()
        self.validate_time_extrapolation()
        self.validate_tf_direction_consistency()
        self.validate_pathway_activation_order()
        self.validate_marker_rank_spearman()
        self.validate_literature()
        self.export_biological_conclusions(result_suffix_name)
        payload = convert_to_native(self.results)
        with open(cfg.RESULT_DIR / f'validation_results_{result_suffix_name}.json', 'w') as f:
            json.dump(payload, f, indent=2, cls=NumpyEncoder)
        with open(cfg.RESULT_DIR / f'validation_results_{result_suffix_name}_seed{RUN_SEED}.json', 'w') as f:
            json.dump(payload, f, indent=2, cls=NumpyEncoder)
        return self.results

# =============================================================================
# 主流程
# =============================================================================
def main():
    print("="*70); print("🔬 LEWorldModel GRU vs ODE Benchmark v2.0"); print("="*70)
    print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 加载数据
    expression_matrix = np.load(cfg.DATA_DIR / 'expression_matrix.npy')
    metadata = pd.read_csv(cfg.DATA_DIR / 'metadata.csv', index_col=0)
    with open(cfg.DATA_DIR / 'gene_names.txt', 'r') as f:
        gene_names = [line.strip() for line in f]
    priors = load_biology_priors(cfg.BIOLOGY_PRIORS_PATH)
    priors['source'] = {'base_json': portable_path(cfg.BIOLOGY_PRIORS_PATH)}
    ext_pathways = load_gene_sets_from_gmt(cfg.EXTERNAL_PATHWAY_GMT)
    ext_tf_targets = load_tf_targets_from_csv(cfg.EXTERNAL_TF_TARGET_CSV)
    priors = merge_priors(priors, extra_pathways=ext_pathways, extra_tf_targets=ext_tf_targets)
    priors['source']['external_gmt'] = portable_path(cfg.EXTERNAL_PATHWAY_GMT) if ext_pathways else None
    priors['source']['external_tf_csv'] = portable_path(cfg.EXTERNAL_TF_TARGET_CSV) if ext_tf_targets else None
    print("🧬 matching biology priors...")
    biology_context = match_gene_sets(gene_names, priors, min_genes=cfg.PATHWAY_MIN_GENES)
    biology_context['source'] = priors.get('source', {})
    if MAX_CELLS > 0 and expression_matrix.shape[0] > MAX_CELLS:
        rng = np.random.default_rng(int(os.getenv("LEWORLD_SEED", "42")))
        chosen = np.sort(rng.choice(expression_matrix.shape[0], size=MAX_CELLS, replace=False))
        expression_matrix = expression_matrix[chosen]
        metadata = metadata.iloc[chosen].copy()
        print(f"⚡ 使用快速子集: {expression_matrix.shape[0]} cells")
    cfg.INPUT_DIM = expression_matrix.shape[1]
    dataset = SCTimeSeriesDataset(expression_matrix, metadata, gene_names)
    
    # 2. 固定数据划分
    all_times = sorted(dataset.time_to_cells.keys())
    train_times = [t for t in all_times if t in cfg.TRAIN_TIMEPOINTS]
    print("\n📋 Training cell selection:")
    eligible_train_idx = build_train_indices(dataset, train_times)
    train_idx, val_idx = split_internal_validation(
        eligible_train_idx,
        dataset.timepoints,
        cfg.INTERNAL_VAL_FRACTION,
        RUN_SEED,
    )
    holdout_idx = [i for t in cfg.HOLDOUT_TIMEPOINTS for i in dataset.get_cells_by_timepoint(t)]
    if set(train_idx) & set(holdout_idx) or set(val_idx) & set(holdout_idx):
        raise RuntimeError("hold-out cells leaked into training or checkpoint selection")
    if not train_idx or not val_idx:
        raise RuntimeError("internal training/validation split is empty")
    print(f"   internal train cells: {len(train_idx)}")
    print(f"   internal validation cells: {len(val_idx)}")
    print(f"   final-only holdout cells: {len(holdout_idx)}")
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=cfg.BATCH_SIZE, shuffle=False)
    train_time_means = {}
    train_idx_set = set(train_idx)
    for t in train_times:
        idxs = [i for i in dataset.get_cells_by_timepoint(t) if i in train_idx_set]
        if idxs:
            train_time_means[int(t)] = dataset.X[idxs].mean(axis=0).astype(np.float32)
    run_metadata = build_run_metadata(dataset, gene_names, train_times, train_idx, val_idx)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 3. 训练 & 验证: GRU
    print("\n🟦 训练 GRU 基线...")
    model_gru = LEWorldModel_GRU(cfg)
    trainer_gru = Trainer(
        model_gru,
        cfg,
        biology_context=biology_context,
        train_time_means=train_time_means,
        train_times=train_times,
        run_metadata=run_metadata,
        device=device
    )
    trainer_gru.train(train_loader, val_loader, model_name='gru')
    
    # 4. 训练 & 验证: ODE
    print("\n🟩 训练 ODE 版...")
    model_ode = LEWorldModel_ODE(cfg)
    trainer_ode = Trainer_ODE(
        model_ode,
        cfg,
        biology_context=biology_context,
        train_time_means=train_time_means,
        train_times=train_times,
        run_metadata=run_metadata,
        device=device
    )
    trainer_ode.train(train_loader, val_loader, model_name='ode')
    
    # 5. 加载最佳模型
    print("\n📥 加载最佳模型...")
    ckpt_gru = torch.load(
        checkpoint_path('gru'), map_location=device, weights_only=True
    )
    model_gru.load_state_dict(ckpt_gru['model_state_dict'])
    ckpt_ode = torch.load(
        checkpoint_path('ode'), map_location=device, weights_only=True
    )
    model_ode.load_state_dict(ckpt_ode['model_state_dict'])
    
    # 6. 运行诊断 (ODE)
    print("\n🔍 运行诊断...")
    diag = DiagnosticAnalyzer(model_ode, dataset, gene_names, cfg, device=device)
    diag_payload = diag.run_diagnosis()
    with open(cfg.RESULT_DIR / tagged_filename('diagnosis_results', 'json', RESULT_TAG, f'seed{RUN_SEED}'), 'w') as f:
        json.dump(convert_to_native(diag_payload), f, indent=2, cls=NumpyEncoder)
    
    # 7. 运行验证 (两者)
    print("\n🎯 运行验证...")
    val_ode = TripleValidation(model_ode, dataset, gene_names, cfg, biology_context=biology_context, run_metadata=run_metadata, checkpoint=checkpoint_path('ode'), device=device)
    val_ode.run_all(result_suffix('ode'))
    val_gru = TripleValidation(model_gru, dataset, gene_names, cfg, biology_context=biology_context, run_metadata=run_metadata, checkpoint=checkpoint_path('gru'), device=device)
    val_gru.run_all(result_suffix('gru'))
    
    # 8. 🆕 生成 Figure 3: Gap 衰减对比图
    plot_gap_decay_comparison(model_gru, model_ode, dataset, gene_names, cfg, device, cfg.RESULT_DIR)
    
    # 9. 🆕 生成 Figure 4: 关键基因轨迹
    plot_gene_trajectory(model_ode, dataset, gene_names, cfg, device, cfg.RESULT_DIR, gene_list=cfg.LPS_GENES)
    
    # 10. 生成综合报告
    gap_fig = tagged_filename('gap_decay_comparison', 'png', RESULT_TAG)
    traj_glob = tagged_filename('*_trajectory', 'png', RESULT_TAG)
    report = f"""
# LEWorldModel Benchmark Report (v2.0)
## Training Results
- GRU Best Val Loss: {trainer_gru.best_val_loss:.4f}
- ODE Best Val Loss: {trainer_ode.best_val_loss:.4f}
## Validation (Hold-out 36/48h)
- GRU Correlation: {val_gru.results.get('time_holdout', {}).get('mean_correlation', 'N/A'):.3f}
- ODE Correlation: {val_ode.results.get('time_holdout', {}).get('mean_correlation', 'N/A'):.3f}
## Mechanistic Consistency
- GRU TF direction consistency: {val_gru.results.get('tf_direction_consistency', {}).get('overall', 'N/A')}
- ODE TF direction consistency: {val_ode.results.get('tf_direction_consistency', {}).get('overall', 'N/A')}
- GRU pathway order accuracy: {val_gru.results.get('pathway_activation_order', {}).get('overall', 'N/A')}
- ODE pathway order accuracy: {val_ode.results.get('pathway_activation_order', {}).get('overall', 'N/A')}
## Gap Decay Analysis
- See: {gap_fig}
## Gene Trajectories
- See: {traj_glob} for {cfg.LPS_GENES}
"""
    with open(cfg.RESULT_DIR / tagged_filename('benchmark_report', 'md', RESULT_TAG), 'w') as f:
        f.write(report)
    with open(cfg.RESULT_DIR / tagged_filename('benchmark_report', 'md', RESULT_TAG, f'seed{RUN_SEED}'), 'w') as f:
        f.write(report)
    print(report)
    
    print("\n" + "="*70)
    print("✅ 全部完成！论文图表已生成:")
    print(f"   📊 Figure 3: {cfg.RESULT_DIR / gap_fig}")
    print(f"   🧬 Figure 4: {cfg.RESULT_DIR / traj_glob} (gene trajectories)")
    print(f"   📁 结果目录: {cfg.RESULT_DIR.absolute()}")
    print("="*70)

if __name__ == "__main__":
    main()
