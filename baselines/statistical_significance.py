#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
LEWorldModel 统计显著性检验模块 (publication bundle 版)
功能：训练基线 → LEWorld推理 → 细胞级Pearson计算 → 配对t检验/Wilcoxon → 可视化
依赖：需先运行 ode_benchmark/train_ode_benchmark.py 生成 GRU checkpoint
=============================================================================
"""
import os, sys, json, time, warnings, math
from pathlib import Path  # ✅ 关键修复：导入 Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_rel, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings('ignore')
np.random.seed(int(os.getenv("LEWORLD_SEED", "42")))
torch.manual_seed(int(os.getenv("LEWORLD_SEED", "42")))

# ================= 路径配置 =================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "leworld_input"
MODEL_DIR = BASE_DIR / "artifacts" / "models" / "ode_benchmark"
RESULT_DIR = BASE_DIR / "artifacts" / "results" / "baselines"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_TIMES = [0, 1, 4, 16, 27]
TEST_TIMES = [36, 48]
N_GENES_MAX = int(os.getenv("LEWORLD_N_GENES", "0"))

# ================= 从 bundle 内 ODE benchmark 导入模型 =================
import importlib.util
train_script = BASE_DIR / "ode_benchmark" / "train_ode_benchmark.py"
spec = importlib.util.spec_from_file_location("ode_bundle_module", train_script)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
LEWorldModel = train_module.LEWorldModel_GRU
Config = train_module.Config
print(f"✅ 从 {train_script.name} 导入 LEWorldModel_GRU")

# ================= 数据加载 =================
def load_and_split():
    """加载数据并按时间划分"""
    X = np.load(DATA_DIR / 'expression_matrix.npy')
    meta = pd.read_csv(DATA_DIR / 'metadata.csv', index_col=0)
    times = meta['timepoint_numeric'].values
    tr_mask = np.isin(times, TRAIN_TIMES)
    te_mask = np.isin(times, TEST_TIMES)
    return X[tr_mask], times[tr_mask], X[te_mask], times[te_mask]

def make_time_features(t):
    """时间特征工程"""
    t = np.asarray(t).reshape(-1, 1)
    return np.hstack([t, t**2, np.log1p(t), np.sin(2 * np.pi * t / 48)])

# ================= 基线训练 =================
def train_baseline(model_cls, X_tr, t_tr, X_te, t_te, name, **kwargs):
    """按基因训练经典机器学习模型"""
    feats_tr, feats_te = make_time_features(t_tr), make_time_features(t_te)
    preds = np.zeros_like(X_te)
    n_genes = X_tr.shape[1] if N_GENES_MAX <= 0 else min(X_tr.shape[1], N_GENES_MAX)
    for g in tqdm(range(n_genes), desc=f"  {name}"):
        m = model_cls(**kwargs).fit(feats_tr, X_tr[:, g])
        preds[:, g] = m.predict(feats_te)
    return preds

# ================= LEWorld 推理 =================
def run_leworld_inference(X_te, t_te, device='cpu'):
    """使用训练好的 LEWorldModel 进行推理"""
    model_path = MODEL_DIR / 'best_gru.pt'
    if not model_path.exists():
        print(f"⚠️  未找到模型文件：{model_path}")
        return None
    
    try:
        # 创建模型实例
        cfg = Config()
        cfg.INPUT_DIM = X_te.shape[1]
        model = LEWorldModel(cfg).to(device)
        
        # 加载权重
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        print(f"✅ 模型加载成功 (epoch: {ckpt['epoch']}, val_loss: {ckpt['val_loss']:.4f})")
        
        # 批量推理
        preds = np.zeros_like(X_te)
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(X_te), batch_size):
                xb = torch.FloatTensor(X_te[i:i+batch_size]).to(device)
                tb = torch.LongTensor(t_te[i:i+batch_size]).to(device)
                out = model(xb, tb)
                # 兼容不同输出格式
                if isinstance(out, dict):
                    preds[i:i+batch_size] = out['x_recon'].cpu().numpy()
                else:
                    preds[i:i+batch_size] = out.cpu().numpy()
        return preds
    except Exception as e:
        print(f"❌ LEWorld 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# ================= 核心计算 =================
def compute_cellwise_corr(true, pred):
    """计算每个细胞的 Pearson 相关性 (返回长度为 N_cells 的数组)"""
    corrs = []
    for i in range(len(true)):
        try:
            c, _ = pearsonr(true[i], pred[i])
            corrs.append(c)
        except:
            corrs.append(np.nan)
    return np.array(corrs)

def run_stat_tests(corrs_dict):
    """执行配对统计检验"""
    leworld = corrs_dict['LEWorldModel']
    results = {}
    for name, arr in corrs_dict.items():
        if name == 'LEWorldModel': continue
        # 过滤 NaN
        valid = ~(np.isnan(leworld) | np.isnan(arr))
        if valid.sum() < 10:
            continue
        # 配对 t 检验 (单侧: LEWorld > Baseline)
        t_stat, p_t = ttest_rel(leworld[valid], arr[valid], alternative='greater')
        # Wilcoxon 符号秩检验 (非参数)
        _, p_w = wilcoxon(leworld[valid], arr[valid], alternative='greater')
        # Cohen's d 效应量
        diff = leworld[valid] - arr[valid]
        cohen_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)
        results[name] = {
            'p_ttest': float(p_t), 
            'p_wilcoxon': float(p_w), 
            'cohens_d': float(cohen_d), 
            'n_valid': int(valid.sum()),
            'mean_leworld': float(np.mean(leworld[valid])),
            'mean_baseline': float(np.mean(arr[valid]))
        }
    return results

# ================= 可视化 =================
def plot_significance(corrs_dict, save_path):
    """绘制带显著性标记的箱线图 (修复 matplotlib 兼容性)"""
    names = list(corrs_dict.keys())
    data = [corrs_dict[n] for n in names]
    
    plt.figure(figsize=(9, 5))
    # ✅ 修复：使用 flierprops 替代 fliersize
    bp = plt.boxplot(data, labels=[n.replace('_', '\n') for n in names], 
                     patch_artist=True, showfliers=True,
                     flierprops={'markersize': 3, 'alpha': 0.5})
    colors = ['#E0E0E0'] * (len(names)-1) + ['#2E86AB']  # 突出 LEWorld
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # 添加散点 (Jitter)
    for i, arr in enumerate(data):
        y = arr[~np.isnan(arr)]
        x = np.random.normal(i+1, 0.04, size=len(y))
        plt.scatter(x, y, alpha=0.3, s=10, color='black')
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5, label='Random baseline')
    plt.ylabel('Cell-level Pearson Correlation')
    plt.title('Paired Statistical Test: LEWorldModel vs Baselines')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 显著性图已保存: {save_path}")

# ================= 主流程 =================
def main():
    print("="*70)
    print("📊 运行配对统计检验 (完整修复版)")
    print("="*70)
    
    # 1. 加载数据
    X_tr, t_tr, X_te, t_te = load_and_split()
    print(f"✅ 数据加载完成 | 训练: {X_tr.shape[0]} | 测试: {X_te.shape[0]}")
    
    # 2. 训练基线模型
    print("\n🧪 训练经典机器学习基线...")
    models = {
        'Naive_LastTimeMean': np.tile(
            X_tr[t_tr == max(TRAIN_TIMES)].mean(axis=0), 
            (len(X_te), 1)
        ),
        'Ridge_Poly': train_baseline(
            Ridge, X_tr, t_tr, X_te, t_te, 'Ridge', alpha=1.0
        ),
        'HGB_Regressor': train_baseline(
            HistGradientBoostingRegressor, X_tr, t_tr, X_te, t_te, 'HGB',
            max_depth=3, min_samples_leaf=50, learning_rate=0.05, max_iter=50
        ),
        'MLP_Sklearn': train_baseline(
            MLPRegressor, X_tr, t_tr, X_te, t_te, 'MLP',
            hidden_layer_sizes=(64, 32), max_iter=200, random_state=42,
            activation='relu', learning_rate='adaptive'
        )
    }
    
    # 3. LEWorld 推理
    print("\n🤖 运行 LEWorldModel 推理...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lw_pred = run_leworld_inference(X_te, t_te, device=device)
    if lw_pred is not None:
        models['LEWorldModel'] = lw_pred
    else:
        print("⚠️  跳过 LEWorld 统计检验")
        return
    
    # 4. 计算细胞级相关性
    print("\n🔍 计算细胞级 Pearson 相关性...")
    corrs_dict = {}
    for name, pred in models.items():
        corrs_dict[name] = compute_cellwise_corr(X_te, pred)
        mean_c = np.nanmean(corrs_dict[name])
        print(f"   {name:25s}: {mean_c:.3f}")
    
    # 5. 统计检验
    print("\n📈 执行配对检验 (H₁: LEWorld > Baseline)...")
    stats = run_stat_tests(corrs_dict)
    
    # 打印结果表
    print("\n" + "="*80)
    print("📋 统计检验结果")
    print("="*80)
    header = f"{'Method':<25} | {'Mean(LEW)':<10} | {'Mean(BL)':<10} | {'p(t)':<10} | {'p(Wil)':<10} | {'d':<6} | {'Sig'}"
    print(header)
    print("-"*80)
    for name, res in stats.items():
        sig = "✅" if res['p_ttest'] < 0.05 else "❌"
        row = f"{name:<25} | {res['mean_leworld']:.3f} | {res['mean_baseline']:.3f} | {res['p_ttest']:.2e} | {res['p_wilcoxon']:.2e} | {res['cohens_d']:.2f} | {sig}"
        print(row)
    
    # 6. 可视化
    plot_significance(corrs_dict, RESULT_DIR / 'significance_test.png')
    
    # 7. 保存结果
    output = {
        'correlations': {k: float(np.nanmean(v)) for k, v in corrs_dict.items()},
        'statistics': stats,
        'config': {
            'train_times': TRAIN_TIMES,
            'test_times': TEST_TIMES,
            'n_test_cells': len(X_te)
        }
    }
    with open(RESULT_DIR / 'statistical_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 结果已保存至: {RESULT_DIR / 'statistical_results.json'}")
    print("🚀 完成！")

if __name__ == "__main__":
    main()