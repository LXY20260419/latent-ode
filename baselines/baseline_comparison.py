#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
LEWorldModel 基线对比脚本
目的：验证深度学习架构 vs 经典机器学习在单细胞时序外推任务上的表现
基线：1. Naive Time Mean  2. Ridge + 多项式时间特征  3. HistGradientBoosting  4. MLP (sklearn)
=============================================================================
"""
import os, json, time, warnings, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

warnings.filterwarnings('ignore')
np.random.seed(int(os.getenv("LEWORLD_SEED", "42")))
BASE_DIR = Path(__file__).resolve().parent.parent
RUN_SEED = int(os.getenv("LEWORLD_SEED", "42"))
RESULT_TAG = os.getenv("LEWORLD_RESULT_TAG", "").strip()


def _parse_int_list(env_name: str, default: list[int]) -> list[int]:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


class BaselineConfig:
    DATA_DIR = Path(os.getenv("LEWORLD_DATA_DIR", str(BASE_DIR / "data" / "leworld_input")))
    RESULT_DIR = BASE_DIR / "artifacts" / "results" / "baselines"
    TRAIN_TIMES = _parse_int_list("LEWORLD_TRAIN_TIMEPOINTS", [0, 1, 4, 16, 27])
    TEST_TIMES = _parse_int_list("LEWORLD_HOLDOUT_TIMEPOINTS", [36, 48])
    MAX_TIME = float(os.getenv("LEWORLD_MAX_TIME", "48.0"))
    N_GENES = int(os.getenv("LEWORLD_N_GENES", "2000"))
    LEWORLD_RESULT_JSON = BASE_DIR / "artifacts" / "results" / "ode_benchmark" / "validation_results_ode.json"

    @staticmethod
    def make_time_features(t):
        """时间特征工程：线性、二次、对数、周期"""
        max_time = cfg.MAX_TIME
        t = np.asarray(t).reshape(-1, 1)
        return np.hstack([
            t,
            t**2,
            np.log1p(t),
            np.sin(2 * np.pi * t / max_time)
        ])

cfg = BaselineConfig()
cfg.RESULT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 数据加载与划分
# =============================================================================
def load_and_split_data():
    print("📂 加载数据...")
    X = np.load(cfg.DATA_DIR / 'expression_matrix.npy')
    meta = pd.read_csv(cfg.DATA_DIR / 'metadata.csv', index_col=0)
    times = meta['timepoint_numeric'].values
    
    # 划分
    train_mask = np.isin(times, cfg.TRAIN_TIMES)
    test_mask = np.isin(times, cfg.TEST_TIMES)
    
    X_train, t_train = X[train_mask], times[train_mask]
    X_test, t_test = X[test_mask], times[test_mask]
    
    print(f"   训练集: {X_train.shape[0]} 细胞, 时间 {np.unique(t_train)}")
    print(f"   测试集: {X_test.shape[0]} 细胞, 时间 {np.unique(t_test)}")
    return X_train, t_train, X_test, t_test

# =============================================================================
# 基线模型定义
# =============================================================================
class BaselineRunner:
    def __init__(self, X_train, t_train, X_test, t_test):
        self.X_train = X_train
        self.t_train = t_train
        self.X_test = X_test
        self.t_test = t_test
        self.features_train = cfg.make_time_features(t_train)
        self.features_test = cfg.make_time_features(t_test)
        self.results = {}
        
    def train_and_predict(self, model_class, name, **kwargs):
        """按基因训练并预测"""
        print(f"\n🧪 训练基线: {name}")
        preds = np.zeros_like(self.X_test)
        start = time.time()

        n_genes = min(cfg.N_GENES, self.X_train.shape[1], self.X_test.shape[1])
        for g in tqdm(range(n_genes), desc=f"  {name}"):
            y_train = self.X_train[:, g]
            model = model_class(**kwargs)
            model.fit(self.features_train, y_train)
            preds[:, g] = model.predict(self.features_test)
            
        self.results[name] = {
            'predictions': preds,
            'train_time': time.time() - start
        }
        return preds

    def evaluate(self):
        """计算细胞级 Pearson 相关性（与 LEWorldModel 完全一致）"""
        print("\n📊 评估基线...")
        metrics = {}
        for name, data in self.results.items():
            corrs = []
            for i in range(len(self.X_test)):
                c, _ = pearsonr(self.X_test[i], data['predictions'][i])
                corrs.append(c)
            corrs = np.array(corrs)
            # 过滤 NaN (方差为0的细胞)
            valid_corrs = corrs[~np.isnan(corrs)]
            metrics[name] = {
                'mean_corr': float(np.mean(valid_corrs)),
                'std_corr': float(np.std(valid_corrs)),
                'train_time': data['train_time']
            }
            print(f"   {name:25s} | Mean Corr: {metrics[name]['mean_corr']:.3f} ± {metrics[name]['std_corr']:.3f} | Time: {metrics[name]['train_time']:.2f}s")
        return metrics

# =============================================================================
# 主流程
# =================================================================================
def main():
    print("="*70)
    print("⚖️ LEWorldModel 基线对比实验")
    print("="*70)
    
    # 1. 数据
    X_train, t_train, X_test, t_test = load_and_split_data()
    runner = BaselineRunner(X_train, t_train, X_test, t_test)
    
    # 2. 运行基线
    # 基线1: Naive (最后训练时间点均值)
    print("\n⏳ 生成 Naive 基线...")
    last_train_idx = np.isin(t_train, max(cfg.TRAIN_TIMES))
    naive_pred = np.tile(X_train[last_train_idx].mean(axis=0), (len(X_test), 1))
    runner.results['Naive_LastTimeMean'] = {'predictions': naive_pred, 'train_time': 0.0}
    
    # 基线2: Ridge + 多项式
    runner.train_and_predict(Ridge, 'Ridge_Poly', alpha=1.0, solver='auto')
    
    # 基线3: HistGradientBoosting (树模型，处理非线性)
    runner.train_and_predict(
        HistGradientBoostingRegressor, 'HGB_Regressor',
        max_depth=3, min_samples_leaf=50, learning_rate=0.05, max_iter=50
    )
    
    # 基线4: MLP (轻量级传统神经网络)
    runner.train_and_predict(
        MLPRegressor, 'MLP_Sklearn',
        hidden_layer_sizes=(64, 32), max_iter=200, random_state=42,
        activation='relu', learning_rate='adaptive'
    )
    
    # 3. 评估
    metrics = runner.evaluate()
    
    # 4. 读取 LEWorldModel 结果进行对比
    leworld_corr = None
    if cfg.LEWORLD_RESULT_JSON.exists():
        try:
            with open(cfg.LEWORLD_RESULT_JSON, 'r') as f:
                lv = json.load(f)
            leworld_corr = lv.get('time_extrapolation', {}).get('mean_correlation')
            if leworld_corr is None:
                leworld_corr = lv.get('time_holdout', {}).get('mean_correlation')
            print(f"\n📥 加载 LEWorldModel 验证结果: {leworld_corr:.3f}")
        except Exception as e:
            print(f"⚠️ 读取 LEWorld 结果失败: {e}")
            
    # 5. 生成对比报告
    print("\n" + "="*70)
    print("📋 综合对比报告")
    print("="*70)
    table = []
    for name, m in metrics.items():
        row = f"{name:25s} | {m['mean_corr']:.3f} ± {m['std_corr']:.3f} | {m['train_time']:.1f}s"
        table.append(row)
        print(f"   {row}")
    if leworld_corr is not None:
        print(f"   {'LEWorldModel (Ours)':25s} | {leworld_corr:.3f} (从 validation_results.json) | N/A")
        
    # 保存 JSON
    tag = f"_{RESULT_TAG}" if RESULT_TAG else ""
    report = {
        'baselines': metrics,
        'leworldmodel_mean_corr': leworld_corr,
        'config': {
            'train_times': cfg.TRAIN_TIMES,
            'test_times': cfg.TEST_TIMES,
            'n_genes': cfg.N_GENES,
            'max_time': cfg.MAX_TIME,
            'feature_engineering': f'time, time^2, log1p(time), sin(2πt/{cfg.MAX_TIME})'
        }
    }
    with open(cfg.RESULT_DIR / f'baseline_comparison{tag}.json', 'w') as f:
        json.dump(report, f, indent=2)
    with open(cfg.RESULT_DIR / f'baseline_comparison{tag}_seed{RUN_SEED}.json', 'w') as f:
        json.dump(report, f, indent=2)
        
    # 6. 可视化
    plot_comparison(metrics, leworld_corr)
    print(f"\n💾 报告已保存至: {cfg.RESULT_DIR / 'baseline_comparison.json'}")
    print(f"📊 可视化已保存至: {cfg.RESULT_DIR / 'baseline_comparison.png'}")

def plot_comparison(metrics, leworld_corr=None):
    tag = f"_{RESULT_TAG}" if RESULT_TAG else ""
    names = list(metrics.keys())
    means = [metrics[n]['mean_corr'] for n in names]
    stds = [metrics[n]['std_corr'] for n in names]
    
    if leworld_corr is not None:
        names.append('LEWorldModel')
        means.append(leworld_corr)
        stds.append(0.0)
        
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, means, yerr=stds, capsize=5, color='skyblue', edgecolor='black', alpha=0.8)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.ylabel('Mean Pearson Correlation (Cell-level)')
    plt.title('Baseline vs LEWorldModel: Time Hold-out Performance')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(cfg.RESULT_DIR / f'baseline_comparison{tag}.png', dpi=300)
    plt.savefig(cfg.RESULT_DIR / f'baseline_comparison{tag}_seed{RUN_SEED}.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()