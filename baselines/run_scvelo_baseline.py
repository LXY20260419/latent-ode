#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scVelo RNA Velocity 基线对比模块 (修复版 + 优雅降级)
说明：10x Genomics V2 3' 数据缺乏内含子 reads，无法生成 spliced/unspliced 层。
      本脚本会自动检测并跳过，输出结构化 N/A 结果，符合顶刊报告规范。
"""
import os, json, numpy as np
import scanpy as sc
import scvelo as scv
import warnings
from pathlib import Path
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULT_DIR = BASE_DIR / "artifacts" / "results" / "baselines"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TEST_TIMES = [36, 48]

def run_scvelo_baseline():
    print("="*60)
    print("🧬 scVelo RNA Velocity 基线")
    print("="*60)
    
    h5ad_path = PROCESSED_DIR / 'processed_dc_lps.h5ad'
    if not h5ad_path.exists():
        result = {
            'method': 'scVelo_Velocity',
            'status': 'skipped',
            'reason': f'missing file: {h5ad_path}',
            'mean_corr': None,
            'std_corr': None,
            'n_cells': 0,
            'note': 'No processed h5ad available in publication bundle.'
        }
        return result
        
    adata = sc.read_h5ad(h5ad_path)
    print(f"✅ 加载成功: {adata.shape} | 已有层: {list(adata.layers.keys())}")
    
    # 检查是否具备运行 scvelo 的先决条件
    has_spliced = 'spliced' in adata.layers
    has_unspliced = 'unspliced' in adata.layers
    has_moments = 'Ms' in adata.layers or 'Mu' in adata.layers
    
    if not (has_spliced or has_unspliced or has_moments):
        print("\n⚠️  跳过 scVelo 基线：数据模态不兼容")
        print("📌 原因：10x V2 3' 端测序不包含内含子 reads，无法分离 spliced/unspliced 计数。")
        print("📌 结论：RNA Velocity 不适用此类数据，采用替代外推基线（见 baseline_comparison.py）")
        
        result = {
            'method': 'scVelo_Velocity',
            'status': 'skipped',
            'reason': '10x V2 data lacks intronic reads (no spliced/unspliced layers)',
            'mean_corr': None,
            'std_corr': None,
            'n_cells': 0,
            'note': 'Incompatible with 10x 3\' chemistry; excluded from correlation comparison.'
        }
        return result
        
    # 若未来使用 Smart-seq2 或 velocyto 处理的数据，可正常运行
    try:
        print("🔄 运行 scVelo (stochastic 模式)...")
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        scv.tl.velocity(adata, mode='stochastic')
        scv.tl.velocity_graph(adata)
        
        test_mask = adata.obs['timepoint_numeric'].isin(TEST_TIMES)
        test_idx = np.where(test_mask)[0]
        X_test = np.array(adata.X[test_idx])
        velocity = adata.layers.get('velocity', adata.obsm.get('velocity'))
        pred = X_test + velocity[test_idx]
        
        corrs = [pearsonr(X_test[i], pred[i])[0] for i in range(len(X_test))]
        valid = [c for c in corrs if not np.isnan(c)]
        
        return {
            'method': 'scVelo_Velocity',
            'status': 'success',
            'mean_corr': float(np.mean(valid)),
            'std_corr': float(np.std(valid)),
            'n_cells': len(valid),
            'note': 'Standard stochastic mode on spliced/unspliced data'
        }
    except Exception as e:
        return {
            'method': 'scVelo_Velocity',
            'status': 'failed',
            'reason': str(e),
            'mean_corr': None,
            'std_corr': None,
            'n_cells': 0,
            'note': 'Runtime error during velocity inference.'
        }

if __name__ == "__main__":
    res = run_scvelo_baseline()
    out_path = RESULT_DIR / 'scvelo_baseline.json'
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\n💾 结果已保存至: {out_path}")
    if res['mean_corr'] is not None:
        print(f"📊 Mean Corr: {res['mean_corr']:.3f} ± {res['std_corr']:.3f}")
    else:
        print("⏭️  该基线已标记为不适用，对比图表将自动跳过。")