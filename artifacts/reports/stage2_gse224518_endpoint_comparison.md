# GSE224518 External Endpoint Comparison

Last updated: 2026-06-04

Fixed training: `0,1,3h` (LPS M0, 498-gene panel). Seeds: 42, 123, 2026.

## Endpoint A: 3h → 8h

| Model | Hold-out @8h | Extrap 3→8h (3-seed) |
|---|---:|---:|
| ODE | 0.4796 ± 0.0103 | **-0.1699 ± 0.0055** |
| GRU | 0.5025 ± 0.0031 | -0.0152 ± 0.1161 |

Source: `stage2_gse224518_agg.json`

## Endpoint B: 3h → 5h (recommended external protocol)

| Model | Hold-out @5h | Extrap 3→5h (3-seed) |
|---|---:|---:|
| ODE | 0.5219 ± 0.0138 | **+0.1346 ± 0.0006** |
| GRU | 0.5363 ± 0.0012 | +0.1773 ± 0.0683 |

Per-seed ODE extrap: `0.135`, `0.135`, `0.134` (extremely low variance).

Source: `stage2_gse224518_3to5h_agg.json`

```bash
export LEWORLD_TRAIN_TIMEPOINTS=0,1,3
export LEWORLD_HOLDOUT_TIMEPOINTS=5
export LEWORLD_MAX_TIME=5
```

## Baselines (cell-level @5h, deterministic)

| Baseline | Mean corr @5h |
|---|---:|
| Naive_LastTimeMean | 0.063 |
| Ridge_Poly | 0.135 |
| HGB_Regressor | 0.034 |
| MLP_Sklearn | 0.066 |

Source: `baseline_comparison_gse224518_3to5h_seed42.json` (cell-level hold-out).

## Extrapolation significance (population 3→5h, endpoint-aligned)

Same metric as ODE `time_extrapolation`: population mean gene correlation at 5h.

| Comparison | Baseline / GRU mean | ODE mean | p (one-sided ODE >) |
|---|---:|---:|---:|
| vs Naive | 0.131 | 0.135 | **0.007** |
| vs HGB | 0.065 | 0.135 | **0.003** |
| vs Ridge | 0.332 | 0.135 | 1.000 (Ridge wins) |
| vs MLP | 0.179 | 0.135 | 0.917 |
| vs GRU | 0.177 | 0.135 | 0.766 |

Source: `extrapolation_significance_gse224518_3to5h.json`, summary: `stage2_gse224518_3to5h_extrapolation_significance_summary.json`.

Note: Ridge is strong on population extrap at this short horizon; cell-level hold-out Ridge (`0.135`) ≠ population extrap Ridge (`0.332`).

## Endpoint B variant: balanced training (P0 B2)

`LEWORLD_BALANCED_TRAIN=1` — subsample each train timepoint to 1024 cells (min across 0/1/3h).

| Model | Hold-out @5h | Extrap 3→5h |
|---|---:|---:|
| ODE | 0.4175 ± 0.0030 | **0.1386 ± 0.0134** |
| GRU | 0.4205 ± 0.0046 | 0.1371 ± 0.0115 |

vs unbalanced ODE extrap `0.1346 ± 0.0006`, hold-out `0.5219 ± 0.0138`.

Interpretation: balanced sampling slightly raises extrap mean but increases seed variance and lowers hold-out; **unbalanced protocol remains primary** for external reporting.

Source: `stage2_gse224518_3to5h_balanced_agg.json`

## Manuscript guidance

- Use **Endpoint B (3→5h)** for GSE224518 external generalization text.
- Do not pool or average metrics across 3→8h and 3→5h in one sentence.
- Note targeted 500-gene BD panel vs full-transcriptome internal data.
