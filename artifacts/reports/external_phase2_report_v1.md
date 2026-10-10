# External Phase-2 production report (3 seeds)

> **Historical three-seed report — not the latest confirmation summary.**
> These values document the initial Phase-2 panel (42/123/2026). Seeds 7 and
> 13 were subsequently preregistered and added for the two mouse candidates;
> use `external_phase3_report_v2.md` for the current five-seed summaries.
> This report remains useful for phase provenance and does not alter the
> frozen internal result or gate.

Generated: 2026-10-02T00:04:16.374932+00:00

Protocol: `artifacts/reports/external_phase2_protocol_v1.json` (locked before any 100-epoch external run was inspected).

All values are `population_mean_across_gene_pearson_r` on the same 2,991-gene panel and train-only preprocessing. External results are stress tests and do not modify the frozen internal primary result.

## GSE148794

- result tag: `ext_gse148794_v2`
- train -> holdout: [0, 3, 6, 9] -> [12]
- replicate note: GEO sample/library groups are not independent animal replicates; the source study pooled biological material before profiling. Protocol/seed-level inference only.

| endpoint | model | per-seed | mean | SD | 27-resample CI95 |
|---|---|---|---:|---:|---|
| 12 | ODE | +0.5534, +0.5616, +0.5560 | +0.5570 | 0.0042 | [+0.5540, +0.5604] |
| 12 | GRU | +0.5600, +0.5809, +0.4769 | +0.5393 | 0.0550 | [+0.4949, +0.5764] |
| 12 | persistence (deterministic) | fixed | +0.5587 | — | — |
| 15 | ODE | +0.6588, +0.6716, +0.6641 | +0.6648 | 0.0064 | [+0.6599, +0.6700] |
| 15 | GRU | +0.6846, +0.6834, +0.6157 | +0.6612 | 0.0394 | [+0.6303, +0.6844] |
| 15 | persistence (deterministic) | fixed | +0.6661 | — | — |

Actual completed epochs (the locked 100-epoch setting is a cap; early stopping is allowed):

- ODE: seed 42: 100, seed 123: 100, seed 2026: 100
- GRU: seed 42: 100, seed 123: 100, seed 2026: 100

## GSE149622

- result tag: `ext_gse149622_v2`
- train -> holdout: [0, 1, 2, 3] -> [4]
- replicate note: one pooled RNA sample per timepoint; no biological-replicate inference

| endpoint | model | per-seed | mean | SD | 27-resample CI95 |
|---|---|---|---:|---:|---|
| 4 | ODE | +0.4046, +0.4027, +0.4031 | +0.4035 | 0.0010 | [+0.4028, +0.4043] |
| 4 | GRU | +0.4050, +0.4028, +0.4030 | +0.4036 | 0.0012 | [+0.4028, +0.4046] |
| 4 | persistence (deterministic) | fixed | +0.4017 | — | — |

Actual completed epochs (the locked 100-epoch setting is a cap; early stopping is allowed):

- ODE: seed 42: 100, seed 123: 100, seed 2026: 100
- GRU: seed 42: 100, seed 123: 100, seed 2026: 100

## Contrasts

| dataset | endpoint | ODE - GRU (mean) | ODE - persistence (mean) |
|---|---|---:|---:|
| GSE148794 | 12 | +0.0177 | -0.0017 |
| GSE148794 | 15 | +0.0036 | -0.0013 |
| GSE149622 | 4 | -0.0001 | +0.0018 |

## Failure-boundary statement

All endpoints above are reported regardless of sign. Endpoints where ODE does not exceed matched GRU or persistence are retained as boundary results and must not be filtered out of the manuscript or supplement.
