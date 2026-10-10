# External Phase-3 report (per-dataset seed panels)

Generated: 2026-10-03T05:44:21.350035+00:00

Protocol: `artifacts/reports/external_phase2_protocol_v1.json` plus the
confirmation preregistration `artifacts/reports/external_phase3_preregistration.json`.
Seeds 42/123/2026 were locked in phase 2. Seeds 7 and 13 were preregistered
in phase 3 before any of their results existed and apply to the two mouse
candidates; GSE305979 was preregistered later with a three-seed panel, and
its 7/13 extension is a separate decision. Each row states its own panel.

All values are `population_mean_across_gene_pearson_r` on the same
2,991-gene panel and train-only preprocessing. External results are
stress tests and do not modify the frozen internal primary result.

## GSE148794

- result tag: `ext_gse148794_v2`
- train -> holdout: [0, 3, 6, 9] -> [12]
- replicate note: GEO sample/library groups are not independent animal replicates; the source study pooled biological material before profiling. Protocol/seed-level inference only.

| endpoint | model | per-seed | mean | SD | seed CI95 |
|---|---|---|---:|---:|---|
| 12 | ODE | +0.5534, +0.5616, +0.5560, +0.5763, +0.5491 | +0.5593 | 0.0105 | [+0.5522, +0.5682] |
| 12 | GRU | +0.5600, +0.5809, +0.4769, +0.5126, +0.5241 | +0.5309 | 0.0408 | [+0.5006, +0.5631] |
| 12 | persistence (deterministic) | fixed | +0.5587 | — | — |
| 15 | ODE | +0.6588, +0.6716, +0.6641, +0.6804, +0.6585 | +0.6667 | 0.0093 | [+0.6597, +0.6743] |
| 15 | GRU | +0.6846, +0.6834, +0.6157, +0.6650, +0.5621 | +0.6422 | 0.0528 | [+0.5971, +0.6802] |
| 15 | persistence (deterministic) | fixed | +0.6661 | — | — |

## GSE149622

- result tag: `ext_gse149622_v2`
- train -> holdout: [0, 1, 2, 3] -> [4]
- replicate note: one pooled RNA sample per timepoint; no biological-replicate inference

| endpoint | model | per-seed | mean | SD | seed CI95 |
|---|---|---|---:|---:|---|
| 4 | ODE | +0.4046, +0.4027, +0.4031, +0.3979, +0.3955 | +0.4008 | 0.0039 | [+0.3975, +0.4036] |
| 4 | GRU | +0.4050, +0.4028, +0.4030, +0.3978, +0.3955 | +0.4008 | 0.0040 | [+0.3975, +0.4038] |
| 4 | persistence (deterministic) | fixed | +0.4017 | — | — |

## GSE305979

- result tag: `ext_gse305979_control_v2`
- train -> holdout: [0, 1, 2] -> [5]
- replicate note: single donor; cells nested within CMO well and library; no donor-level inference. Protocol/seed-level uncertainty only.

| endpoint | model | per-seed | mean | SD | seed CI95 |
|---|---|---|---:|---:|---|
| 5 | ODE | +0.6451, +0.6477, +0.6499 | +0.6476 | 0.0024 | [+0.6456, +0.6494] |
| 5 | GRU | +0.6424, +0.6573, +0.6315 | +0.6437 | 0.0130 | [+0.6338, +0.6540] |
| 5 | persistence (deterministic) | fixed | +0.6468 | — | — |
| 7 | ODE | +0.5829, +0.5854, +0.5919 | +0.5867 | 0.0046 | [+0.5834, +0.5904] |
| 7 | GRU | +0.6039, +0.5451, +0.5660 | +0.5716 | 0.0298 | [+0.5496, +0.5957] |
| 7 | persistence (deterministic) | fixed | +0.5845 | — | — |

## Contrasts

| dataset | endpoint | ODE - GRU (mean) | ODE - persistence (mean) |
|---|---|---:|---:|
| GSE148794 | 12 | +0.0284 | +0.0006 |
| GSE148794 | 15 | +0.0245 | +0.0006 |
| GSE149622 | 4 | -0.0001 | -0.0009 |
| GSE305979 | 5 | +0.0039 | +0.0007 |
| GSE305979 | 7 | +0.0151 | +0.0022 |

## Failure-boundary statement

All endpoints above are reported regardless of sign. Endpoints where ODE does not exceed matched GRU or persistence are retained as boundary results and must not be filtered out of the manuscript or supplement.

