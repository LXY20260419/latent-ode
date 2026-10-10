# Mechanism-proxy metrics across internal and external arms

Generated: 2026-10-03T20:02:43.365672+00:00

These are prior-consistency proxies recorded by the training script's
validation stage, aggregated across seeds for the first time. The
pathway sets feed the training objective (`pathway_score` weight 0.2,
`pathway_delta` weight 0.1), and the TF/marker sets were merged into
the same priors object, so none of these numbers is an independent
mechanism validation: they measure how far each trained arm sits from
the provided biology priors, nothing more. All seeds are kept,
including values that cut against ODE.

| layer | arm | TF direction | pathway order | marker rank Spearman |
|---|---|---:|---:|---:|
| internal_primary_freeze | ODE | +0.583 ± 0.000 (n=3) | +0.444 ± 0.192 (n=3) | -0.475 ± 0.043 (n=3) |
| internal_primary_freeze | GRU | +0.278 ± 0.096 (n=3) | +0.500 ± 0.167 (n=3) | +0.042 ± 0.247 (n=3) |
| gse148794 | ODE | +0.267 ± 0.149 (n=5) | +0.333 ± 0.471 (n=5) | -0.050 ± 0.570 (n=5) |
| gse148794 | GRU | +0.533 ± 0.380 (n=5) | +0.800 ± 0.217 (n=5) | +0.250 ± 0.500 (n=5) |
| gse149622 | ODE | +0.500 ± 0.354 (n=5) | not evaluable | +0.200 ± 0.622 (n=5) |
| gse149622 | GRU | +0.500 ± 0.354 (n=5) | not evaluable | +0.200 ± 0.622 (n=5) |
| gse305979 | ODE | not evaluable | +0.333 ± 0.577 (n=3) | not evaluable |
| gse305979 | GRU | not evaluable | +0.333 ± 0.577 (n=3) | not evaluable |

## ODE minus GRU (mean level)

| layer | TF direction | pathway order | marker rank |
|---|---:|---:|---:|
| internal_primary_freeze | +0.306 | -0.056 | -0.517 |
| gse148794 | -0.267 | -0.467 | -0.300 |
| gse149622 | +0.000 | n/a | +0.000 |
| gse305979 | n/a | +0.000 | n/a |

## Reading notes

- Positive TF/pathway/marker values mean the arm's extrapolated changes agree more with the direction implied by the priors.
- GSE149622 has a single holdout timepoint, so pathway order is structurally not evaluable there; that is recorded, not imputed.
- GSE305979 has pathway-order values for its three seeds, while the validation artifacts do not provide TF-direction or marker-rank values; those cells remain not evaluable.
- The internal layer uses the frozen rerun artifacts that back the 0.4516 composite; the June-era recovered copies (0.4833 lineage) are deliberately not aggregated here.
- Differences are descriptive mean-level contrasts with three or five seeds; they are not significance tests and must not be used to overturn or replace the endpoint claims.

## Source files

- `artifacts/results/ode_benchmark/validation_results_ode_primary_internal_freeze_seed42.json` — `c538d600e018e7fd`
- `artifacts/results/ode_benchmark/validation_results_ode_primary_internal_freeze_seed123.json` — `37857c94fb2c5fe8`
- `artifacts/results/ode_benchmark/validation_results_ode_primary_internal_freeze_seed2026.json` — `3a50859de6fe31b2`
- `artifacts/results/ode_benchmark/validation_results_gru_primary_internal_freeze_seed42.json` — `6eaa534c468b806e`
- `artifacts/results/ode_benchmark/validation_results_gru_primary_internal_freeze_seed123.json` — `da0de2b5aac26e05`
- `artifacts/results/ode_benchmark/validation_results_gru_primary_internal_freeze_seed2026.json` — `e44218a6f39d883e`
- `artifacts/results/ode_benchmark/validation_results_ode_ext_gse148794_v2_seed42.json` — `9cee382cd03282b3`
- `artifacts/results/ode_benchmark/validation_results_ode_ext_gse148794_v2_seed123.json` — `1855fca120036568`
- `artifacts/results/ode_benchmark/validation_results_ode_ext_gse148794_v2_seed2026.json` — `6f19938d1dad7980`
- `artifacts/results/ode_benchmark/validation_results_ode_ext_gse148794_v2_seed7.json` — `c74ee37c2fc20e73`
- `artifacts/results/ode_benchmark/validation_results_ode_ext_gse148794_v2_seed13.json` — `55c11f4a67de0619`
- `artifacts/results/ode_benchmark/validation_results_gru_ext_gse148794_v2_seed42.json` — `0ada6f5477bc9c50`
- `artifacts/results/ode_benchmark/validation_results_gru_ext_gse148794_v2_seed123.json` — `2bb1eef87bcb7227`
- `artifacts/results/ode_benchmark/validation_results_gru_ext_gse148794_v2_seed2026.json` — `a7434ed4bd4bd2b4`
- `artifacts/results/ode_benchmark/validation_results_gru_ext_gse148794_v2_seed7.json` — `c50c184f6827231d`
- `artifacts/results/ode_benchmark/validation_results_gru_ext_gse148794_v2_seed13.json` — `ce7062adf0cbdafa`
- `artifacts/results/ode_benchmark/validation_results_ode_ext_gse149622_v2_seed42.json` — `e0940a69d89d1c55`
- `artifacts/results/ode_benchmark/validation_results_ode_ext_gse149622_v2_seed123.json` — `a7168d99c2e82fe8`
- `artifacts/results/ode_benchmark/validation_results_ode_ext_gse149622_v2_seed2026.json` — `b3af2c4863393881`
- `artifacts/results/ode_benchmark/validation_results_ode_ext_gse149622_v2_seed7.json` — `ab45aecb1d48ed22`
- `artifacts/results/ode_benchmark/validation_results_ode_ext_gse149622_v2_seed13.json` — `95de81464c442228`
- `artifacts/results/ode_benchmark/validation_results_gru_ext_gse149622_v2_seed42.json` — `3b6d5f6833d5d1dc`
- `artifacts/results/ode_benchmark/validation_results_gru_ext_gse149622_v2_seed123.json` — `07564b7b63cc53d3`
- `artifacts/results/ode_benchmark/validation_results_gru_ext_gse149622_v2_seed2026.json` — `baf0d68824d1007c`
- `artifacts/results/ode_benchmark/validation_results_gru_ext_gse149622_v2_seed7.json` — `dd7b9699979fd165`
- `artifacts/results/ode_benchmark/validation_results_gru_ext_gse149622_v2_seed13.json` — `2e01cd32f646f13e`
- `artifacts/results/ode_benchmark/validation_results_ode_ext_gse305979_control_v2_seed42.json` — `8da62a1dbd93730e`
- `artifacts/results/ode_benchmark/validation_results_ode_ext_gse305979_control_v2_seed123.json` — `6180e3f8554132e9`
- `artifacts/results/ode_benchmark/validation_results_ode_ext_gse305979_control_v2_seed2026.json` — `83da08cf70d731bb`
- `artifacts/results/ode_benchmark/validation_results_gru_ext_gse305979_control_v2_seed42.json` — `47964ca74f7d11c7`
- `artifacts/results/ode_benchmark/validation_results_gru_ext_gse305979_control_v2_seed123.json` — `0d56da6b0f75cbda`
- `artifacts/results/ode_benchmark/validation_results_gru_ext_gse305979_control_v2_seed2026.json` — `8ec6ca9aca07c0fa`

