# Cross-dataset horizon-decay table

Generated: 2026-10-03T19:54:25.757728+00:00

Population-mean across-gene Pearson r at each extrapolation horizon,
grouped by dataset. `gap` is the extrapolation distance in the
dataset's native time unit; `steps` is the recorded predict_steps
(ODE integration / rollout steps). Negative GRU values and horizon
reversals are reported as-is: no arm or horizon is filtered out.
`persistence` is the deterministic no-training source-time carry-
forward baseline on the same endpoint.

| dataset | source->target | gap | steps | n seeds | ODE mean ± SD | GRU mean ± SD | ODE−GRU | persistence | ODE−persistence |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| GSE151658 (frozen rerun) | 27->36 | 9 h | 9 | 3 | +0.6838 ± 0.0050 | -0.2093 ± 0.0519 | +0.8931 | +0.6823 | +0.0016 |
| GSE151658 (frozen rerun) | 27->48 | 21 h | 21 | 3 | +0.2193 ± 0.0071 | -0.2559 ± 0.0184 | +0.4752 | +0.2112 | +0.0081 |
| GSE148794 | 9->12 | 3 d | 3 | 5 | +0.5593 ± 0.0105 | +0.5309 ± 0.0408 | +0.0284 | +0.5587 | +0.0006 |
| GSE148794 | 9->15 | 6 d | 6 | 5 | +0.6667 ± 0.0093 | +0.6422 ± 0.0528 | +0.0245 | +0.6661 | +0.0006 |
| GSE149622 | 3->4 | 1 d | 1 | 5 | +0.4008 ± 0.0039 | +0.4008 ± 0.0040 | -0.0001 | +0.4017 | -0.0009 |
| GSE305979 (human CD34+ HSPC, DMSO) | 2->5 | 3 d | 3 | 3 | +0.6476 ± 0.0024 | +0.6437 ± 0.0130 | +0.0039 | +0.6468 | +0.0007 |
| GSE305979 (human CD34+ HSPC, DMSO) | 2->7 | 5 d | 5 | 3 | +0.5867 ± 0.0046 | +0.5716 ± 0.0298 | +0.0151 | +0.5845 | +0.0022 |

## Reading notes

- Horizons differ by dataset; gaps are not pooled or compared as if they were the same experiment.
- The internal dataset drops from 0.684 (36 h) to 0.219 (48 h); GSE148794 runs the other way for every arm including persistence (0.559 -> 0.667 ODE, 0.559 -> 0.666 persistence), so there the longer-gap endpoint is intrinsically closer to the source profile. Decay is not a stable law across datasets; this table shows each shape rather than fitting a universal one.
- GSE149622 has a single holdout horizon, so its decay shape is a single point by design.
- GSE305979 contributes the preregistered 2->5 and 2->7 day vehicle-control horizons with three seeds; its ODE and persistence curves are close, and the table does not upgrade this stress test to a universal decay claim.

