# Failure-boundary table (all endpoints, all signs)

Generated: 2026-10-04T06:37:28.665793+00:00

Policy (pre-committed): every endpoint is retained and reported,
including ODE ties and losses. A `win` requires a positive mean
difference with a 95% interval lower bound above zero; intervals that
span zero are `tie`. Rows with only a one-sided p-value are
`win_p_only` / `loss_p_only` when p < 0.05 and `not_significant`
otherwise; rows with no interval and no p-value stay at
`positive_point_estimate` / `negative_point_estimate` and must not be
upgraded to significance. Mechanism controls are `null` while
indistinguishable from their reference, and the random-prior row
compares the observed value against the null distribution quantiles
rather than a difference interval.

| layer | endpoint | comparator | left | right | diff | 95% CI | p | verdict |
|---|---|---|---:|---:|---:|---|---:|---|
| internal_frozen | 27->36/48 composite | matched GRU | +0.4516 (ODE) | -0.2326 (GRU) | +0.6842 | — | 5.633e-05 | win_p_only |
| internal_frozen | 27->36/48 composite | population persistence | +0.4516 (ODE) | +0.4468 (persistence) | +0.0048 | — | 0.1449 | not_significant |
| external_phase3 | GSE148794 9->12 | matched GRU | +0.5593 (ODE) | +0.5309 (GRU) | +0.0284 | [-0.0091, +0.0659] | — | tie |
| external_phase3 | GSE148794 9->12 | population persistence | +0.5593 (ODE) | +0.5587 (persistence) | +0.0006 | — | — | positive_point_estimate |
| external_phase3 | GSE148794 9->15 | matched GRU | +0.6667 (ODE) | +0.6422 (GRU) | +0.0245 | [-0.0186, +0.0676] | — | tie |
| external_phase3 | GSE148794 9->15 | population persistence | +0.6667 (ODE) | +0.6661 (persistence) | +0.0006 | — | — | positive_point_estimate |
| external_phase3 | GSE149622 3->4 | matched GRU | +0.4008 (ODE) | +0.4008 (GRU) | -0.0001 | [-0.0002, +0.0001] | — | tie |
| external_phase3 | GSE149622 3->4 | population persistence | +0.4008 (ODE) | +0.4017 (persistence) | -0.0009 | — | — | negative_point_estimate |
| external_phase3 | GSE305979 2->5 | matched GRU | +0.6476 (ODE) | +0.6437 (GRU) | +0.0039 | [-0.0120, +0.0197] | — | tie |
| external_phase3 | GSE305979 2->5 | population persistence | +0.6476 (ODE) | +0.6468 (persistence) | +0.0007 | — | — | positive_point_estimate |
| external_phase3 | GSE305979 2->7 | matched GRU | +0.5867 (ODE) | +0.5716 (GRU) | +0.0151 | [-0.0212, +0.0513] | — | tie |
| external_phase3 | GSE305979 2->7 | population persistence | +0.5867 (ODE) | +0.5845 (persistence) | +0.0022 | — | — | positive_point_estimate |
| historic_external | GSE224518 3->5 | matched GRU | +0.1346 (ODE) | +0.1773 (GRU) | -0.0427 | — | — | negative_point_estimate |
| historic_external | GSE224518 3->5 (balanced) | matched GRU | +0.1386 (ODE) | +0.1371 (GRU) | +0.0015 | [-0.0173, +0.0203] | — | tie |
| historic_external | GSE224518 3->8 | matched GRU | -0.1699 (ODE) | -0.0152 (GRU) | -0.1547 | [-0.3099, +0.0006] | — | tie |
| historic_external | GSE226824 24->72 | matched GRU | +0.2335 (ODE) | +0.1702 (GRU) | +0.0634 | [+0.0106, +0.1162] | — | win |
| mechanism_control | internal 27->36/48 composite | full model (pure MSE minus full) | +0.4571 (ODE pure-MSE) | +0.4516 (ODE full) | +0.0055 | [-0.0277, +0.0387] | 0.5486 | tie |
| mechanism_control | internal 27->36/48 composite | matched-size random pathway priors | +0.4516 (ODE primary) | +0.4543 (random-prior null mean) | -0.0028 | [+0.4476, +0.4623] | 0.6667 | null |
| mechanism_control_external | GSE148794 12 | full model (pure MSE minus full) | +0.5587 (ODE pure-MSE) | +0.5570 (ODE full) | +0.0017 | [-0.0201, +0.0234] | 0.7708 | tie |
| mechanism_control_external | GSE148794 15 | full model (pure MSE minus full) | +0.6662 (ODE pure-MSE) | +0.6648 (ODE full) | +0.0014 | [-0.0297, +0.0325] | 0.8649 | tie |
| mechanism_control_external | GSE149622 4 | full model (pure MSE minus full) | +0.4031 (ODE pure-MSE) | +0.4035 (ODE full) | -0.0003 | [-0.0017, +0.0010] | 1 | tie |

## Notes per row

- **27->36/48 composite** vs matched GRU: one-sided paired t-test, df=2, exploratory (`artifacts/reports/composite_drift_report.json`)
- **27->36/48 composite** vs population persistence: not significant at the composite endpoint (p=0.145) (`artifacts/reports/composite_drift_report.json`)
- **GSE148794 9->12** vs matched GRU: 5 seeds; normal-approximation interval from the seed-level paired differences (`artifacts/reports/external_phase3_forest.json`)
- **GSE148794 9->12** vs population persistence: deterministic persistence, point estimate only (`artifacts/reports/external_phase3_aggregate_v2.json`)
- **GSE148794 9->15** vs matched GRU: 5 seeds; normal-approximation interval from the seed-level paired differences (`artifacts/reports/external_phase3_forest.json`)
- **GSE148794 9->15** vs population persistence: deterministic persistence, point estimate only (`artifacts/reports/external_phase3_aggregate_v2.json`)
- **GSE149622 3->4** vs matched GRU: 5 seeds; normal-approximation interval from the seed-level paired differences (`artifacts/reports/external_phase3_forest.json`)
- **GSE149622 3->4** vs population persistence: deterministic persistence, point estimate only (`artifacts/reports/external_phase3_aggregate_v2.json`)
- **GSE305979 2->5** vs matched GRU: 3 seeds; normal-approximation interval from the seed-level paired differences (`artifacts/reports/external_phase3_forest.json`)
- **GSE305979 2->5** vs population persistence: deterministic persistence, point estimate only (`artifacts/reports/external_phase3_aggregate_v2.json`)
- **GSE305979 2->7** vs matched GRU: 3 seeds; normal-approximation interval from the seed-level paired differences (`artifacts/reports/external_phase3_forest.json`)
- **GSE305979 2->7** vs population persistence: deterministic persistence, point estimate only (`artifacts/reports/external_phase3_aggregate_v2.json`)
- **GSE224518 3->5** vs matched GRU: 3 seeds; retained as boundary evidence (`artifacts/reports/stage2_gse224518_3to5h_agg.json`)
- **GSE224518 3->5 (balanced)** vs matched GRU: 3 seeds; retained as boundary evidence (`artifacts/reports/stage2_gse224518_3to5h_balanced_agg.json`)
- **GSE224518 3->8** vs matched GRU: 3 seeds; retained as boundary evidence (`artifacts/reports/stage2_gse224518_agg.json`)
- **GSE226824 24->72** vs matched GRU: 3 seeds; retained as boundary evidence (`artifacts/reports/stage2_gse226824_agg.json`)
- **internal 27->36/48 composite** vs full model (pure MSE minus full): null: removing pathway terms does not hurt the endpoint (`artifacts/reports/pure_mse_vs_primary_comparison.json`)
- **internal 27->36/48 composite** vs matched-size random pathway priors: null: prior content is not distinguishable from random sets; the interval column holds the null-distribution quantiles [q025, q975], not a difference CI (`artifacts/reports/random_gene_set_negative_control.json`)
- **GSE148794 12** vs full model (pure MSE minus full): null replicates on the external endpoint: removing the pathway losses does not move the population endpoint (`artifacts/reports/external_pure_mse_contrast.json`)
- **GSE148794 15** vs full model (pure MSE minus full): null replicates on the external endpoint: removing the pathway losses does not move the population endpoint (`artifacts/reports/external_pure_mse_contrast.json`)
- **GSE149622 4** vs full model (pure MSE minus full): null replicates on the external endpoint: removing the pathway losses does not move the population endpoint (`artifacts/reports/external_pure_mse_contrast.json`)

## Counting

- negative_point_estimate: 2
- not_significant: 1
- null: 1
- positive_point_estimate: 4
- tie: 11
- win: 1
- win_p_only: 1

