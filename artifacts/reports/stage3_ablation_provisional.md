# Stage 3 ODE Ablation (Provisional Framing)

> **Historical provisional plan — superseded for current claims.** The
> 20-epoch ablation plan and command template below predate the tagged
> 100-epoch internal freeze and the matched pure-MSE comparison. They are
> retained for audit history only; do not interpret the old contribution
> ranking as current evidence or launch these commands without a new,
> explicitly authorized result tag. The current pathway-term conclusion is
> the null matched pure-MSE comparison in
> `artifacts/reports/pure_mse_vs_primary_comparison.json`.

Purpose: quantify contribution of each ODE-related component to extrapolation gain.

## Planned ablation axes

1. `pathway_score_mse` off (`LEWORLD_PATHWAY_WEIGHT=0`)
2. `pathway_delta_mse` off (`LEWORLD_PATHWAY_DELTA_WEIGHT=0`)
3. External priors off (unset `LEWORLD_EXTERNAL_PATHWAY_GMT` and `LEWORLD_EXTERNAL_TF_TARGET_CSV`)
4. Full model (reference)

## Fixed controls

- Seeds: `42, 123, 2026`
- Epochs: `LEWORLD_EPOCHS=20`
- Training rollout cap: `LEWORLD_TRAIN_MAX_PREDICT_STEPS=8`
- Endpoint definition unchanged from current reports

## Command template (single axis example)

```bash
conda activate leworld_sc
cd .

LEWORLD_PATHWAY_WEIGHT=0 LEWORLD_EPOCHS=20 LEWORLD_TRAIN_MAX_PREDICT_STEPS=8 python scripts/run_experiment.py ode --seed 42
LEWORLD_PATHWAY_WEIGHT=0 LEWORLD_EPOCHS=20 LEWORLD_TRAIN_MAX_PREDICT_STEPS=8 python scripts/run_experiment.py ode --seed 123
LEWORLD_PATHWAY_WEIGHT=0 LEWORLD_EPOCHS=20 LEWORLD_TRAIN_MAX_PREDICT_STEPS=8 python scripts/run_experiment.py ode --seed 2026
```

## Provisional interpretation anchor (already observed)

- Gap-aligned training is associated with the largest extrapolation shift in ODE (from negative to positive).
- GRU remains a useful negative/control comparator for extrapolation.
- Full ablation is still needed before writing final component-level causal claims.
