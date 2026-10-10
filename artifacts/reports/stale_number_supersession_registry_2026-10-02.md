# Stale-number supersession registry — 2026-10-02

Purpose: historical JSON/Markdown/CSV records below still contain the
pre-recovery numbers without an in-file marker. The release retains numerical
records while removing private paths and publication coordination pointers.
Those non-numerical redactions are recorded in REDACTION_MANIFEST.json.
Every downstream consumer must read the authoritative source listed in the
right-hand column, never the stale value.

## Stale values and their replacements

| Stale value | Where it still appears | Authoritative replacement |
|---|---|---|
| ODE extrapolation `0.4833 ± 0.0088` | `stage4_freeze_manifest.json` (`internal_primary_endpoint`), `stage1_endpoint_aligned_check.json`, `stage1_statistical_closure_provisional.json`, `primary_recovery_record.json` | `primary_internal_freeze`: **ODE 0.4516 ± 0.0058**, CI95 [0.4480, 0.4562] (`horizon_specific_uncertainty.json` `composite` block) |
| GRU extrapolation `-0.1444 ± 0.0960` | same four records | `primary_internal_freeze`: **GRU −0.2326 ± 0.0168** (`primary_internal_freeze_manifest.json` `promotion_record`) |
| ODE seed extrapolations `-0.4412 / -0.5189 / -0.5539`, plus aggregate `0.4833 / -0.1444` | `method_summary_table.csv` (`leworld` seed rows and `leworld_gap_aligned_trainfix` aggregate rows); notes point to untagged `validation_results_{ode,gru}_seed*.json` | Use only the tagged `primary_internal_freeze` validation files and manifest promotion record: **ODE 0.4516 ± 0.0058**, **GRU −0.2326 ± 0.0168**; retain the CSV as historical audit material, not as a sole source |
| ODE reconstruction `0.3018`, GRU `0.3168` | historical freeze-era documents | tagged freeze: **ODE 0.3521 ± 0.0010 / GRU 0.3518 ± 0.0035** — effectively tied (tagged validation artifacts) |
| "ODE beats persistence" framing | `PRIMARY_RECOVERY_2026-09-26.md` §4 ("significantly above persistence" wording) | **not significant**: p = 0.145 (`composite_drift_report.json` `endpoint_aligned_contrasts.ode_vs_persistence.composite`); vs the no-training naive control p = 0.041 |
| "pathway_score_mse most important" | `stage3_ablation_contribution_ranking.json` era documents | **matched pure-MSE null**: +0.0055, CI95 [−0.028, +0.039], p = 0.55 (`pure_mse_vs_primary_comparison.json`) |
| 20-control random null "supports specificity" reading | early ablation notes | **null**: primary 0.4516 vs null mean 0.4543, p = 0.67 (`random_gene_set_negative_control.json`) |

## Authoritative source list (read order)

1. `artifacts/reports/primary_internal_freeze_manifest.json`
2. `artifacts/reports/stage2_primary_internal_freeze_extrapolation_significance_summary.json`
3. `artifacts/reports/composite_drift_report.json`
4. `artifacts/reports/persistence_vs_primary_comparison.json` — **legacy; do not cite it for the persistence claim.** It predates the tagged rerun (`primary_internal_recovered`, p = 0.0139). The current source of truth is item 3, `composite_drift_report.json` (`endpoint_aligned_contrasts.ode_vs_persistence`).
5. `artifacts/reports/pure_mse_ablation_agg.json` and `pure_mse_vs_primary_comparison.json`
6. `artifacts/reports/random_gene_set_negative_control.json`
7. `artifacts/reports/external_phase2_aggregate_v1.json` (3-seed external record)
8. `artifacts/reports/external_phase3_aggregate_v2.json` (5-seed confirmation, once complete)

## Notes

- June-era records are historical only; use the tagged validation artifacts
  and the analysis reports listed above for current research summaries.
- `method_summary_table.csv` is likewise retained as a June-era machine-readable
  summary. Its current untagged result-file notes no longer resolve to internal
  primary artifacts (those filenames now hold the GSE224518 3→5 h external
  run); `primary_internal_freeze_manifest.json` already records this CSV as
  "not a valid sole source for paper-v1-freeze claims." The
  `.broken_20260529_084911` sibling is an unreferenced duplicate historical
  artifact and is covered by the same audit warning.
- This registry changes no numerical result or research claim.
- Pointer-corrected 2026-10-04: the persistence row and read-order item 4 no
  longer cite the superseded `persistence_vs_primary_comparison.json`
  (actual legacy p = 0.0139); rows 1–2 now point to where the SD and CI95
  actually live. Historical numerical values remain unchanged.
