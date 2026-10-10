# Existing external boundary audit — 2026-10-01

Scope: read-only audit; no retraining, manuscript edit, primary-freeze edit, or gate edit.

## GSE224518

- 3→5 h: ODE is positive but does not beat matched GRU; Ridge is stronger on the same population extrapolation endpoint.
- 3→8 h: ODE is negative with low seed spread; retain as the long-horizon failure boundary.
- Sample-tag, batch, and replicate composition changes across timepoints; cells are not independent biological replicates.
- The bundle uses a targeted 498-gene panel, so it is not directly comparable to the full-transcriptome internal bundle.

## GSE226824

- 24→72 h: ODE mean is positive but seed variance is large; treat as exploratory stress-test evidence.
- Hashtags and clusters are not donor-level biological replication by themselves.
- No sample/donor-level inferential uncertainty is claimed.

## Decision

- Do not pool these endpoints or promote them to broad generalization claims.
- Finish Phase-1 QC/smoke for GSE148794 and GSE149622 before any production expansion.

Machine-readable source: `artifacts/reports/existing_external_boundary_audit_2026-10-01.json`
