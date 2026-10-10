# External GEO intake audit — GSE305979

> **Pre-training intake snapshot — Phase 4 subsequently completed.** The
> `GO_FOR_QC_THEN_SMOKE` assessment and stop conditions below were frozen
> before conversion and training. Source-label auditing, the `^DMSO$` vehicle
> decision, QC, smoke, and the preregistered three-seed production all
> completed afterward; current results are in
> `external_phase3_report_v2.md` and
> `external_gse305979_stats.json`. Use this
> file for intake provenance and limitations, not for current run status.

- Audit time (UTC): `2026-10-03T00:30:00+00:00`
- Scope: metadata-only preflight plus structural verification of the two
  selected H5AD supplements; no model results were inspected.
- Assessment: **GO_FOR_QC_THEN_SMOKE — human CD34+ HSPC megakaryocyte
  differentiation time course (day 0/1/2/5/7) with an untreated day-0 sample
  and DMSO vehicle controls inside the hashed days 1–7 libraries. This is a
  cross-species and cross-system stress test; it must not be merged into the
  internal frozen result or the primary external comparison.**

## Series metadata

- Title: Perturbational single-cell RNA-seq time course of CD34+ HSPC
  megakaryocyte differentiation with CMO multiplexing (Days 0–7, 1 uM
  compounds)
- Organism: `Homo sapiens` (dual-mobilized peripheral blood CD34+ HSPCs,
  single donor)
- GEO sample count: `34` (GEX and ADT libraries across day 0/1/2/5/7)
- PubMed IDs: `41129612` (Science, DOI `10.1126/science.adi8577`)
- GEO page: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE305979>
- Supplementary listing: 4 H5AD files (see table below)

## Selected supplementary files

| file | remote size (bytes) | role | status |
|---|---:|---|---|
| `GSE305979_day0_raw_counts.h5ad` | 42,727,199 | untreated day-0 baseline | downloaded 2026-10-03, HDF5 magic verified |
| `GSE305979_day1-7_demuxed_counts.h5ad` | 1,373,371,436 | demultiplexed singlet counts with CMO assignments | downloaded 2026-10-03; HDF5 magic and SHA256 verified |
| `GSE305979_day0-7_normalized_counts_with_celltype_annotations.h5ad` | 2,984,499,642 | not downloaded | excluded by design |
| `GSE305979_day1-7_raw_counts.h5ad` | 1,904,213,868 | not downloaded | pre-demux; excluded by design |

Design rationale for the two selected files: the protocol re-derives
normalization and feature selection from counts with fit-on-train statistics,
so the 2.98 GB normalized file adds only size. The demuxed file is required
because the day 1–7 libraries multiplex 12 CMO-tagged samples per library and
only the demuxed object carries the well/treatment assignments needed to
select the vehicle-control trajectory.

## Structural verification (day-0 file)

Read directly from `GSE305979_day0_raw_counts.h5ad`:

- `X`: CSR sparse, `float32`, shape `(2875 cells, 58676 genes)`, raw counts
  (max 6940, min 0, mean library sum ≈ 45,146).
- `obs.TIMEPOINT_HOURS`: single value `0`.
- `obs`: two libraries (`Day 0_Lib 1` n=1361, `Day 0_Lib 2` n=1514), all
  `ASSAY_ID = 10X 3'`, `LIBRARY_TYPE = CELL`, `SPECIES = Homo sapiens`.
- `var`: 58,676 gene symbols with `gene_ids`, `feature_types`, `genome`.
- The day-0 file contains no treatment column because day-0 cells are
  unhashed and untreated.

## Locked first-pass protocol (frozen before any training run)

- Result tag: `ext_gse305979_control_v1`
- Condition filter: vehicle/untreated trajectory only. Day 0 (untreated) plus
  the DMSO vehicle wells from the demuxed day 1–7 file. Compound-treated wells
  are excluded from the primary analysis and reserved as an independent
  perturbation stress test in a later stage.
- Time unit: **days**, matching the existing external protocol convention
  (`GSE148794` used days; `GSE149622` used days). The initial preregistration
  proposed deriving days by dividing `TIMEPOINT_HOURS` by 24. The demux audit
  found that all Day 7 cells were recorded as 24.0 h, so the source-label
  amendment locked before conversion/training derives days from `PLATE_NAME`
  (fallback `sample_name`/`POOL_NAME`) and keeps `TIMEPOINT_HOURS` for audit only.
  See `external_gse305979_protocol_amendment_2026-10-03.json`.
- Train timepoints: `0, 1, 2` days
- Primary holdout: `5` days (gap = 3 days from the last train time)
- Secondary holdout: `7` days (gap = 5 days from the last train time)
- Primary endpoint: `2d->5d population_mean_across_gene_pearson_r`
- Secondary endpoint: `2d->7d population_mean_across_gene_pearson_r`
- Replicate unit: `cell nested within CMO well and library; single donor, so
  no donor-level inference`. Treat every interval as protocol/seed-level
  uncertainty.
- Model arms: ODE, matched GRU, population persistence, plus the classical
  baseline family in a later stage. Pure-MSE only if the endpoint survives
  the three-seed smoke review.
- Seeds: `42, 123, 2026` after the one-seed smoke passes.

The split mirrors the GSE148794 structure (short train range, two subsequent
future holdouts) so the cross-dataset gap-decay comparison stays interpretable.
Both gaps (3 days, 5 days) stay inside the 8-step rollout budget
(`LEWORLD_TRAIN_MAX_PREDICT_STEPS=8`) that the frozen protocol uses; with
`LEWORLD_MAX_TIME=7` the normalized time axis is comparable to the other
external runs.

## Caveats and stop conditions

- Single donor and in vitro differentiation: cross-system generality, not
  population-level biology.
- Compound, plate/library and time are partially confounded in the hashed
  libraries; the vehicle filter is the only clean longitudinal trajectory.
  If the demuxed file does not expose an explicit vehicle label, map the CMO
  assignment against the series metadata and freeze the mapping before any
  training. If the mapping cannot be made unambiguous, stop and record.
- Day-0 cells are unhashed with no compound exposure; day 1+ cells are
  hashed. Library-level batch structure between day 0 and days 1–7 must be
  reported, not hidden.
- Cell counts per timepoint and per CMO well must be recorded in the QC file.
- At intake, production was gated on the demuxed-file hash, HDF5 integrity,
  timepoint mapping, and vehicle-filter QC; those gates were subsequently
  satisfied and recorded in `external_gse305979_qc.json` and the vehicle
  decision report.
- Any orientation, treatment or timepoint mismatch is a stop condition.

## Provenance status

- GEO series metadata and supplementary listing queried directly
  (`2026-10-03`).
- SHA256 values for both selected files are recorded in
  `external_gse305979_qc.json`; the demuxed input hash is
  `f60525c952fcab98418285dfd799f1df2650a636037deb630b80af4d6501f763`.
- Post-intake status: source-label audit, vehicle decision, conversion, smoke
  and the preregistered three-seed production runs all completed; the results
  remain an independent stress-test layer.
- This report does not promote the candidate into any manuscript claim and
  does not modify the frozen internal result or the submission gate.
