# External GEO intake audit — GSE148794

> **Metadata-only intake snapshot — later phases complete.** This report
> records the pre-download audit and its original stop conditions. The exact
> input hashes and leakage-audited bundle QC were subsequently recorded in
> `external_gse148794_qc.json` (`PASS_FOR_SMOKE`), followed by the completed
> three- and five-seed reports
> `external_phase2_report_v1.md` and
> `external_phase3_report_v2.md`. The pending-scope manuscript
> decision remains separate; do not treat the historical intake wording below
> as current pipeline status.

- Audit time (UTC): `2026-10-01T15:30:03.618128+00:00`
- Scope: metadata-only; no expression matrix was downloaded in this phase.
- Assessment: **GO_FOR_PHASE1_WITH_REPLICATE_LIMITATION_AUDIT: the IBD time course has multiple GEO sample/library groups, but source-study pooling means they must not be treated as independent animal replicates; exclude all Serpin intervention arms before conversion.**

## Series metadata

- Title: Longitudinal single cell RNA-seq of the inflamed colon
- Organism: `Mus musculus`
- GEO supplementary types: `TSV`
- GEO sample count: `88`
- PubMed IDs: `33862275`
- GEO page: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE148794

## Selected RNA samples

| timepoint | selected samples | accessions |
|---:|---:|---|
| 0 | 10 | GSM4481180, GSM4481181, GSM4481182, GSM4481183, GSM4481184, GSM4481185, GSM4481186, GSM4481187, GSM4481188, GSM4481189 |
| 3 | 10 | GSM4481190, GSM4481191, GSM4481192, GSM4481193, GSM4481194, GSM4481195, GSM4481196, GSM4481197, GSM4481198, GSM4481199 |
| 6 | 10 | GSM4481200, GSM4481201, GSM4481202, GSM4481203, GSM4481204, GSM4481205, GSM4481206, GSM4481207, GSM4481208, GSM4481209 |
| 9 | 10 | GSM4481210, GSM4481211, GSM4481212, GSM4481213, GSM4481214, GSM4481215, GSM4481216, GSM4481217, GSM4481218, GSM4481219 |
| 12 | 10 | GSM4481220, GSM4481221, GSM4481222, GSM4481223, GSM4481224, GSM4481225, GSM4481226, GSM4481227, GSM4481228, GSM4481229 |
| 15 | 10 | GSM4481230, GSM4481231, GSM4481232, GSM4481233, GSM4481234, GSM4481235, GSM4481236, GSM4481237, GSM4481238, GSM4481239 |

## Selected supplementary files

| file | remote size | last modified | status |
|---|---:|---|---|
| `GSE148794_tc_ibd.count_table.tsv.gz` | 27.3 MiB | Mon, 06 Apr 2020 02:56:27 GMT | present |
| `GSE148794_tc_ibd.metadata.tsv.gz` | 1.3 KiB | Mon, 06 Apr 2020 03:46:46 GMT | present |

## Locked first-pass protocol

- Result tag: `ext_gse148794_v1`
- Train timepoints: `[0, 3, 6, 9]`
- Primary holdout: `[12]`
- Secondary holdout: `[15]`
- Primary endpoint: `9h->12h population_mean_across_gene_pearson_r`
- Condition filter: Time Course IBD only; exclude Serpin arms
- Replicate unit: `cell nested within GEO sample/library group; source-study pooling means these are not independent animal replicates`

## Caveats and phase-1 stop conditions

- The dataset is a tissue-level DSS colitis trajectory, not a macrophage-only culture.
- The source study pooled biological material before profiling; GEO sample/library groups and cells are not independent animal replicates.
- Cell composition and sample/batch effects must be separated from within-state expression.
- The primary endpoint is fixed before any model output is inspected.
- Do not start three-seed production until the downloaded file hashes, matrix orientation, timepoint mapping and per-timepoint cell counts pass QC.
- Evaluate 12-day and 15-day holdouts separately; 12 days remains primary and 15 days secondary.
- Any sample/condition mismatch is a stop condition and must be recorded rather than repaired post hoc.

## Provenance status

- GEO metadata and supplementary directory listing were queried directly.
- At intake time, SHA256 values were pending. They were resolved after
  download and are recorded in `external_gse148794_qc.json` under
  `input_sha256`, `gene_names_sha256`, `metadata_sha256`, and
  `expression_matrix_sha256`; the QC status is `PASS_FOR_SMOKE`.
- This report does not promote the candidate into a manuscript claim.
