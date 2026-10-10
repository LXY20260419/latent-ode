# External GEO intake audit — GSE149622

> **Metadata-only intake snapshot — later phases complete.** This report
> records the pre-download audit and its original stop conditions. The exact
> input hashes and leakage-audited bundle QC were subsequently recorded in
> `external_gse149622_qc.json` (`PASS_FOR_SMOKE`), followed by the completed
> three- and five-seed reports
> `external_phase2_report_v1.md` and
> `external_phase3_report_v2.md`. The pending-scope manuscript
> decision remains separate; do not treat the historical intake wording below
> as current pipeline status.

- Audit time (UTC): `2026-10-01T15:30:04.502549+00:00`
- Scope: metadata-only; no expression matrix was downloaded in this phase.
- Assessment: **GO_FOR_PHASE1_WITH_REPLICATE_LIMITATION_AUDIT: five RNA time points and MTX files are available; sample-level replication must be verified from barcode/sample metadata before any inferential claim.**

## Series metadata

- Title: Skin-resident innate lymphoid cells converge on a pathogenic effector state
- Organism: `Mus musculus`
- GEO supplementary types: `BED, FA, MTX, TBI, TSV`
- GEO sample count: `7`
- PubMed IDs: `33536623, 37398022`
- GEO page: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149622

## Selected RNA samples

| timepoint | selected samples | accessions |
|---:|---:|---|
| 0 | 1 | GSM4506037 |
| 1 | 1 | GSM4506038 |
| 2 | 1 | GSM4506039 |
| 3 | 1 | GSM4506040 |
| 4 | 1 | GSM4506041 |

## Selected supplementary files

| file | remote size | last modified | status |
|---|---:|---|---|
| `GSE149622_RNA_timecourse_unadjusted_matrix.mtx.gz` | 144.1 MiB | Mon, 27 Apr 2020 22:11:59 GMT | present |
| `GSE149622_RNA_timecourse_unadjusted_barcodes.tsv.gz` | 122.7 KiB | Wed, 29 Apr 2020 22:46:23 GMT | present |
| `GSE149622_RNA_timecourse_unadjusted_genes.tsv.gz` | 66.5 KiB | Wed, 29 Apr 2020 22:46:23 GMT | present |

## Locked first-pass protocol

- Result tag: `ext_gse149622_v1`
- Train timepoints: `[0, 1, 2, 3]`
- Primary holdout: `[4]`
- Secondary holdout: `[]`
- Primary endpoint: `day3->day4 population_mean_across_gene_pearson_r`
- Condition filter: RNA time-course samples only; exclude ATAC samples
- Row unit: single cell, nested within one pooled RNA sample per timepoint
- Biological replicate unit: unavailable at each timepoint; no biological-replicate inference

## Caveats and phase-1 stop conditions

- The series is a skin ILC/IL-23 inflammatory trajectory, not LPS macrophage data.
- ATAC samples must never enter the RNA expression matrix.
- The GEO series may provide one pooled RNA sample per time point; if so, only seed/protocol uncertainty can be estimated.
- Do not start three-seed production until the downloaded file hashes, matrix orientation, timepoint mapping and per-timepoint cell counts pass QC.
- Any sample/condition mismatch is a stop condition and must be recorded rather than repaired post hoc.

## Provenance status

- GEO metadata and supplementary directory listing were queried directly.
- At intake time, SHA256 values were pending. They were resolved after
  download and are recorded in `external_gse149622_qc.json` under
  `input_sha256`, `gene_names_sha256`, `metadata_sha256`, and
  `expression_matrix_sha256`; the QC status is `PASS_FOR_SMOKE`.
- This report does not promote the candidate into a manuscript claim.
