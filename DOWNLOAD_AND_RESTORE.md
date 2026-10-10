# Public data acquisition and restoration

Run commands from the repository root using the environment in README.md.
Only data is retrieved; do not download or execute an installer or script.
No network retrieval was performed while preparing this release candidate.
The filenames below come from the recorded research inputs; availability has
not been newly checked. If a direct URL fails, use the accession's GEO
Supplementary files page and retain the exact filename.

## GEO source index

| Accession | GEO page | Prepared matrix directory |
|---|---|---|
| GSE151658 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE151658 | `data/leworld_input/` |
| GSE148794 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE148794 | `data/leworld_input_gse148794/` |
| GSE149622 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149622 | `data/leworld_input_gse149622/` |
| GSE305979 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE305979 | `data/leworld_input_gse305979/` |
| GSE224518 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE224518 | `data/leworld_input_gse224518/` |
| GSE226824 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE226824 | `data/leworld_input_gse226824/` |

All large matrices and raw inputs are omitted. The historical pre-contract
metadata snapshot is retained as requested, but only for provenance; do not
reconstruct the current benchmark from superseded manifests.

## 1. Primary GSE151658 frozen contract

```bash
mkdir -p data/raw
curl --fail --location --continue-at - \
  --output data/raw/GSE151658_integrated.0h.1h_4h_16h_27h_36h_48h.rds.gz \
  'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE151658&format=file&file=GSE151658_integrated.0h.1h_4h_16h_27h_36h_48h.rds.gz'

python preprocess/preprocess_full_v7_1.4g_integrated_full.py --verify-inputs-only
python preprocess/preprocess_full_v7_1.4g_integrated_full.py
bash scripts/link_data.sh
python scripts/run_experiment.py check
```

The source RDS checksum must be
`4f2847d9946dc06a31df956780dc26d3d76848ee8b7582888b5620871ef14c25`.
The preprocessor verifies this and the frozen cell/gene/summary manifests.
It writes an isolated candidate under `data/integrated_processed/leworld_input/`
and refuses conflicting overwrites. The linking helper only fills missing
files after verifying existing manifests. Do not delete or regenerate the
included frozen manifests to bypass a mismatch.

Expected output: float32 matrix `(38439, 2991)`, timepoints
`0,1,4,16,27,36,48`. Frozen metadata SHA-256:
`80223b5d11d743425550e06ce2413c6dc79a735b073b6ef0a82fcc5ed4ffe3c6`;
gene manifest SHA-256:
`741e496abe9e14fea59c6070b3b7da86b62fa59c2b982fa66e2ab547bcfbe5b2`.
Training holds out `36,48` and extrapolates from the last training time `27`.

## 2. Later external protocols: train-only preprocessing

### GSE148794

```bash
mkdir -p data/external_geo/GSE148794
for file in GSE148794_tc_ibd.count_table.tsv.gz GSE148794_tc_ibd.metadata.tsv.gz; do
  curl --fail --location --continue-at - \
    --output "data/external_geo/GSE148794/$file" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE148nnn/GSE148794/suppl/$file"
done
python scripts/prepare_external_candidate_bundle.py --accession GSE148794
```

Only the IBD time-course subset is used. Train days `0,3,6,9`, primary holdout
day `12`, secondary holdout day `15`. Read the QC and protocol files before
training; some historical intake prose uses an incorrect hour suffix, while
the recorded external training protocol uses days.

### GSE149622

```bash
mkdir -p data/external_geo/GSE149622
for file in GSE149622_RNA_timecourse_unadjusted_matrix.mtx.gz \
            GSE149622_RNA_timecourse_unadjusted_barcodes.tsv.gz \
            GSE149622_RNA_timecourse_unadjusted_genes.tsv.gz; do
  curl --fail --location --continue-at - \
    --output "data/external_geo/GSE149622/$file" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE149nnn/GSE149622/suppl/$file"
done
python scripts/prepare_external_candidate_bundle.py --accession GSE149622
```

Use RNA time-course samples only; ATAC inputs are excluded. Train days
`0,1,2,3`, hold out day `4`. One pooled sample per timepoint does not support
biological-replicate inference.

### GSE305979

```bash
mkdir -p data/external_geo/GSE305979
for file in GSE305979_day0_raw_counts.h5ad GSE305979_day1-7_demuxed_counts.h5ad; do
  curl --fail --location --continue-at - \
    --output "data/external_geo/GSE305979/$file" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE305nnn/GSE305979/suppl/$file"
done
python scripts/prepare_gse305979_bundle.py --inspect
python scripts/prepare_gse305979_bundle.py --vehicle-pattern '^DMSO$'
```

The recorded vehicle filter is exactly `^DMSO$`. Read
`external_gse305979_vehicle_decision.json` and its QC report. Days are derived
from plate/sample labels because the source hour field mislabels Day 7.
Train days `0,1,2`, primary holdout day `5`, secondary day `7`.
The normalized supplementary matrix is not needed. This protocol has a
single donor; cell/seed uncertainty is not donor-level inference.

### Integrity and training

The converters fit feature selection and scaling on training timepoints only.
Expected input/metadata/gene/matrix hashes are recorded in each
`artifacts/reports/external_gse*_qc.json` file. Compare regenerated manifests
and matrix hashes with those records. A mismatch is a stop condition to
investigate, not permission to silently revise the protocol. The converters
write existing output locations, so retain independent copies when comparing
regenerated manifests.

```bash
python scripts/reproduce_external.py --dry-run
python scripts/reproduce_external.py
python scripts/summarize_release_results.py \
  --external-tag-prefix reproduction_ --skip-primary
```

GSE148794 and GSE149622 use seeds `42,123,2026,7,13`; GSE305979 uses
`42,123,2026`. The driver keeps fresh result tags and refuses overwriting
partial runs. Its numerical summary does not require the historical models.

## 3. Historical external converters

These legacy arms remain separate from the later train-only protocols.
Their numerical results are historical boundary evidence, not a uniform
cross-dataset inference.

### GSE224518

```bash
mkdir -p data/external_geo/GSE224518
curl --fail --location --continue-at - \
  --output data/external_geo/GSE224518/GSE224518_samptag.all_cellannotations_metadata.txt.gz \
  'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE224518&format=file&file=GSE224518_samptag.all_cellannotations_metadata.txt.gz'
curl --fail --location --continue-at - \
  --output data/external_geo/GSE224518/GSE224518_RAW.tar \
  'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE224nnn/GSE224518/suppl/GSE224518_RAW.tar'
tar -tf data/external_geo/GSE224518/GSE224518_RAW.tar | head
python scripts/convert_gse224518_to_leworld_input.py

for seed in 42 123 2026; do
  LEWORLD_DATA_DIR=data/leworld_input_gse224518 \
  LEWORLD_TRAIN_TIMEPOINTS=0,1,3 LEWORLD_HOLDOUT_TIMEPOINTS=5 LEWORLD_MAX_TIME=5 \
  LEWORLD_DATASET_ID=GSE224518 LEWORLD_SOURCE_ACCESSION=GSE224518 \
  LEWORLD_RESULT_TAG=reproduction_gse224518_3to5h LEWORLD_SEED="$seed" \
    python ode_benchmark/train_ode_benchmark.py
done
```

This restores the legacy `3 -> 5 h` arm. The separate `3 -> 8 h` and
balanced-sampling arms have their own result tags and must not be conflated.

### GSE226824

```bash
mkdir -p data/external_geo/GSE226824
curl --fail --location --continue-at - \
  --output data/external_geo/GSE226824/GSE226824_HSPC-all_filtered.h5ad.gz \
  'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE226824&format=file&file=GSE226824_HSPC-all_filtered.h5ad.gz'
python scripts/convert_gse226824_to_leworld_input.py
```

The exploratory arm uses training hours `0,6,12,24` and target `72` as
recorded in the legacy boundary audit. It is not a replacement for the
primary benchmark.

## Troubleshooting

- Missing matrix: finish the relevant download and converter; metadata alone
  is intentionally insufficient for training.
- Interrupted data transfer: repeat with `--continue-at -`.
- Unexpected HTML instead of a data archive: use the GEO Supplementary files
  link and verify gzip/tar/HDF5 content before parsing.
- GSE151658 checksum or manifest mismatch: stop; do not switch gene panels or
  overwrite frozen metadata to suppress the error.
- Some external directories have no original `data_summary.json`; the release
  does not invent one. Their existing QC/protocol JSON provides the contract.
- Exact retraining varies by platform/library stack. Historical source and
  checkpoint hashes are not proof of a byte-identical replay by this release.
