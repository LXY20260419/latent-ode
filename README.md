# latent-ode: Protocol-Controlled Latent ODE Benchmark

A protocol-controlled evaluation framework for temporal extrapolation of
single-cell RNA expression with latent ordinary differential equations (ODEs).
It compares latent ODE and matched GRU models with classical and population
persistence baselines under explicitly recorded train/holdout timepoints,
random seeds, checkpoint rules, and gene panels. Horizon-specific outcomes,
negative controls, uncertainty, and failure boundaries are reported rather
than selecting only favorable endpoints.

This is a standalone, anonymized research release candidate. Large expression
matrices, raw data, trained weights, execution traces, private publication
materials and source-repository history are not included. The repository has
one new local history and no remote. Public release awaits author approval of
licensing, citation and the final privacy review.

## Environment

Python 3.11 is the recorded reference version. For a fresh environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-ode.txt
```

The optional `requirements-lock-py311.txt` records the complete environment
resolved for the reference experiments, including platform-specific CUDA
packages. Use it only on a compatible platform:

```bash
python -m pip install -r requirements-lock-py311.txt
```

The broad requirements are more portable but do not imply bit-identical
retraining. No dependency installation is performed by the release preparation.
The optional `baselines/run_scvelo_baseline.py` additionally requires a compatible
scVelo installation and suitable spliced/unspliced inputs; it is not part of
the core latent-ODE reproduction path.

## Data

The included `data/` directories contain only existing `metadata.csv`,
`gene_names.txt` and `data_summary.json` manifests. Metadata contains public
cell/sample/library identifiers and experimental annotations, not personal
contact information. Matrix row and gene order must match the manifests.
The earlier `leworld_input.pre_summary_contract_*` metadata snapshot is
retained for provenance only; use the current `data/leworld_input/` manifests
for primary-data reconstruction.

| GEO accession | Role | Prepared directory |
|---|---|---|
| GSE151658 | Primary LPS time course; frozen cell/gene contract | `data/leworld_input/` |
| GSE148794 | External colitis time-course boundary evaluation | `data/leworld_input_gse148794/` |
| GSE149622 | External skin-inflammation RNA time course | `data/leworld_input_gse149622/` |
| GSE305979 | External vehicle-control trajectory | `data/leworld_input_gse305979/` |
| GSE224518 | Historical LPS macrophage external protocols | `data/leworld_input_gse224518/` |
| GSE226824 | Historical exploratory HSPC protocol | `data/leworld_input_gse226824/` |

See [DOWNLOAD_AND_RESTORE.md](DOWNLOAD_AND_RESTORE.md) for exact supplementary
filenames, GEO locations, preprocessing commands and integrity checks. Downloads
are data only; no fetched executable is required. Data-source reuse conditions
are independent of this repository's code license.

## End-to-end reproduction

Run commands from this repository root. First install the requirements and
restore GSE151658 as described in the download guide:

```bash
# Run after placing the specified GSE151658 RDS file in data/raw/.
python preprocess/preprocess_full_v7_1.4g_integrated_full.py --verify-inputs-only
python preprocess/preprocess_full_v7_1.4g_integrated_full.py
bash scripts/link_data.sh
python scripts/run_experiment.py check
python -m unittest discover -s tests -v
```

Use a new result tag so the distributed historical artifacts remain unchanged:

```bash
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export LEWORLD_DATA_DIR=data/leworld_input
export LEWORLD_DATASET_ID=internal_lps_primary
export LEWORLD_SOURCE_ACCESSION=GSE151658
export LEWORLD_PROTOCOL_ID=paper-v1-freeze-internal-population-extrapolation
export LEWORLD_TRAIN_TIMEPOINTS=0,1,4,16,27
export LEWORLD_HOLDOUT_TIMEPOINTS=36,48
export LEWORLD_MAX_TIME=48
export LEWORLD_EPOCHS=100
export LEWORLD_RESULT_TAG=reproduction_primary

for seed in 42 123 2026; do
  python scripts/run_experiment.py ode --seed "$seed" --result-tag "$LEWORLD_RESULT_TAG"
done

# Recompute endpoint-aligned classical baselines and paired comparisons.
python scripts/run_experiment.py stats_extrapolation
python scripts/reviewer_controls.py uncertainty --result-tag "$LEWORLD_RESULT_TAG" \
  --output artifacts/reports/reproduction_primary_uncertainty.json
```

Each ODE run also trains the matched GRU. New tagged validation files, training
histories and biological analyses are written under `artifacts/results/`; new
checkpoints are generated locally and remain ignored. The lightweight `check`
command is specific to the frozen GSE151658 contract, not a generic external
dataset validator. Without restored matrices, its missing-matrix error is
expected. Do not use untagged filenames as primary evidence.

For external reproduction, restore and convert the inputs in the download
guide, then run the supplied local research driver:

```bash
python scripts/reproduce_external.py --datasets GSE148794,GSE149622,GSE305979
```

It uses fresh `reproduction_*` tags, the recorded splits and seed panels, and
does not replace the included external results. For purely artifact-based
inspection, no matrix restoration or training is needed:

```bash
python scripts/summarize_release_results.py --verify
```

## Research scope and historical results

The primary endpoint is population-mean expression Pearson correlation across
genes, not prediction of tracked individual cells. Seeds describe stochastic
training repeats; they are not biological replicates. Historical GSE224518
and GSE226824 converters have a different preprocessing contract and must not
be pooled with the later train-only external converters as one uniform claim.

Read the tagged `primary_internal_freeze` artifacts and
`artifacts/reports/composite_drift_report.json` before interpreting earlier
records. Historical summary values are preserved, not silently promoted.
The supersession registry identifies stale numbers. The random-prior aggregate
contains the completed 20-control subset; the 100-control configuration plan
is included for provenance, not described as a completed experiment.

One retained historical inconsistency is explicit: the freeze manifest records
GRU composite SD `0.01677773592521134`, while the tagged seed artifacts and
drift report yield `0.01682723193703617`. Release verification uses the tagged
artifacts and drift report; it does not edit historical numerical records.

Privacy redactions change some historical document bytes and pointer fields,
but not scientific numerical JSON fields. Old source/checkpoint hashes remain
historical records: the release does not contain the original weights or all
private evidence. A full legacy provenance audit therefore cannot pass in a
metadata-only checkout. Use `REDACTION_MANIFEST.json` and the result summary
verifier for release-level checks. Full numerical retraining was not performed
during packaging.

## Layout

```text
latent-ode/
├── preprocess/              # Primary-data reconstruction and frozen contract
├── ode_benchmark/           # Latent ODE and matched GRU training
├── baselines/               # Classical and optional velocity baselines
├── configs/                 # Required gene/pathway/TF prior configurations
├── scripts/                 # Research preparation, evaluation and aggregation
├── tests/                   # Retained research unit and contract tests
├── data/                    # Metadata and gene manifests only
├── artifacts/
│   ├── results/             # Historical numerical/biological results and plots
│   └── reports/             # Scientific protocols and analysis reports
├── requirements*.txt
├── DOWNLOAD_AND_RESTORE.md
├── EXCLUDED_FILES.md
├── RELEASE_INVENTORY.md
├── REDACTION_MANIFEST.json
├── PRIVACY_SCAN_REPORT.md
├── VERIFICATION_REPORT.md
└── LICENSE
```

## Citation placeholder

```bibtex
@misc{latent_ode,
  title = {latent-ode: Protocol-Controlled Latent ODE Benchmark},
  author = {The latent-ode authors},
  year = {2026},
  note = {Citation metadata and persistent identifier pending approval}
}
```

No personal names, affiliations or contact addresses are supplied in this
placeholder. Replace it with approved publication/archive metadata before
public distribution.

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE).
The license notice uses a collective author designation and contains no
personal names or contact details. The code license does not replace the reuse
conditions for GEO data or other third-party resources. Nothing has been pushed
or uploaded by this preparation.
