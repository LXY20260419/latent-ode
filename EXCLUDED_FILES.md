# Release boundary and excluded material

This repository is a metadata-only research release candidate. The full
file-level selection is recorded in [RELEASE_INVENTORY.md](RELEASE_INVENTORY.md).

## Included

- Preprocessing, latent ODE/matched GRU, classical baselines and research scripts.
- Retained research tests and the three required prior configuration files.
- All existing requirement text files.
- All existing selected data manifests: `metadata.csv`, `gene_names.txt`, `data_summary.json`.
- Historical result JSON/CSV/Markdown/PNG artifacts, excluding execution traces.
- Scientific protocol, uncertainty, ablation, mechanism and failure-boundary reports.
- New public restoration, reproduction, privacy and verification documentation.

## Excluded as entire classes

- All journal correspondence and submission-process materials, including identity forms.
- All personal author names, contact addresses, affiliations and credentials.
- All handover, infrastructure, progress, resource-coordination and execution-state records.
- All backups, environments, caches, temporary files and source Git history.
- All execution traces, including trace files with non-standard extensions,
  locks, downloader status tables and orchestration summaries.
- All pretrained checkpoints and model-weight directories.
- All raw GEO downloads and expression matrices, regardless of size.
- Broken historical duplicate tables. Historical data manifests are retained
  as provenance only; they are not alternative training contracts.
- Submission-specific, backup-recovery, watcher and detached orchestration utilities,
  plus tests whose only purpose is those excluded utilities.

## Restore instead of distributing

| Excluded payload | Restoration |
|---|---|
| Primary RDS and all raw supplementary files | Read-only data retrieval from GEO; see DOWNLOAD_AND_RESTORE.md |
| `data/**/expression_matrix.npy` | Run the corresponding preprocessing converter |
| Trained weights | Retrain with a fresh result tag; do not treat excluded weights as bundled |
| Local runtime/environment | Install the supplied requirement set on a compatible platform |

Newly generated local data and training artifacts are ignored by Git. Do not
force-add them to a public release. Private source materials are not part of
the restoration workflow.
