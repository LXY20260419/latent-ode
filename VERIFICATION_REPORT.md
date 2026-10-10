# Release verification report

Prepared on 2026-10-11. This report concerns packaging and reproducibility
contracts, not a newly trained experiment or an unconditional publication approval.

| Check | Outcome |
|---|---|
| Retained research tests plus public helper/privacy tests | 143 tests passed; zero skips, failures or errors |
| Test environment | Existing Python 3.11.16 environment, NumPy 2.4.6, SciPy 1.17.1 |
| Dependency installation / fetched software execution | None |
| Original numerical JSON comparison | 814 files compared; zero numerical field changes |
| Core training entry point | Byte-identical to the source |
| Primary and external artifact re-aggregation | PASS against tagged results and scientific analysis reports |
| External driver dry run | 13 planned fresh-tag runs; correct locked splits and seed panels |
| PNG inspection | 82 images; 74 unique pixel files reviewed; scientific plots, no observed personal identity content |
| Matrix restoration, retraining, original checkpoint audit | Not performed; raw data and weights intentionally omitted |
| Source Git history | Not imported |

Core training entry-point SHA-256:
`8c54ac80420f9a239308028e27267fafc8863087e7579168858e167eb9a69967`.

The first system-Python test attempt lacked scientific dependencies. Tests
were rerun successfully using the existing local reference environment,
without installing packages. The reference environment itself is not included.

The lightweight primary data check correctly reports the absent expression
matrix in this metadata-only release. Restore data before running the full
check or training. Unit tests exercise its data contract using synthetic files.

The freeze manifest's historical GRU SD `0.01677773592521134` differs from
the tagged seed-file/drift-report SD `0.01682723193703617`. This difference is
documented in README.md and the matrix-free result verifier. Historical
numerical values have not been altered to conceal it.

See REDACTION_MANIFEST.json for changed source-document hashes and
PRIVACY_SCAN_REPORT.md for the privacy boundary and scan limitations.
The code license is MIT. The public candidate pauses before final citation
metadata/archive identification and any external publication or upload.
