# Privacy and release-boundary scan

Scan date: 2026-10-11. The scan was run from the release root with:

```bash
python scripts/scan_release_privacy.py
find . -type f \( -name '*.log' -o -name '*.log.*' -o -name '*.lock' \
  -o -name '*.pyc' -o -name '*.pt' -o -name '*.npy' -o -name '*.docx' \
  -o -name '*.pdf' \) -print
```

A separate `grep -rIlEi` scan was run for contact addresses, the requested
restricted academic domain, local absolute paths, private runtime references
and service credentials. It returned no filenames. Pattern definitions are
in the scan script; matched values and private identity lists are not copied
into this public report.

## Results

| Category | Matches |
|---|---:|
| Files scanned, including this report | 1,124 |
| Text files scanned, including this report | 1,042 |
| PNG metadata blocks scanned | 82 |
| Contact-address pattern | 0 |
| Requested restricted academic-domain pattern | 0 |
| Local machine absolute-path pattern | 0 |
| Private runtime reference | 0 |
| Token/service-credential patterns | 0 |
| Private-key material | 0 |
| Credential assignments | 0 |
| Personal identity fields | 0 |
| Denied path classes | 0 |
| Execution traces (`*.log`, locks, status files) | 0 |
| Symbolic links | 0 |
| Unexpected binary formats | 0 |
| Personal metadata columns | 0 |
| Git commit contact/path findings | 0 after local commit |

The scan does not print matched secret values. It decodes JSON strings and PNG
text metadata, and separately checks the eventual Git log/config. No approved
real-name denylist was available to this agent, so exact unknown-name absence
cannot be proven by pattern matching alone; author review remains required.
The release contains no author, email, affiliation, credential or source-history
document by construction. The post-commit scan passed, including the new Git
log and local configuration. The repository has a single new root commit,
an anonymous preparation identity with an empty email field, a clean worktree,
and no remote. Placement in the requested sibling directory remains pending
filesystem approval; the environment's approval service did not execute that
directory creation or copy.
