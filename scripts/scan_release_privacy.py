#!/usr/bin/env python3
"""Read-only release scan; report counts and relative filenames, never matched values."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import struct
import zlib

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'contact_address': re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
    'restricted_domain': re.compile(re.escape('jhun' + '.edu' + '.cn'), re.I),
    'machine_path': re.compile(r'/(?:home|mnt|opt|tmp)/', re.I),
    'private_runtime_reference': re.compile('open' + 'code', re.I),
    'service_credential': re.compile(r'\bsk' + r'-[A-Za-z0-9_-]{12,}|\bgh[pousr]_[A-Za-z0-9]{20,}|'
                                    r'\bgithub_pat_[A-Za-z0-9_]{20,}|\bAKIA[A-Z0-9]{16}\b'),
    'private_key_material': re.compile(r'-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----'),
    'credential_assignment': re.compile(r'\b(?:api[_-]?key|access[_-]?token|secret|password)\b'
                                        r'\s*[=:]\s*[\"\x27]?[A-Za-z0-9_/+.-]{16,}', re.I),
    'personal_identity_field': re.compile(r'(?:作者姓名|真实姓名|姓名|邮箱)\s*[:：]\s*\S+|'
                                        r'[\"\x27](?:author_name|full_name|contact_name|'
                                        r'email_address|affiliation)[\"\x27]\s*:', re.I),
}
TEXT_SUFFIXES = {'.py', '.sh', '.json', '.md', '.csv', '.txt', '.gmt'}
DENIED_PARTS = {'manuscript', '26.7.31-转投', 'backups', '.venv', 'venv', 'tmp',
                '__pycache__', '.tools', '.codex', '.agents', 'logs', 'log'}
PERSONAL_METADATA_FIELDS = {'author', 'author_name', 'full_name', 'contact_name',
                            'email', 'email_address', 'affiliation', 'donor_name',
                            'patient_name', 'subject_name', '姓名', '邮箱'}


def png_text(data: bytes) -> str:
    """Decode PNG text chunks without imaging dependencies or pixel processing."""
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('invalid PNG signature')
    offset, values = 8, []
    while offset + 12 <= len(data):
        size = struct.unpack('>I', data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + size]
        if kind == b'tEXt':
            values.append(payload.decode('latin1'))
        elif kind == b'zTXt':
            keyword, compressed = payload.split(b'\0', 1)
            values.append(keyword.decode('latin1') + '=' +
                          zlib.decompress(compressed[1:]).decode('latin1'))
        elif kind == b'iTXt':
            keyword, body = payload.split(b'\0', 1)
            flag, method = body[0], body[1]
            _, _, text = body[2:].split(b'\0', 2)
            if flag == 1:
                text = zlib.decompress(text)
            values.append(keyword.decode('latin1') + '=' + text.decode('utf-8'))
        offset += size + 12
    return '\n'.join(values)


def scan(root: Path, personal_terms: list[str] | None = None) -> dict:
    counts = {label: 0 for label in PATTERNS}
    counts.update({'blocked_path': 0, 'execution_trace': 0, 'symlink': 0,
                   'unexpected_format': 0, 'personal_metadata_column': 0,
                   'supplied_identity_denylist': 0, 'git_contact_or_path': 0})
    files, text_files, images, bytes_total, findings = 0, 0, 0, 0, []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if '.git' in relative.parts:
            # Git binary objects are covered by the pre-commit scan; scan its
            # decompressed identity/config via the separate Git checks below.
            continue
        if path.is_symlink():
            counts['symlink'] += 1
            findings.append({'category': 'symlink', 'path': relative.as_posix()})
            continue
        if not path.is_file():
            continue
        files += 1
        bytes_total += path.stat().st_size
        lowered = relative.as_posix().lower()
        if (any(part.lower() in DENIED_PARTS or part.lower().endswith('_logs')
                for part in relative.parts) or lowered.startswith('artifacts/models/')
                or any(fragment in lowered for fragment in ('handoff', '交接', '作者信息'))
                or path.name.lower() in {'auth.json', '.bashrc', '.env'}
                or path.suffix.lower() in {'.pem', '.key', '.docx', '.pdf', '.pyc', '.pt', '.npy'}):
            counts['blocked_path'] += 1
            findings.append({'category': 'blocked_path', 'path': relative.as_posix()})
            # Never open potentially credential-bearing disallowed files.
            continue
        if re.search(r'\.log(?:\.|$)', path.name, re.I) or path.suffix.lower() in {'.lock', '.status'}:
            counts['execution_trace'] += 1
            findings.append({'category': 'execution_trace', 'path': relative.as_posix()})
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {'.gitignore', 'LICENSE'}:
            text = path.read_text(encoding='utf-8')
            text_files += 1
        elif path.suffix.lower() == '.png':
            text = png_text(path.read_bytes())
            images += 1
        else:
            counts['unexpected_format'] += 1
            findings.append({'category': 'unexpected_format', 'path': relative.as_posix()})
            continue
        # For JSON, also inspect decoded strings to catch escaped paths.
        if path.suffix.lower() == '.json':
            text += '\n' + json.dumps(json.loads(text), ensure_ascii=False)
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                counts[label] += 1
                findings.append({'category': label, 'path': relative.as_posix()})
        if personal_terms and any(term.casefold() in text.casefold() for term in personal_terms):
            counts['supplied_identity_denylist'] += 1
            findings.append({'category': 'supplied_identity_denylist', 'path': relative.as_posix()})
        if path.name == 'metadata.csv':
            with path.open(encoding='utf-8', newline='') as handle:
                headers = next(csv.reader(handle))
            if {item.casefold() for item in headers} & PERSONAL_METADATA_FIELDS:
                counts['personal_metadata_column'] += 1
                findings.append({'category': 'personal_metadata_column', 'path': relative.as_posix()})
    import subprocess
    if (root / '.git').is_dir():
        checks = [
            ['git', '-C', str(root), 'log', '--all', '--format=%an <%ae>%n%cn <%ce>'],
            ['git', '-C', str(root), 'config', '--local', '--list'],
        ]
        for command in checks:
            result = subprocess.run(command, text=True, capture_output=True, check=True)
            if PATTERNS['contact_address'].search(result.stdout) or PATTERNS['machine_path'].search(result.stdout):
                counts['git_contact_or_path'] += 1
                findings.append({'category': 'git_contact_or_path', 'path': '.git'})
    return {'status': 'PASS' if not any(counts.values()) else 'FAIL',
            'files_scanned': files, 'text_files_scanned': text_files,
            'png_metadata_scanned': images, 'payload_bytes': bytes_total,
            'categories': counts, 'findings': findings,
            'supplied_name_denylist_used': bool(personal_terms),
            'limit': 'Pattern scans cannot prove absence of unknown personal names. '
                     'Author approval is required; use an external approved name denylist for exact matching.'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--personal-name-denylist', type=Path,
                        help='optional approved names, one per line, kept outside the public release')
    args = parser.parse_args()
    terms = None
    if args.personal_name_denylist:
        if args.personal_name_denylist.resolve().is_relative_to(args.root.resolve()):
            parser.error('the personal-name list must stay outside the release')
        terms = [line.strip() for line in args.personal_name_denylist.read_text(encoding='utf-8')
                 .splitlines() if line.strip()]
    report = scan(args.root.resolve(), terms)
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
