#!/usr/bin/env python3
"""Read-only, matrix-free summary of tagged validation artifacts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'artifacts/results/ode_benchmark'
ARMS = {
    'GSE148794': ('ext_gse148794_v2', (42, 123, 2026, 7, 13), (12, 15)),
    'GSE149622': ('ext_gse149622_v2', (42, 123, 2026, 7, 13), (4,)),
    'GSE305979': ('ext_gse305979_control_v2', (42, 123, 2026), (5, 7)),
}


def safe_fragment(value: str) -> str:
    if value and (not re.fullmatch(r'[A-Za-z0-9._-]+', value) or value in {'.', '..'}):
        raise ValueError('unsafe result tag or prefix')
    return value


def load_run(tag: str, model: str, seed: int) -> dict:
    path = RESULTS / f'validation_results_{model}_{safe_fragment(tag)}_seed{seed}.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    metadata = payload.get('run_metadata', {})
    protocol = metadata.get('protocol', {})
    if protocol.get('seed') != seed or protocol.get('result_tag') != tag:
        raise ValueError(f'{path.name}: tag/seed metadata mismatch')
    if protocol.get('checkpoint_selection', {}).get('holdout_used_for_selection') is not False:
        raise ValueError(f'{path.name}: checkpoint rule lacks explicit holdout exclusion')
    pairs = payload['time_extrapolation']['pairs']
    if any(not math.isfinite(float(row['corr'])) for row in pairs.values()):
        raise ValueError(f'{path.name}: non-finite endpoint')
    return payload


def primary_summary(tag: str) -> dict:
    summary = {}
    for model in ('ode', 'gru'):
        means = []
        for seed in (42, 123, 2026):
            payload = load_run(tag, model, seed)
            pairs = payload['time_extrapolation']['pairs']
            if any(int(pairs[str(t)]['from_time']) != 27 for t in (36, 48)):
                raise ValueError('primary extrapolation must start at time 27')
            means.append(statistics.mean(float(pairs[str(t)]['corr']) for t in (36, 48)))
        summary[model] = {'mean': statistics.mean(means), 'sd': statistics.stdev(means),
                          'seed_composites': means}
    return summary


def external_summary(prefix: str) -> dict:
    summary = {}
    for accession, (tag, seeds, horizons) in ARMS.items():
        summary[accession] = {'seeds': list(seeds), 'horizons': {}}
        for horizon in horizons:
            block = {}
            for model in ('ode', 'gru'):
                values = [float(load_run(prefix + tag, model, seed)['time_extrapolation']
                                ['pairs'][str(horizon)]['corr']) for seed in seeds]
                block[model] = {'mean': statistics.mean(values), 'sd': statistics.stdev(values),
                                'seed_values': values}
            summary[accession]['horizons'][str(horizon)] = block
    return summary


def close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f'aggregate mismatch: {actual} vs {expected}')


def verify_historical(primary: dict | None, external: dict) -> None:
    if primary is not None:
        drift = json.loads((ROOT / 'artifacts/reports/composite_drift_report.json')
                           .read_text(encoding='utf-8'))
        expected = drift['rerun']
        for model in ('ode', 'gru'):
            for metric in ('mean', 'sd'):
                close(primary[model][metric], expected[model][metric])
    recorded = json.loads((ROOT / 'artifacts/reports/external_phase3_aggregate_v2.json')
                          .read_text(encoding='utf-8'))
    for accession, item in external.items():
        for horizon, block in item['horizons'].items():
            for model in ('ode', 'gru'):
                expected = recorded['datasets'][accession]['horizons'][horizon][model]['summary']
                for metric in ('mean', 'sd'):
                    close(block[model][metric], expected[metric])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary-tag', default='primary_internal_freeze')
    parser.add_argument('--external-tag-prefix', default='')
    parser.add_argument('--skip-primary', action='store_true')
    parser.add_argument('--verify', action='store_true',
                        help='compare distributed historical aggregates; not a checkpoint audit')
    args = parser.parse_args()
    if args.verify and (args.primary_tag != 'primary_internal_freeze' or args.external_tag_prefix):
        parser.error('--verify is restricted to the historical distributed tags')
    primary = None if args.skip_primary else primary_summary(safe_fragment(args.primary_tag))
    external = external_summary(safe_fragment(args.external_tag_prefix))
    if args.verify:
        verify_historical(primary, external)
    print(json.dumps({'primary': primary, 'external': external,
                      'historical_aggregate_verification': 'PASS' if args.verify else 'not_requested',
                      'historical_caveat': 'The freeze manifest records GRU SD 0.01677773592521134; '
                      'tagged files and the drift report give 0.01682723193703617. '
                      'Verification uses the drift report and leaves historical numbers unchanged.'},
                     indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
