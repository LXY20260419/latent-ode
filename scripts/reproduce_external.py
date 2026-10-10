#!/usr/bin/env python3
"""Run the recorded external seed panels locally with fresh, safe result tags."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PANELS = {
    'GSE148794': {'times': '0,3,6,9', 'targets': '12,15', 'max_time': '15',
                  'seeds': (42, 123, 2026, 7, 13), 'tag': 'reproduction_ext_gse148794_v2'},
    'GSE149622': {'times': '0,1,2,3', 'targets': '4', 'max_time': '4',
                  'seeds': (42, 123, 2026, 7, 13), 'tag': 'reproduction_ext_gse149622_v2'},
    'GSE305979': {'times': '0,1,2', 'targets': '5,7', 'max_time': '7',
                  'seeds': (42, 123, 2026), 'tag': 'reproduction_ext_gse305979_control_v2'},
}


def run_environment(accession: str, seed: int, threads: int = 4) -> dict[str, str]:
    spec = PANELS[accession]
    # Do not inherit accidental protocol overrides from a previous shell run.
    env = {key: value for key, value in os.environ.items() if not key.startswith('LEWORLD_')}
    env.update({
        'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': str(threads),
        'MKL_NUM_THREADS': str(threads), 'OPENBLAS_NUM_THREADS': str(threads),
        'LEWORLD_DATA_DIR': f'data/leworld_input_{accession.lower()}',
        'LEWORLD_DATASET_ID': accession, 'LEWORLD_SOURCE_ACCESSION': accession,
        'LEWORLD_RESULT_TAG': spec['tag'], 'LEWORLD_SEED': str(seed),
        'LEWORLD_PROTOCOL_ID': 'external-time-extrapolation-v1',
        'LEWORLD_TRAIN_TIMEPOINTS': spec['times'],
        'LEWORLD_HOLDOUT_TIMEPOINTS': spec['targets'],
        'LEWORLD_MAX_TIME': spec['max_time'], 'LEWORLD_EPOCHS': '100',
        'LEWORLD_INTERNAL_VAL_FRACTION': '0.1', 'LEWORLD_TRAIN_MAX_PREDICT_STEPS': '8',
    })
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', default=','.join(PANELS))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    accessions = args.datasets.split(',')
    if not accessions or any(a not in PANELS for a in accessions) or args.threads < 1:
        parser.error('select listed accessions and a positive thread count')
    if len(set(accessions)) != len(accessions):
        parser.error('duplicate accessions are not allowed')
    if not args.dry_run:
        for accession in accessions:
            directory = ROOT / f'data/leworld_input_{accession.lower()}'
            for name in ('expression_matrix.npy', 'metadata.csv', 'gene_names.txt'):
                if not (directory / name).is_file():
                    parser.error(f'restore {directory.relative_to(ROOT) / name} first')
    for accession in accessions:
        spec = PANELS[accession]
        for seed in spec['seeds']:
            outputs = [ROOT / 'artifacts/results/ode_benchmark' /
                       f'validation_results_{model}_{spec["tag"]}_seed{seed}.json'
                       for model in ('ode', 'gru')]
            if all(path.is_file() for path in outputs):
                print(f'SKIP {accession} seed={seed}: tagged result pair already exists')
                continue
            if any(path.exists() for path in outputs):
                parser.error(f'partial tagged run for {accession} seed={seed}; inspect it before retrying')
            print(f'RUN {accession} seed={seed} train={spec["times"]} '
                  f'holdout={spec["targets"]} tag={spec["tag"]}', flush=True)
            if not args.dry_run:
                subprocess.run([sys.executable, str(ROOT / 'ode_benchmark/train_ode_benchmark.py')],
                               cwd=ROOT, env=run_environment(accession, seed, args.threads), check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
