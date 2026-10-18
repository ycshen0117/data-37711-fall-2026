"""Run the complete HW2 experiments or selected questions."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'results/mplconfig'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--questions', nargs='+', choices=['2', '3', '4', '5'],
                        default=['2', '3', '4', '5'])
    parser.add_argument('--download-data', action='store_true',
                        help='Download the datasets and verify their checksums')
    parser.add_argument('--taxi-target', default='trip_duration')
    parser.add_argument('--taxi-unit', choices=['seconds', 'minutes'], default='minutes')
    args = parser.parse_args()
    if args.download_data:
        from data_sources import retrieve
        retrieve(args.data_dir, args.questions)
    (ROOT / 'results').mkdir(parents=True, exist_ok=True)
    metadata = dict(python=platform.python_version(), platform=platform.platform(),
                    packages={n: importlib.metadata.version(n) for n in
                              ['numpy', 'pandas', 'scipy', 'scikit-learn',
                               'torch', 'pyarrow', 'matplotlib']})
    (ROOT / 'results/environment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    for q in args.questions:
        print('Starting question', q, flush=True)
        if q == '2':
            from wine import run
            result = run(args.data_dir)
        else:
            from supplied_data import run_question
            result = run_question(q, args.data_dir, args.taxi_target, args.taxi_unit)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
