"""Reproduce all three problems in the corrected HW2."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'results/mplconfig'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--questions', nargs='+', choices=['1', '2', '3'], default=['1', '2', '3'])
    parser.add_argument('--download-data', action='store_true')
    args = parser.parse_args()
    if args.download_data:
        from data_sources import retrieve
        retrieve(args.data_dir, args.questions)
    (ROOT / 'results').mkdir(parents=True, exist_ok=True)
    metadata = dict(python=platform.python_version(), platform=platform.platform(),
                    packages={n: importlib.metadata.version(n) for n in
                              ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'pyarrow', 'matplotlib']})
    (ROOT / 'results/environment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from supplied_data import taxi, retail
    from toy import run as toy
    for q in args.questions:
        start = time.perf_counter()
        print('Starting corrected question', q, flush=True)
        for d in ['results', 'figures']:
            (ROOT/d).mkdir(parents=True, exist_ok=True)
        if q == '1':
            result = taxi(args.data_dir, 'trip_duration', 'minutes')
        elif q == '2':
            result = retail(args.data_dir)
        else:
            result = toy(args.data_dir)
        print(json.dumps(result, indent=2), flush=True)
        print('Runtime seconds:', round(time.perf_counter()-start, 2), flush=True)


if __name__ == '__main__':
    main()
