"""Retrieve checksum-pinned historical course data explicitly authorized for HW2.

These are historical student-archived inputs, not confirmed 2026 course files.
Only data files are retrieved; no archived solutions or student code are used.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

SOURCE_COMMIT = 'f386491eef32a7e1c9c0dde29e69430822dcc066'
BASE = f'https://raw.githubusercontent.com/Chenxi1818/foundation_of_ML/{SOURCE_COMMIT}/'
FILES = {
    'dset0.csv': ('a3_decision_tree/data/dset0.csv',
                  'bc6fe805760b5cf1bd6da32ed47b2773e5330679224b8fffb7eecb0021af4519'),
    'dset1.csv': ('a3_decision_tree/data/dset1.csv',
                  'cd33ccaaa01e7b770772de2900f8ad627f104432f0d9e101dc402ccf46c32f75'),
    'taxi_trips.parquet': ('a2_deep_learning/data/taxi_trips.parquet',
                         '6a713446584efe41bb161f77f068331b964a04c84abd3b69164bc2bdb585e3dc'),
    'retail.csv': ('a2_deep_learning/data/retail.csv',
                   '5da6724e4ee820f19e1bdb5076fd24355a10f8e75159062824c85f17ea052597'),
}


def retrieve(data_dir: Path, questions: list[str]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    needed = {'2': [], '3': ['dset0.csv', 'dset1.csv'],
              '4': ['taxi_trips.parquet'], '5': ['retail.csv']}
    records = {}
    for question in questions:
        for filename in needed[question]:
            path, expected = FILES[filename]
            destination = data_dir / filename
            if not destination.exists():
                with urlopen(BASE+path, timeout=120) as response:
                    raw = response.read()
                if hashlib.sha256(raw).hexdigest() != expected:
                    raise ValueError(f'Download checksum mismatch: {filename}')
                destination.write_bytes(raw)
            actual = hashlib.sha256(destination.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'{filename} differs from the historical release; '
                                 'omit --download-historical-data for another supplied version')
            records[filename] = dict(url=BASE+path, sha256=actual)
    (data_dir/'historical_provenance.json').write_text(json.dumps(records, indent=2)+'\n')
