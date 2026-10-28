"""Retrieve verified Q1/Q2 bytes and copy bundled, exact Q3 splits."""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = 'f386491eef32a7e1c9c0dde29e69430822dcc066'
BASE = f'https://raw.githubusercontent.com/Chenxi1818/foundation_of_ML/{SOURCE_COMMIT}/'
FILES = {
    'taxi_trips.parquet': ('a2_deep_learning/data/taxi_trips.parquet',
                         '6a713446584efe41bb161f77f068331b964a04c84abd3b69164bc2bdb585e3dc'),
    'retail.csv': ('a2_deep_learning/data/retail.csv',
                   '5da6724e4ee820f19e1bdb5076fd24355a10f8e75159062824c85f17ea052597'),
}


def retrieve(data_dir: Path, questions: list[str]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    needed = {'1': ['taxi_trips.parquet'], '2': ['retail.csv'], '3': ['train.csv','val.csv','test.csv']}
    records = {}
    manifest = json.loads((ROOT/'inputs/manifest.json').read_text())
    for question in questions:
        for filename in needed[question]:
            destination = data_dir/filename
            if filename in FILES:
                path, expected = FILES[filename]
                source = BASE+path
                if not destination.exists():
                    with urlopen(source, timeout=120) as response:
                        raw = response.read()
                    if hashlib.sha256(raw).hexdigest() != expected:
                        raise ValueError(f'Download checksum mismatch: {filename}')
                    destination.write_bytes(raw)
            else:
                expected = manifest[filename]['sha256']
                source = 'bundled inputs/'+filename
                if not destination.exists():
                    shutil.copy2(ROOT/'inputs'/filename, destination)
            actual = hashlib.sha256(destination.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'{filename} does not match the verified corrected assignment')
            records[filename] = dict(source=source, sha256=actual)
    (data_dir/'data_provenance.json').write_text(json.dumps(records,indent=2)+'\n')
