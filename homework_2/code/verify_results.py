"""Check reproduced metrics, split integrity, and ordered Q3 prediction files."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def compare(actual, expected, where=''):
    if isinstance(expected, dict):
        assert actual.keys()==expected.keys(),where
        for k,v in expected.items(): compare(actual[k],v,where+'/'+k)
    elif isinstance(expected,list):
        assert len(actual)==len(expected),where
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,where+'/'+str(i))
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        assert np.isclose(actual,expected,rtol=1e-4,atol=1e-6), (where,actual,expected)
    else:
        assert actual==expected,(where,actual,expected)


def main():
    for name in ['taxi','retail','toy']:
        filename=name+'_metrics.json'
        actual=json.loads((ROOT/'results'/filename).read_text())
        expected=json.loads((ROOT/'reference_results'/filename).read_text())
        compare(actual,expected,name)
        if name!='toy':
            split=np.load(ROOT/'results'/f'{name}_splits.npz')
            ids=np.concatenate([split[k] for k in ['train','validation','test']])
            assert len(ids)==actual['rows'] and np.array_equal(np.sort(ids),np.arange(actual['rows']))
    manifest=json.loads((ROOT/'inputs/manifest.json').read_text())
    for name,info in manifest.items():
        assert hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()==info['sha256']
    for name in ['rks','mlp']:
        filename=name+'_test_predictions.csv'
        pred=np.loadtxt(ROOT/filename,delimiter=',')
        expected=np.loadtxt(ROOT/'reference_results'/filename,delimiter=',')
        assert pred.shape==(200,) and np.isfinite(pred).all()
        assert np.allclose(pred,expected,rtol=1e-4,atol=1e-6)
    toy=json.loads((ROOT/'results/toy_metrics.json').read_text())
    assert toy['rks']['selected_sigma']==min(toy['rks']['sweep'],key=lambda r:r['validation_mse'])['parameter']
    assert toy['mlp']['selected_width']==min(toy['mlp']['sweep'],key=lambda r:r['validation_mse'])['parameter']
    print('PASS: all three metric files reproduce; splits partition the data; Q3 input hashes and both ordered prediction files match.')


if __name__=='__main__':main()
