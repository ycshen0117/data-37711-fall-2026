"""Q2: exhaustive, validation-only model selection for white wine."""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SEED = 37711
URL = 'https://archive.ics.uci.edu/static/public/186/wine+quality.zip'


def run(data_dir: Path) -> dict:
    start = time.perf_counter()
    for folder in ('data', 'results', 'figures'):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    supplied = data_dir / 'winequality-white.csv'
    if supplied.exists():
        raw = supplied.read_bytes()
        source = str(supplied.resolve())
    else:
        archive = ROOT / 'data/wine-quality.zip'
        if not archive.exists():
            with urlopen(URL, timeout=60) as response:
                archive.write_bytes(response.read())
        with zipfile.ZipFile(archive) as z:
            raw = z.read('winequality-white.csv')
        source = URL + '#winequality-white.csv'
    expected_hash = '76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836'
    if not supplied.exists() and hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError('The downloaded white-wine source has changed')
    frame = pd.read_csv(io.BytesIO(raw), sep=';')
    if 'quality' not in frame:
        raise ValueError('Expected semicolon-separated wine CSV with quality target')
    X = frame.drop(columns='quality').to_numpy(dtype=float)
    y = frame['quality'].to_numpy(dtype=float)
    indices = np.arange(len(y))
    train, remainder = train_test_split(indices, test_size=.30, random_state=SEED)
    val, test = train_test_split(remainder, test_size=2/3, random_state=SEED)
    np.savez(ROOT / 'results/wine_splits.npz', train=train, validation=val, test=test)
    scaler = StandardScaler().fit(X[train])
    scaled = scaler.transform(X)
    rows = []
    selected = {}

    # Each candidate is fit on the training partition only. Warm-start adds
    # trees for nested forest sizes; it gives the same forest as a fresh fit.
    best_loss = np.inf
    for depth, leaf, features, samples in itertools.product(
            (3, 6, 9, 12, None), (1, 2, 4, 8), (.5, .75, 1.), (.6, .8, 1.)):
        model = RandomForestRegressor(
            max_depth=depth, min_samples_leaf=leaf, max_features=features,
            max_samples=samples, random_state=SEED, n_jobs=1, warm_start=True)
        for trees in (25, 50, 75):
            model.set_params(n_estimators=trees)
            model.fit(X[train], y[train])
            loss = mean_squared_error(y[val], model.predict(X[val]))
            params = dict(n_estimators=trees, max_depth=depth,
                          min_samples_leaf=leaf, max_features=features,
                          max_samples=samples)
            rows.append(dict(model='RandomForest', validation_mse=loss,
                             parameters=json.dumps(params)))
            if loss < best_loss:
                best_loss, selected['RandomForest'] = loss, params
    # 5*4*3*3*3=540 forests; 100*2*3=600 nearest-neighbor models.
    best_loss = np.inf
    for neighbors, weights, p in itertools.product(range(1, 101),
                                                  ('uniform', 'distance'), (1, 2, 3)):
        model = KNeighborsRegressor(n_neighbors=neighbors, weights=weights,
                                     p=p, algorithm='brute', n_jobs=1)
        model.fit(scaled[train], y[train])
        loss = mean_squared_error(y[val], model.predict(scaled[val]))
        params = dict(n_neighbors=neighbors, weights=weights, p=p)
        rows.append(dict(model='KNeighbors', validation_mse=loss,
                         parameters=json.dumps(params)))
        if loss < best_loss:
            best_loss, selected['KNeighbors'] = loss, params
    grid = pd.DataFrame(rows)
    grid.to_csv(ROOT / 'results/wine_grid.csv', index=False)
    answer = dict(source=source, data_sha256=hashlib.sha256(raw).hexdigest(),
                  rows=len(y), features=list(frame.drop(columns='quality').columns),
                  split_sizes=dict(train=len(train), validation=len(val), test=len(test)),
                  seed=SEED, models={})
    predictions = pd.DataFrame(dict(row=test, observed=y[test]))
    for name in ('RandomForest', 'KNeighbors'):
        params = selected[name]
        if name == 'RandomForest':
            model = RandomForestRegressor(**params, random_state=SEED, n_jobs=1)
            features = X
        else:
            model = KNeighborsRegressor(**params, algorithm='brute', n_jobs=1)
            features = scaled
        model.fit(features[train], y[train])
        pred = model.predict(features[test])
        predictions[name] = pred
        count = int((grid.model == name).sum())
        assert count > len(val), 'Each model class must exceed validation size'
        answer['models'][name] = dict(
            candidates=count, parameters=params,
            train_mse=float(mean_squared_error(y[train], model.predict(features[train]))),
            validation_mse=float(mean_squared_error(y[val], model.predict(features[val]))),
            test_mse=float(mean_squared_error(y[test], pred)))
    mean = y[train].mean()
    answer['mean_baseline_test_mse'] = float(mean_squared_error(y[test], np.full(len(test), mean)))
    answer['runtime_seconds'] = time.perf_counter() - start
    predictions.to_csv(ROOT / 'results/wine_test_predictions.csv', index=False)
    (ROOT / 'results/wine_metrics.json').write_text(json.dumps(answer, indent=2)+'\n')
    fig, ax = plt.subplots(figsize=(6, 3.3))
    for name in ('RandomForest', 'KNeighbors'):
        losses = np.sort(grid.loc[grid.model == name, 'validation_mse'])
        ax.plot(np.arange(1, len(losses)+1), losses, label=name)
    ax.set(xlabel='Candidate rank within model class', ylabel='Validation MSE')
    ax.legend(); fig.tight_layout()
    fig.savefig(ROOT / 'figures/wine_search.pdf'); plt.close(fig)
    return answer
