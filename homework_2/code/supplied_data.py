"""Corrected Q1-Q2 experiments using the verified instructor data."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SEED = 37711


def save(name, obj):
    (ROOT / 'results').mkdir(parents=True, exist_ok=True)
    (ROOT / 'figures').mkdir(parents=True, exist_ok=True)
    (ROOT / 'results' / (name+'.json')).write_text(json.dumps(obj, indent=2)+'\n')


def provenance(path):
    return dict(filename=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def split(n, label):
    tr, rest = train_test_split(np.arange(n), test_size=.3, random_state=SEED)
    va, te = train_test_split(rest, test_size=2/3, random_state=SEED)
    np.savez(ROOT/'results'/f'{label}_splits.npz', train=tr, validation=va, test=te)
    return tr, va, te


class Dense(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, outputs))

    def forward(self, x):
        return self.net(x)


def train_network(model, inputs, y, tr, va, objective, label, epochs=150):
    """Return validation-selected weights; the test partition is never used."""
    torch.manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tensors = [torch.as_tensor(a) for a in inputs]
    target = torch.as_tensor(y, dtype=torch.float32)
    dataset = TensorDataset(*(a[tr] for a in tensors), target[tr])
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True, generator=generator)
    best, best_epoch, stale, state = np.inf, 0, 0, None
    history = []
    for epoch in range(1, epochs+1):
        model.train(); total = 0.
        for batch in loader:
            optimizer.zero_grad()
            loss = objective(model(*batch[:-1]), batch[-1])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{label}: nonfinite loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step(); total += float(loss.detach()) * len(batch[-1])
        model.eval()
        with torch.no_grad():
            vloss = float(objective(model(*(a[va] for a in tensors)), target[va]))
        history.append(dict(epoch=epoch, train_loss=total/len(tr), validation_loss=vloss))
        if vloss < best-1e-6:
            best, best_epoch, stale = vloss, epoch, 0
            state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 15:
            break
    if state is None:
        raise RuntimeError('No finite validation checkpoint')
    model.load_state_dict(state); model.eval()
    torch.save(state, ROOT/'results'/f'{label}_weights.pt')
    pd.DataFrame(history).to_csv(ROOT/'results'/f'{label}_history.csv', index=False)
    plot_history(history, label)
    return model, dict(best_epoch=best_epoch, epochs_run=epoch,
                       best_validation_loss=best, batch_size=1024, learning_rate=.001,
                       optimizer='Adam', weight_decay=.0001, patience=15)


def plot_history(history, label):
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    for group in ['train', 'validation']:
        ax.plot([r['epoch'] for r in history], [r[group+'_loss'] for r in history],
                label=group)
    ax.set(xlabel='Epoch', ylabel='Standardized training objective', title=label)
    ax.legend(); fig.tight_layout()
    fig.savefig(ROOT/'figures'/f'{label}_learning.pdf'); plt.close(fig)


def gaussian_loss(output, y):
    mu = output[:, 0]
    variance = nn.functional.softplus(output[:, 1]) + 1e-4
    return .5 * (np.log(2*np.pi) + torch.log(variance) + (y-mu)**2/variance).mean()


def taxi(data_dir, target, unit):
    path = data_dir/'taxi_trips.parquet'
    frame = pd.read_parquet(path)
    if len(frame.columns) != 8 or target not in frame:
        raise ValueError(f'Expected 8 columns and target {target!r}; got {list(frame.columns)}')
    y = frame[target].to_numpy(dtype=float)
    # All reported losses and probabilities use minutes.
    if unit == 'seconds':
        y = y/60.
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Trip durations must be finite and nonnegative')
    X = frame.drop(columns=target).to_numpy(dtype=float)
    tr, va, te = split(len(y), 'taxi')
    imputer = SimpleImputer(strategy='median').fit(X[tr])
    clean = imputer.transform(X)
    scaler = StandardScaler().fit(clean[tr])
    z = scaler.transform(clean).astype('float32')
    center, scale = float(y[tr].mean()), float(y[tr].std())
    if scale <= 0:
        raise ValueError('Training target has zero variance')
    yz = ((y-center)/scale).astype('float32')
    torch.manual_seed(SEED)
    model, training = train_network(Dense(z.shape[1], 2), [z], yz,
                                     tr, va, gaussian_loss, 'taxi')
    with torch.no_grad():
        output = model(torch.from_numpy(z)).numpy()
    mu = output[:, 0]*scale+center
    variance = (np.logaddexp(0., output[:, 1])+1e-4)*scale**2
    ridge = Ridge(alpha=1.).fit(z[tr], y[tr])
    ridge_mu = ridge.predict(z)
    ridge_var = max(float(np.mean((y[tr]-ridge_mu[tr])**2)), 1e-6)
    candidates = {'constant': (np.full(len(y), center), np.full(len(y), scale**2)),
                  'ridge': (ridge_mu, np.full(len(y), ridge_var)),
                  'heteroscedastic_MLP': (mu, variance)}
    results = {}
    predictions = pd.DataFrame(dict(row=te, observed_minutes=y[te], under45=y[te]<45))
    for name, (m, v) in candidates.items():
        results[name] = {}
        for label, ix in [('validation', va), ('test', te)]:
            prob = ndtr((45-m[ix])/np.sqrt(v[ix]))
            event = (y[ix] < 45).astype(int)
            results[name][label] = dict(
                mse=float(mean_squared_error(y[ix], m[ix])),
                gaussian_nll=float(np.mean(.5*(np.log(2*np.pi*v[ix])+(y[ix]-m[ix])**2/v[ix]))),
                accuracy=float(np.mean((prob >= .5) == event)),
                brier=float(np.mean((prob-event)**2)),
                event_log_loss=float(log_loss(event, np.c_[1-prob, prob], labels=[0, 1])),
                coverage90=float(np.mean(np.abs(y[ix]-m[ix]) <= 1.6448536269514722*np.sqrt(v[ix]))))
        predictions[name+'_mean'] = m[te]
        predictions[name+'_variance'] = v[te]
        predictions[name+'_p_under45'] = ndtr((45-m[te])/np.sqrt(v[te]))
    predictions.to_csv(ROOT/'results/taxi_test_predictions.csv', index=False)
    answer = dict(**provenance(path), rows=len(y), seed=SEED, target=target,
                  input_unit=unit, reported_unit='minutes',
                  features=list(frame.drop(columns=target).columns),
                  splits=dict(train=len(tr), validation=len(va), test=len(te)),
                  architecture=[z.shape[1],64,64,2], training=training, models=results,
                  test_event_prevalence=float(np.mean(y[te]<45)),
                  majority_accuracy=float(max(np.mean(y[te]<45), np.mean(y[te]>=45))))
    save('taxi_metrics', answer)
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    order = np.argsort(mu[te]); take = order[::max(1, len(order)//500)]
    ax.scatter(mu[te][take], y[te][take], s=5, alpha=.4)
    ax.set(xlabel='Predicted mean (minutes)', ylabel='Observed duration (minutes)')
    fig.tight_layout(); fig.savefig(ROOT/'figures/taxi_predictions.pdf'); plt.close(fig)
    return answer


def target_means(train_x, train_y, x):
    """Exact training-category means; unseen categories receive the training mean."""
    result = np.empty((len(x), 2), dtype='float32')
    for j, col in enumerate(train_x.columns):
        table = pd.DataFrame({'category': train_x[col].to_numpy(), 'y': train_y})
        means = table.groupby('category')['y'].mean()
        result[:, j] = x[col].map(means).fillna(float(np.mean(train_y)))
    return result


class RetailNet(nn.Module):
    def __init__(self, cardinalities, mode):
        super().__init__()
        self.mode = mode
        self.embeddings = nn.ModuleList([nn.Embedding(c+1, 16, padding_idx=0)
                                          for c in cardinalities])
        dimensions = (32 if mode != 'target' else 0)+(2 if mode != 'embedding' else 0)
        self.head = nn.Sequential(nn.Linear(dimensions, 64), nn.ReLU(),
                                  nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))

    def forward(self, ids, means):
        components = []
        if self.mode != 'target':
            components += [embedding(ids[:, j]) for j, embedding in enumerate(self.embeddings)]
        if self.mode != 'embedding':
            components.append(means)
        return self.head(torch.cat(components, dim=1)).squeeze(1)


def retail(data_dir):
    path = data_dir/'retail.csv'
    frame = pd.read_csv(path, dtype={'StockCode':str, 'CustomerID':str})
    if list(frame.columns) != ['Quantity', 'StockCode', 'CustomerID'] or frame.isna().any().any():
        raise ValueError('Expected complete Quantity, StockCode, CustomerID data')
    X = frame[['StockCode','CustomerID']]
    y = frame['Quantity'].to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError('Quantity must be finite and positive')
    tr, va, te = split(len(y), 'retail')
    ids = np.zeros((len(y), 2), dtype='int64')
    cards, maps = [], {}
    for j, col in enumerate(X):
        categories = sorted(X.iloc[tr][col].unique())
        mapping = {c:i+1 for i,c in enumerate(categories)}
        ids[:,j] = X[col].map(mapping).fillna(0).to_numpy(dtype='int64')
        cards.append(len(categories)); maps[col] = mapping
    # The encoding is the mean of raw Quantity, as requested, not mean(log Quantity).
    means = target_means(X.iloc[tr], y[tr], X)
    for fit, held in KFold(5, shuffle=True, random_state=SEED).split(tr):
        means[tr[held]] = target_means(X.iloc[tr[fit]], y[tr[fit]], X.iloc[tr[held]])
    means = np.log1p(means)
    feature_scaler = StandardScaler().fit(means[tr])
    means = feature_scaler.transform(means).astype('float32')
    logy = np.log1p(y)
    center, scale = float(logy[tr].mean()), float(logy[tr].std())
    yz = ((logy-center)/scale).astype('float32')
    answer = dict(**provenance(path), rows=len(y), seed=SEED,
                  cardinality={c:int(X[c].nunique()) for c in X},
                  training_cardinality=dict(zip(X.columns, cards)),
                  splits=dict(train=len(tr), validation=len(va), test=len(te)),
                  target_quantiles={str(q):float(np.quantile(y,q)) for q in [0,.5,.9,.99,1]},
                  dense_onehot_float32_gib=len(y)*sum(cards)*4/1024**3,
                  sparse_onehot_nonzeros=2*len(y), embedding_dimensions=[16,16],
                  target_encoding='5-fold out-of-fold raw Quantity means; log1p then training standardization; train-only evaluation map',
                  loss='MSE on training-standardized log1p(Quantity), equivalent to MSLE up to fixed scale',
                  prediction='expm1 predicted log1p target, clipped at zero; estimates log-scale conditional location, not raw conditional mean',
                  unknown_test_fraction={c:float(np.mean(ids[te,j]==0)) for j,c in enumerate(X)},
                  models={}, baselines={})
    def scores(pred, ix):
        return dict(msle=float(np.mean((np.log1p(pred[ix])-logy[ix])**2)),
                    mse=float(mean_squared_error(y[ix],pred[ix])),
                    mae=float(np.mean(np.abs(y[ix]-pred[ix]))),
                    r2=float(r2_score(y[ix],pred[ix])))
    for name, val in [('log_constant',np.expm1(center)), ('raw_mean',y[tr].mean())]:
        pred=np.full(len(y),val)
        answer['baselines'][name] = dict(prediction=float(val), validation=scores(pred,va), test=scores(pred,te))
    predictions = pd.DataFrame(dict(row=te, observed=y[te]))
    for mode in ['target','embedding','combined']:
        torch.manual_seed(SEED)
        model, training = train_network(RetailNet(cards, mode), [ids,means], yz, tr, va,
                                         lambda pred,y: ((pred-y)**2).mean(),
                                         'retail_'+mode, epochs=100)
        values = []
        with torch.no_grad():
            for start in range(0,len(y),8192):
                ix = slice(start,start+8192)
                values.append(model(torch.from_numpy(ids[ix]), torch.from_numpy(means[ix])).numpy())
        predlog = np.concatenate(values).astype(float)*scale+center
        pred = np.maximum(np.expm1(predlog),0.)
        if not np.isfinite(pred).all():
            raise FloatingPointError('Nonfinite retail predictions')
        answer['models'][mode] = dict(training=training, validation=scores(pred,va), test=scores(pred,te))
        predictions[mode] = pred[te]
        print('Retail',mode,answer['models'][mode],flush=True)
    predictions.to_csv(ROOT/'results/retail_test_predictions.csv', index=False)
    # Save the training-only preprocessing needed to apply the restored weights.
    torch.save(dict(category_maps=maps, feature_mean=feature_scaler.mean_,
                    feature_scale=feature_scaler.scale_, log_target_mean=center,
                    log_target_scale=scale),ROOT/'results/retail_preprocessing.pt')
    save('retail_metrics',answer)
    return answer
