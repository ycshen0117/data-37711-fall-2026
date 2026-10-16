"""Q3-Q5 experiments using the exact instructor-supplied files.

No replacement data are synthesized. Missing files or unsupported schemas fail
explicitly. Taxi durations must have their units confirmed before execution.
"""
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
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score
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


def boosting(data_dir):
    answers = {}
    for name in ['dset0', 'dset1']:
        path = data_dir / (name+'.csv')
        frame = pd.read_csv(path, header=None)
        if frame.isna().any().any():
            raise ValueError(f'{path}: missing values require an explicit policy')
        y, X = frame.iloc[:, 0].to_numpy(), frame.iloc[:, 1:].to_numpy(dtype=float)
        if X.shape[1] == 0:
            raise ValueError('Expected target in first column and covariates after it')
        tr, te = train_test_split(np.arange(len(y)), test_size=.2, random_state=SEED)
        np.savez(ROOT/'results'/f'{name}_splits.npz', train=tr, test=te)
        classification = len(np.unique(y)) == 2
        model_cls = GradientBoostingClassifier if classification else GradientBoostingRegressor
        records, pred = [], pd.DataFrame(dict(row=te, observed=y[te]))
        for count in [1, 10, 100, 300, 1000]:
            model = model_cls(n_estimators=count, learning_rate=.1, max_depth=3,
                              random_state=SEED, subsample=1.)
            model.fit(X[tr], y[tr])
            row = dict(trees=count)
            for label, ix in [('train', tr), ('test', te)]:
                if classification:
                    row[label+'_loss'] = float(log_loss(y[ix], model.predict_proba(X[ix]),
                                                        labels=model.classes_))
                    row[label+'_accuracy'] = float(accuracy_score(y[ix], model.predict(X[ix])))
                else:
                    row[label+'_loss'] = float(mean_squared_error(y[ix], model.predict(X[ix])))
            records.append(row)
            pred[str(count)] = model.predict(X[te])
        pred.to_csv(ROOT/'results'/f'{name}_test_predictions.csv', index=False)
        answers[name] = dict(**provenance(path), rows=len(y), features=X.shape[1],
                             task='binary classification' if classification else 'regression',
                             seed=SEED, train_size=len(tr), test_size=len(te),
                             learning_rate=.1, max_depth=3, records=records)
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        for label in ['train', 'test']:
            ax.plot([r['trees'] for r in records], [r[label+'_loss'] for r in records],
                    'o-', label=label)
        ax.set(xscale='log', xlabel='Number of trees',
               ylabel='Log loss' if classification else 'MSE', title=name)
        ax.legend(); fig.tight_layout(); fig.savefig(ROOT/'figures'/f'{name}.pdf')
        plt.close(fig)
    save('boosting_metrics', answers)
    return answers


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


def retail_diagnostics(y, te, predictions):
    test_y = y[te]
    details = dict(target_quantiles={str(q):float(np.quantile(y,q))
                                    for q in [0,.5,.9,.99,1]},
                   test_max=float(test_y.max()), top10_squared_error_fraction={})
    for mode in ['target','embedding','combined']:
        errors = (test_y-predictions[mode].to_numpy())**2
        details['top10_squared_error_fraction'][mode] = float(np.sort(errors)[-10:].sum()/errors.sum())
    return details


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
    X = frame[['StockCode','CustomerID']].fillna('<missing>')
    y = frame['Quantity'].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError('Quantity contains nonfinite values')
    tr, va, te = split(len(y), 'retail')
    ids = np.zeros((len(y), 2), dtype='int64')
    cards, maps = [], {}
    for j, col in enumerate(X):
        categories = sorted(X.iloc[tr][col].unique())
        mapping = {c:i+1 for i,c in enumerate(categories)}
        ids[:,j] = X[col].map(mapping).fillna(0).to_numpy(dtype='int64')
        cards.append(len(categories)); maps[col] = mapping
    means = target_means(X.iloc[tr], y[tr], X)
    for fit, held in KFold(5, shuffle=True, random_state=SEED).split(tr):
        means[tr[held]] = target_means(X.iloc[tr[fit]], y[tr[fit]], X.iloc[tr[held]])
    feature_scaler = StandardScaler().fit(means[tr])
    means = feature_scaler.transform(means).astype('float32')
    center, scale = float(y[tr].mean()), float(y[tr].std())
    if scale <= 0:
        raise ValueError('Training target has zero variance')
    yz = ((y-center)/scale).astype('float32')
    answer = dict(**provenance(path), rows=len(y), seed=SEED,
                  cardinality={c:int(X[c].nunique()) for c in X},
                  training_cardinality=dict(zip(X.columns, cards)),
                  splits=dict(train=len(tr), validation=len(va), test=len(te)),
                  negative_quantity_count=int((y<0).sum()),
                  dense_onehot_float32_gib=len(y)*sum(cards)*4/1024**3,
                  sparse_onehot_nonzeros=2*len(y), embedding_dimensions=[16,16],
                  target_encoding='5-fold out-of-fold exact means; train-only evaluation map',
                  unknown_test_fraction={c:float(np.mean(ids[te,j]==0)) for j,c in enumerate(X)},
                  models={}, mean_baseline_test_mse=float(np.mean((y[te]-center)**2)))
    predictions = pd.DataFrame(dict(row=te, observed=y[te]))
    for mode in ['target','embedding','combined']:
        torch.manual_seed(SEED)
        model, training = train_network(RetailNet(cards, mode), [ids,means], yz, tr, va,
                                         lambda pred,y: ((pred-y)**2).mean(),
                                         'retail_'+mode, epochs=100)
        # Batched predictions avoid holding all embedding activations in memory.
        values = []
        with torch.no_grad():
            for start in range(0,len(y),8192):
                ix = slice(start,start+8192)
                values.append(model(torch.from_numpy(ids[ix]), torch.from_numpy(means[ix])).numpy())
        pred = np.concatenate(values)*scale+center
        answer['models'][mode] = dict(training=training,
            validation_mse=float(mean_squared_error(y[va],pred[va])),
            test_mse=float(mean_squared_error(y[te],pred[te])),
            test_mae=float(np.mean(np.abs(y[te]-pred[te]))),
            test_r2=float(r2_score(y[te],pred[te])))
        predictions[mode] = pred[te]
    predictions.to_csv(ROOT/'results/retail_test_predictions.csv', index=False)
    answer['diagnostics'] = retail_diagnostics(y, te, predictions)
    save('retail_metrics',answer)
    return answer


def run_question(q, data_dir, taxi_target, taxi_unit):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for d in ['results','figures']:
        (ROOT/d).mkdir(parents=True,exist_ok=True)
    required = {'3':['dset0.csv','dset1.csv'],
                '4':['taxi_trips.parquet'], '5':['retail.csv']}[q]
    missing = [f for f in required if not (data_dir/f).is_file()]
    if missing:
        raise FileNotFoundError('Missing instructor-supplied data: '+', '.join(missing))
    start = time.perf_counter()
    if q == '3':
        answer = boosting(data_dir)
    elif q == '4':
        answer = taxi(data_dir, taxi_target, taxi_unit)
    else:
        answer = retail(data_dir)
    print('Question',q,'runtime seconds:',round(time.perf_counter()-start,2),flush=True)
    return answer
