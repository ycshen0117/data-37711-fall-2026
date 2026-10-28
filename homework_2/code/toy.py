"""Corrected Q3: paired sinusoidal random features and a one-hidden-layer MLP."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.linalg import solve
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataloader import HW2Dataset
from supplied_data import provenance

ROOT = Path(__file__).resolve().parents[1]
SEED = 37711


class RandomKitchenSinks:
    def __init__(self, D=1000, sigma=1., ridge_lambda=.01):
        if D <= 0 or D % 2 or sigma <= 0 or ridge_lambda <= 0:
            raise ValueError('D must be positive and even; sigma and lambda must be positive')
        self.D, self.sigma, self.ridge_lambda = D, sigma, ridge_lambda

    def features(self, X):
        phase = X @ self.W.T + self.b
        return np.stack([np.sin(phase), np.cos(phase)], axis=-1).reshape(len(X), self.D)

    def fit(self, X, y):
        rng = np.random.default_rng(SEED)
        self.W = rng.standard_normal((self.D//2, X.shape[1]))/self.sigma
        self.b = rng.uniform(0, 2*np.pi, self.D//2)
        phi = self.features(X)
        self.phi_mean, self.y_mean = phi.mean(axis=0), float(y.mean())
        A = phi-self.phi_mean
        # Dual ridge: min ||y-c-Phi beta||^2 + lambda ||beta||^2.
        # Centering leaves the intercept unpenalized.
        self.coef_ = A.T @ solve(A @ A.T + self.ridge_lambda*np.eye(len(X)),
                                y-self.y_mean, assume_a='pos')
        return self

    def predict(self, X):
        return self.y_mean + (self.features(X)-self.phi_mean) @ self.coef_


class MLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, X):
        return self.net(X).squeeze(-1)

    def predict(self, X):
        with torch.no_grad():
            return self(self._tensor(X)).numpy()*self.y_scale+self.y_mean

    @staticmethod
    def _tensor(X):
        return torch.as_tensor(X, dtype=torch.float32)


def fit_mlp(X, y, Xval, yval, hidden, fixed_epochs=None):
    torch.manual_seed(SEED)
    model = MLP(hidden)
    model.y_mean, model.y_scale = float(y.mean()), float(y.std())
    X, y = model._tensor(X), model._tensor((y-model.y_mean)/model.y_scale)
    xv, yv = model._tensor(Xval), model._tensor((yval-model.y_mean)/model.y_scale)
    opt = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=.0001)
    gen = torch.Generator().manual_seed(SEED)
    best, state, best_epoch, stale = np.inf, None, 0, 0
    history = []
    for epoch in range(1, (fixed_epochs or 5000)+1):
        model.train()
        order = torch.randperm(len(y), generator=gen)
        for ids in order.split(128):
            opt.zero_grad()
            loss = ((model(X[ids])-y[ids])**2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite MLP loss')
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            t = float(((model(X)-y)**2).mean())*model.y_scale**2
            v = float(((model(xv)-yv)**2).mean())*model.y_scale**2
        history.append(dict(epoch=epoch, train_mse=t, validation_mse=v))
        if v < best-1e-8:
            best, best_epoch, stale = v, epoch, 0
            state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if fixed_epochs is None and stale >= 300:
            break
    if fixed_epochs is None:
        model.load_state_dict(state)
    model.eval()
    return model, history, best_epoch


def mse(y, pred):
    return float(np.mean((y-pred)**2))


def sweep_plots(records, name, xlabel):
    for group in ['train', 'validation']:
        fig, ax = plt.subplots(figsize=(4.8, 2.8))
        ax.plot([r['parameter'] for r in records], [r[group+'_mse'] for r in records], 'o-')
        ax.set(xscale='log', yscale='log', xlabel=xlabel, ylabel=group.title()+' MSE')
        ax.set_xticks([r['parameter'] for r in records], labels=[f"{r['parameter']:g}" for r in records])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(ROOT/'figures'/f'{name}_{group}_sweep.pdf'); plt.close(fig)


def run(data_dir):
    paths = [data_dir/(s+'.csv') for s in ['train', 'val', 'test']]
    frames = [pd.read_csv(p) for p in paths]
    if [list(f.columns) for f in frames] != [['x1', 'x2', 'y'], ['x1', 'x2', 'y'], ['x1', 'x2']]:
        raise ValueError('Expected labeled train/val and unlabeled two-column test')
    ds = HW2Dataset.from_csv(*map(str, paths))
    x, y, xv, yv, xt = ds.train.X, ds.train.y, ds.val.X, ds.val.y, ds.test.X
    if any(not np.isfinite(f.to_numpy()).all() for f in frames):
        raise ValueError('Nonfinite toy data')
    if any(np.abs(z).max() > 3 for z in [x, xv, xt]):
        raise ValueError('Toy covariates outside [-3,3]')
    # Exactly one swept parameter for each model; randomness is held fixed.
    rks_records = []
    for sigma in [.1, .2, .4, .7, 1., 1.5, 2., 3., 5.]:
        model = RandomKitchenSinks(sigma=sigma).fit(x, y)
        rks_records.append(dict(parameter=sigma, train_mse=mse(y, model.predict(x)),
                                validation_mse=mse(yv, model.predict(xv))))
    best_rks = min(rks_records, key=lambda r:r['validation_mse'])
    mlp_records = []
    for width in [16, 32, 64, 128, 256]:
        model, hist, epoch = fit_mlp(x, y, xv, yv, width)
        pd.DataFrame(hist).to_csv(ROOT/'results'/f'toy_mlp_width{width}_history.csv', index=False)
        row = dict(parameter=width, train_mse=mse(y,model.predict(x)),
                   validation_mse=mse(yv,model.predict(xv)), best_epoch=epoch, epochs_run=len(hist))
        mlp_records.append(row)
        print('MLP sweep:', row, flush=True)
    best_mlp = min(mlp_records, key=lambda r:r['validation_mse'])
    # Refit from scratch on the original training rows, preserving validation.
    rks = RandomKitchenSinks(sigma=best_rks['parameter']).fit(x, y)
    mlp, hist, _ = fit_mlp(x, y, xv, yv, best_mlp['parameter'], fixed_epochs=best_mlp['best_epoch'])
    np.savez(ROOT/'results/rks_weights.npz', W=rks.W, b=rks.b, coef=rks.coef_,
             phi_mean=rks.phi_mean, y_mean=rks.y_mean)
    torch.save(dict(state=mlp.state_dict(), hidden=best_mlp['parameter'],
                    y_mean=mlp.y_mean, y_scale=mlp.y_scale), ROOT/'results/mlp_weights.pt')
    for name, model in [('rks',rks), ('mlp',mlp)]:
        pred = np.asarray(model.predict(xt)).reshape(-1)
        if pred.shape != (len(xt),) or not np.isfinite(pred).all():
            raise ValueError('Invalid unlabeled-test predictions')
        np.savetxt(ROOT/f'{name}_test_predictions.csv', pred, delimiter=',', fmt='%.12g')
    pd.DataFrame(rks_records).to_csv(ROOT/'results/rks_sweep.csv', index=False)
    pd.DataFrame(mlp_records).to_csv(ROOT/'results/mlp_sweep.csv', index=False)
    sweep_plots(rks_records, 'rks', 'Kernel scale sigma')
    sweep_plots(mlp_records, 'mlp', 'Hidden width')
    plot_outputs(x, y, rks, mlp)
    result = dict(seed=SEED, data=[provenance(p) for p in paths],
                  splits=dict(train=len(y),validation=len(yv),unlabeled_test=len(xt)),
                  training_response_quantiles={str(q):float(np.quantile(y,q)) for q in [0,.5,1]},
                  validation_constant_mse=mse(yv,np.full(len(yv),y.mean())),
                  rks=dict(D=1000,h=500,ridge_lambda=.01,objective='sum squared error + lambda * squared coefficient norm; unpenalized intercept',
                           sweep=rks_records,selected_sigma=best_rks['parameter'],
                           refit_train_mse=mse(y,rks.predict(x)),refit_validation_mse=mse(yv,rks.predict(xv))),
                  mlp=dict(activation='tanh',learning_rate=.003,batch_size=128,weight_decay=.0001,
                           max_epochs=5000,patience=300,sweep=mlp_records,selected_width=best_mlp['parameter'],
                           refit_epochs=best_mlp['best_epoch'],refit_train_mse=mse(y,mlp.predict(x)),
                           refit_validation_mse=mse(yv,mlp.predict(xv))),
                  refit_policy='original training split only; same seeds; MLP epoch fixed at selected best checkpoint',
                  prediction_format='200 finite rows, one column, no header, original test order; no test labels available')
    (ROOT/'results/toy_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def plot_outputs(x, y, rks, mlp):
    fig, ax = plt.subplots(figsize=(5.6,4.1))
    sc = ax.scatter(x[:,0],x[:,1],c=y,s=18,cmap='viridis',vmin=-2,vmax=6)
    ax.set(xlabel='$x_1$',ylabel='$x_2$',xlim=(-3,3),ylim=(-3,3),title='Training responses (600 observations)')
    fig.colorbar(sc,ax=ax,label='$y$'); fig.tight_layout();fig.savefig(ROOT/'figures/toy_training.pdf');plt.close(fig)
    grid = np.linspace(-3,3,160);g1,g2=np.meshgrid(grid,grid);gx=np.c_[g1.ravel(),g2.ravel()]
    fig, axes = plt.subplots(1,2,figsize=(10,3.9),sharex=True,sharey=True,layout='constrained')
    for ax, name, model in zip(axes,['RKS + ridge','One-hidden-layer MLP'],[rks,mlp]):
        z=model.predict(gx).reshape(g1.shape)
        sc=ax.imshow(z,origin='lower',extent=(-3,3,-3,3),aspect='auto',cmap='viridis',vmin=-2,vmax=6)
        ax.set(title=name,xlabel='$x_1$',ylabel='$x_2$')
    fig.colorbar(sc,ax=axes,label='Predicted response',shrink=.85)
    fig.savefig(ROOT/'figures/toy_surfaces.pdf');plt.close(fig)
