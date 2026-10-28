# Corrected Homework 2 — reproducible experiments

Yichen Shen · DATA 37711 · Fall 2026

This release follows the instructor's corrected Homework 2 announced on October
27, due November 6, 2026 at 11:59 pm: Q1 taxi uncertainty, Q2 retail categorical
encodings, and Q3 Random Kitchen Sinks versus a one-hidden-layer MLP.

## Install and reproduce

Tested with CPython 3.12.6, macOS 15.6 arm64, CPU only. From this homework directory:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r code/requirements.lock
.venv/bin/python code/run_experiments.py --download-data
.venv/bin/python code/verify_results.py
```

The third line is the single command that retrieves the verified data and reruns
all experiments, sweeps, plots, and required prediction files. The last line checks
saved reference metrics, disjoint exhaustive splits, exact Q3 input hashes, and
both ordered prediction files. `code/requirements.txt` pins direct dependencies;
`code/requirements.lock` pins the full tested environment. No course checkout,
previous environment, credentials, or GPU is required. Other platforms may need
compatible PyTorch wheels and can differ numerically. Allow approximately 6–10
minutes for experiments, plus installation and download, and roughly 2 GB of disk
space. A CPU is sufficient; experiments use one computation thread.

Installations require access to PyPI. First data retrieval downloads about 16 MB
from `raw.githubusercontent.com`; subsequent runs use the local data cache.
`code/data_sources.py` records pinned URLs and SHA-256 values. The taxi file is
byte-identical to the corrected attachment; every parsed value and row of the
retail file was verified against the supplied PDF printout. Exact corrected Q3
CSV files are bundled in `inputs/`, with instructor file IDs and checksums in
`inputs/manifest.json`. They are copied without changing row order. No test labels
are available for Q3. The instructor's original PDFs are not part of this release.

For supplied files stored elsewhere, omit `--download-data` and specify:

```sh
.venv/bin/python code/run_experiments.py --data-dir /absolute/path/to/data
```

That directory must contain `taxi_trips.parquet`, `retail.csv`, `train.csv`,
`val.csv`, and `test.csv`. Taxi expects the provided eight columns, with target
`trip_duration` in **minutes** and seven numeric features. Retail expects positive
`Quantity` and categorical `StockCode`, `CustomerID`. Q3 expects the exact supplied
600/200/200 split: labeled `x1,x2,y` in training/validation, `x1,x2` in test.
Use `--questions 1`, `--questions 2`, or `--questions 3` to rerun a subset.

## Methods

Seed 37711 fixes every split and model initialization. Taxi and retail use
70/10/20 random-row splits, saved under `results/`. Preprocessing is fitted on
training rows only; validation selects neural checkpoints; test rows never tune
models. Category target means use five-fold out-of-fold training encodings and
training-only evaluation maps, with global means for unseen categories. Unknown
embeddings use fixed zero vectors.

Taxi uses a 7–64–64–2 ReLU MLP. A real mean and square-root softplus standard
deviation define a Gaussian likelihood, with variance floor 1e-4 on the
standardized target scale. Means and standard deviations are returned to minutes
before scoring. Constant and ridge Gaussian baselines are fitted on training data.
Adam uses lr .001, weight decay .0001, batch size 1024, gradient clipping at 10,
150-epoch cap, validation-NLL patience 15, and restored best weights.

Retail compares target encoding, two 16-dimensional embeddings, and both together
using the same 64–32 ReLU head. Exact raw-Quantity category means are transformed
with log1p and standardized. The response is training-standardized log1p(Quantity),
and squared loss is optimized. Predicted quantities are max(0, expm1(predicted
log-target)). This estimates log-scale location rather than the raw conditional
mean. Main evaluation is MSLE; original-scale MAE and MSE are also retained.
Optimizer settings match taxi, with a 100-epoch cap and validation patience 15.

Q3 keeps the supplied splits. RKS uses unnormalized paired sine/cosine features,
h=500, D=1000, independent uniform phases and N(0,I/sigma²) projections. The sweep
varies only sigma over .1/.2/.4/.7/1/1.5/2/3/5, with common underlying random draws.
Ridge minimizes **sum** squared error + .01 ||beta||² with an unpenalized intercept;
a centered dual linear solve avoids matrix inversion. The MLP has exactly one
tanh hidden layer and sweeps widths 16/32/64/128/256. All other settings are fixed:
Adam lr .003, weight decay .0001, batch size 128, 5000-epoch cap and validation
patience 300. Targets use training-only standardization; reported losses use the
original scale. Selected models are refitted from scratch on the original training
split with the same seed; MLP epochs are fixed at its selected checkpoint's epoch.
Validation remains held out of refit training. It was used for selection, so its
scores are not an unbiased final test comparison.

`code/dataloader.py` is the unchanged supplied loader. `code/toy.py` independently
implements the permitted models; the starter skeleton's TODOs are not executed.

## Outputs and reference values

`results/` contains raw JSON metrics, sweeps, training histories, split indices,
held-out predictions, weights, and environment metadata. `figures/` contains all
PDF plots. Directories are created on a clean run. `reference_results/` preserves
small baseline outputs for the verification command. Large generated weights,
splits, cached data, and per-row Q1/Q2 predictions are excluded from Git.

| Experiment | Representative results |
| --- | --- |
| Taxi MLP, test | MSE 15.11841167; Gaussian NLL 2.62815454; Brier .00306904; accuracy .99630415 |
| Retail target encoding, test | MSLE .38669556; MAE 8.10144766; MSE 84502.2261 |
| Retail embeddings, test | MSLE .27967012; MAE 7.01115896; MSE 83501.4630 |
| Retail combined, test | MSLE .27202708; MAE 6.86896955; MSE 83530.6478 |
| RKS refit, validation | sigma 3; MSE .03652104 |
| One-hidden-layer MLP refit, validation | width 32; 2186 epochs; MSE .03606205 |

The two required Q3 submissions are generated directly in this homework directory:
`rks_test_predictions.csv` and `mlp_test_predictions.csv`. Each contains 200 finite
values, exactly one numeric column, no header/index, in supplied test row order.
No Q3 test MSE is reported. The solution PDF, separate reflection, and instructor
assignment are not needed to execute the code.

Verification uses rtol 1e-4 and atol 1e-6 for metrics and predictions, with exact
input hashes, split structure, selected hyperparameters, and epoch counts on the
tested platform. Cross-platform floating-point differences can be larger and can
change checkpoint selection. RKS and MLP have very similar validation scores;
one split does not establish a reliable winner. Taxi's majority-event baseline
already has 99.5435% accuracy. Retail raw MSE remains dominated by extreme
quantities; log-scale improvements do not imply good extreme-order prediction.
