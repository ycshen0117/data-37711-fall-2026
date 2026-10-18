# Homework 2 — reproducible experiments

Author: Yichen Shen. DATA 37711, Machine Learning 1, Fall 2026.

## Data

Questions 3–5 use `dset0.csv`, `dset1.csv`, `taxi_trips.parquet`, and `retail.csv`.
Automatic retrieval uses a pinned public mirror. Exact URLs and SHA-256 values
are in `code/data_sources.py`, and each experiment records its input checksum.
Downloaded datasets are stored locally in `data/`.

Question 2 retrieves `winequality-white.csv` (4898 rows, 11 features) from
[UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality),
DOI [10.24432/C56S3T](https://doi.org/10.24432/C56S3T), by Cortez and colleagues.
The archive URL and exact CSV checksum are checked in `code/wine.py`.

The taxi dataset contains standardized numeric features and trip duration in
**minutes**. Additional preprocessing uses training rows only. Nothing is
filtered from any dataset. The retail dataset has 397924 rows and positive
quantities.

## Installation and complete run

Tested with CPython **3.12.6**, macOS 15.6 arm64, CPU only. Direct dependencies are
pinned in `code/requirements.txt`. A different platform may need compatible
PyTorch wheels; numerical equality across platforms is not guaranteed.

After cloning the repository, run from its root:

```sh
cd homework_2
python3.12 -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
.venv/bin/python code/run_experiments.py --download-data --taxi-unit minutes
```

The last line is the single command for all required searches, model fits,
evaluations, and plots. It downloads approximately 17 MB of data to `data/` on
first run. Installation also needs internet for Python packages; first data
retrieval needs access to `archive.ics.uci.edu` and `raw.githubusercontent.com`.
No credentials, course checkout, previous environment, or GPU are required.
Typical experiment runtime is about **5 minutes** on the tested machine, plus
package installation. Allow roughly 1–2 GB of memory and disk space for the
environment, datasets, generated predictions, and model weights.

To use locally available files, put `dset0.csv`, `dset1.csv`,
`taxi_trips.parquet`, and `retail.csv` in a separate directory and omit
`--download-data`:

```sh
.venv/bin/python code/run_experiments.py --data-dir /absolute/path/to/data --taxi-target trip_duration --taxi-unit minutes
```

The dset files have **no header**, binary target in the first column,
and five numeric predictors. Taxi expects the exact eight-column schema
and seven numeric features. Change the explicit unit flag to `seconds` if the
input file records seconds; this affects all event probabilities. Retail
expects columns `Quantity`, `StockCode`, `CustomerID`. The optional
`winequality-white.csv` must be semicolon separated with `quality` as its target.
Other schemas require review rather than silently guessing columns or units.

Use `--questions 2`, `--questions 3`, `--questions 4`, or `--questions 5` to rerun a
subset. Every run overwrites its own generated outputs.

## Methods

All seeds are 37711. Wine, taxi, and retail use random 70/10/20 train/validation/
test partitions with rounding; boosting uses 80/20 train/test. Splits are saved
as index arrays. The wine grids contain 540 forest and 600 nearest-neighbor
configurations, each greater than the 490 validation rows. Forest sizes use
warm starts for nested sizes, yielding the same trees as fresh fixed-seed fits.
Only validation MSE selects configurations; test rows are never used for tuning,
and selected models are not refitted on the validation set. Nearest-neighbor
features use training-only standardization.

Boosting compares all five required tree counts with depth 3, learning rate 0.1,
and binary log loss. Its test curves are descriptive comparisons of prespecified
fits, not a further tuning loop. Taxi uses a 7–64–64–2 ReLU MLP and Gaussian
negative log-likelihood, positive softplus variance, and a 1e-4 variance floor on
the standardized target scale. Baselines are training-mean and ridge Gaussians.
The original target is recovered before calculating MSE, density scores, or
under-45-minute probabilities. Accuracy uses probability threshold 0.5; Brier
score, event log loss, and interval coverage complement it.

Retail compares three MSE-trained 64–32 ReLU networks: two target means, two
16-dimensional embeddings, and their concatenation. Five-fold out-of-fold means
exclude each training observation's own target. Evaluation means use training
labels only. Unseen categories use the training global mean or an all-zero
embedding. All networks use Adam (lr 0.001, weight decay 0.0001), batches of 1024,
gradient norm clipping at 10, and validation early stopping with patience 15.
Taxi has a 150-epoch cap and retail a 100-epoch cap. Restored checkpoints are the
best validation checkpoints, not the final epochs. Test scores do not choose the
stopping epoch.

## Outputs and comparison targets

`results/` contains per-question JSON metrics, environment versions, all wine grid
scores, split indices, held-out predictions, neural training histories, and model
weights. `figures/` contains the wine search, boosting train/test curves, taxi
prediction scatter, and neural learning curves. Directories are created on a
clean run. Committed JSON/history/grid files are small reference outputs;
predictions, weights, splits, downloaded data, and caches are regenerated.

| Experiment | Representative test results |
| --- | --- |
| Wine random forest | MSE 0.38275483; validation MSE 0.40998893 |
| Wine nearest neighbors | MSE 0.38638179; validation MSE 0.39834150 |
| dset0 log loss, 1/10/100/300/1000 trees | 0.67328207 / 0.58884754 / 0.41932750 / 0.47430258 / 0.79810538 |
| dset1 log loss, 1/10/100/300/1000 trees | 0.65512467 / 0.47108122 / 0.39141495 / 0.44543177 / 0.80335801 |
| Taxi heteroscedastic MLP | MSE 15.11841167; Gaussian NLL 2.62815454; accuracy 0.99630415; Brier 0.00306904 |
| Retail target means | MSE 83804.8435; MAE 8.637458 |
| Retail embeddings | MSE 83421.6771; MAE 8.154370 |
| Retail combined | MSE 83666.0197; MAE 8.566309 |

On the tested platform expect exact splits and selected parameters. Compare
classical-model metrics with absolute tolerance 1e-6. For neural metrics use
relative tolerance 1e-4 (and absolute tolerance 1e-6 for probability scores),
while recording any changed stopping epochs. Cross-platform differences may be
larger. Check finiteness and data checksums before interpreting a discrepancy.
The taxi majority-event accuracy is already 0.99543510. Retail has extreme
quantities: the ten largest squared errors account for over 99% of test squared
error, and all three R-squared values are near zero. These are limitations, not
strong predictive performance claims.
