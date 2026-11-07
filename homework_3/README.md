# DATA 37711 Homework 3

Runnable experiments for the white wine model search (Q2c) and boosted tree
classification (Q3). Tested with CPython 3.12.6 on macOS 15.6, Apple Silicon.
CPU only. Allow about two minutes for the full run on that machine.

## Reproduce

From the homework directory:

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r code/requirements.lock
    python code/run_experiments.py

Internet access to PyPI and public GitHub raw files is required for setup and
the first run. The script downloads three CSVs to data/ without credentials.
It checks their SHA-256 digests after converting CRLF line endings to LF.
For an offline run with dependencies installed, place the same three files
there manually. Save the HW3 attachments dset0-1.csv and dset1-1.csv as
data/dset0.csv and data/dset1.csv.

- The white wine data are the 4,898-row semicolon-delimited file from the
  [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
  (Cortez et al., 2009). The script uses a pinned public mirror that matched
  the locally supplied course copy byte for byte.
- The two boosting files are headerless (500 rows each), with binary labels
  followed by five features. The pinned public mirrors match the two HW3
  attachments in all parsed values, schema, and row order. Line endings are
  the only byte difference.

See the [random forest](https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.RandomForestRegressor.html),
[nearest neighbors](https://scikit-learn.org/1.9/modules/generated/sklearn.neighbors.KNeighborsRegressor.html),
and [gradient boosting](https://scikit-learn.org/1.9/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
documentation for the estimators.

## Outputs and checks

The run uses seed 42 for the wine split (3,428 train, 490 validation, 980 test),
fitting the scaler on training data only. It scores 512 configurations for
each regressor on validation MSE, then evaluates the selected training-only
fits on the test set. Results include both complete grids, split indices,
predictions, and the summary in results/wine_metrics.json.

| Model | Validation MSE | Test MSE |
| --- | ---: | ---: |
| Random forest | 0.401425 | 0.321230 |
| Nearest neighbors | 0.420880 | 0.349469 |

The boosting experiment uses the same settings for both datasets. Ten
stratified 350/150 train/test splits use seeds 42–51, with depth 3, learning
rate 0.1, and evaluation at 1, 10, 100, 300, and 1,000 trees. Individual
scores, split indices, a logistic baseline, and aggregate scores are written
to results/. Mean test log loss at 100 trees is 0.391163 (dset0) and 0.432379
(dset1). Plots are written to figures/dset0_boosting.pdf and
figures/dset1_boosting.pdf. For a same-platform check, compare the numeric
summaries within 0.00001 of the figures above. On other platforms tied
distances or split scores may lead to small differences.
