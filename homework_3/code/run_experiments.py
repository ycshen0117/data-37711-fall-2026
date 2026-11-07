"""Reproduce the wine model search and the two gradient boosting experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from itertools import product
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HOMEWORK = Path(__file__).resolve().parents[1]
DATA = HOMEWORK / "data"
RESULTS = HOMEWORK / "results"
FIGURES = HOMEWORK / "figures"
SEED = 42
BOOST_SEEDS = range(SEED, SEED + 10)
STAGES = (1, 10, 100, 300, 1000)
WINE_URL = (
    "https://raw.githubusercontent.com/zygmuntz/wine-quality/"
    "5896f43e4ef7f64ac83a8e85a93fcc03293c0347/winequality/winequality-white.csv"
)
DSET_BASE = (
    "https://raw.githubusercontent.com/Chenxi1818/foundation_of_ML/"
    "f386491eef32a7e1c9c0dde29e69430822dcc066/a3_decision_tree/data/"
)
INPUTS = {
    "winequality-white.csv": (
        WINE_URL,
        "76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836",
    ),
    "dset0.csv": (
        DSET_BASE + "dset0.csv",
        "bc6fe805760b5cf1bd6da32ed47b2773e5330679224b8fffb7eecb0021af4519",
    ),
    "dset1.csv": (
        DSET_BASE + "dset1.csv",
        "cd33ccaaa01e7b770772de2900f8ad627f104432f0d9e101dc402ccf46c32f75",
    ),
}


def input_files():
    for filename, (url, expected) in INPUTS.items():
        path = DATA / filename
        if path.is_file():
            raw = path.read_bytes()
        else:
            with urlopen(url, timeout=60) as response:
                raw = response.read()
        normalized = raw.replace(b"\r\n", b"\n")
        if hashlib.sha256(normalized).hexdigest() != expected:
            raise ValueError(f"Input checksum mismatch for {filename}")
        if not path.exists():
            path.write_bytes(normalized)


def write_csv(name, records):
    with (RESULTS / name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def wine_search():
    path = DATA / "winequality-white.csv"
    with path.open(newline="") as file:
        columns = next(csv.reader(file, delimiter=";"))
    data = np.loadtxt(path, delimiter=";", skiprows=1)
    if columns[-1] != "quality" or data.shape != (4898, 12):
        raise ValueError("Unexpected white wine schema")

    train, held_out = train_test_split(np.arange(len(data)), test_size=0.30, random_state=SEED)
    val, test = train_test_split(held_out, test_size=980, random_state=SEED)
    x, y = data[:, :-1], data[:, -1]
    x_train, y_train = x[train], y[train]
    x_val, y_val = x[val], y[val]
    write_csv("wine_splits.csv", [
        {"row": int(i), "split": name}
        for name, indices in (("train", train), ("validation", val), ("test", test))
        for i in indices
    ])
    metrics = {
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "test_train_mean_mse": mean_squared_error(y[test], np.full(len(test), y_train.mean())),
    }

    # Reuse the same seeded trees when only the number of trees changes.
    forest_results = []
    for depth, leaf, features in product(
        (4, 6, 8, 10, 12, 16, 24, None), (1, 2, 4, 8), (0.3, 0.5, 0.7, 1.0)
    ):
        model = RandomForestRegressor(
            max_depth=depth, min_samples_leaf=leaf, max_features=features,
            random_state=SEED, warm_start=True,
        )
        for trees in (16, 32, 64, 128):
            model.set_params(n_estimators=trees).fit(x_train, y_train)
            forest_results.append({
                "n_estimators": trees,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "max_features": features,
                "validation_mse": mean_squared_error(y_val, model.predict(x_val)),
            })
    write_csv("wine_forest_grid.csv", forest_results)
    print("Forest grid complete", flush=True)

    scaler = StandardScaler().fit(x_train)
    scaled_train, scaled_val = scaler.transform(x_train), scaler.transform(x_val)
    neighbor_results = []
    for neighbors, weights, p in product(range(1, 129), ("uniform", "distance"), (1, 2)):
        model = KNeighborsRegressor(n_neighbors=neighbors, weights=weights, p=p, algorithm="brute")
        model.fit(scaled_train, y_train)
        neighbor_results.append({
            "n_neighbors": neighbors,
            "weights": weights,
            "p": p,
            "validation_mse": mean_squared_error(y_val, model.predict(scaled_val)),
        })
    write_csv("wine_neighbors_grid.csv", neighbor_results)

    # Lock both winners before scoring either on the test set.
    grids = {"forest": forest_results, "neighbors": neighbor_results}
    winners = {
        name: min(records, key=lambda record: record["validation_mse"])
        for name, records in grids.items()
    }
    for name, winner in winners.items():
        params = {key: value for key, value in winner.items() if key != "validation_mse"}
        if name == "forest":
            model = RandomForestRegressor(**params, random_state=SEED)
            fit_x, val_x, test_x = x_train, x_val, x[test]
        else:
            model = KNeighborsRegressor(**params, algorithm="brute")
            fit_x, val_x, test_x = scaled_train, scaled_val, scaler.transform(x[test])
        model.fit(fit_x, y_train)
        np.testing.assert_allclose(
            mean_squared_error(y_val, model.predict(val_x)), winner["validation_mse"],
            rtol=0, atol=1e-12,
        )
        predictions = model.predict(test_x)
        metrics[name] = {
            "configurations": len(grids[name]),
            "parameters": params,
            "validation_mse": winner["validation_mse"],
            "test_mse": mean_squared_error(y[test], predictions),
        }
        write_csv(f"wine_{name}_test_predictions.csv", [
            {"row": int(i), "quality": float(y[i]), "prediction": float(pred)}
            for i, pred in zip(test, predictions, strict=True)
        ])
    (RESULTS / "wine_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def boosting_study():
    scores, baselines, splits = [], [], []
    for name in ("dset0", "dset1"):
        data = np.loadtxt(DATA / f"{name}.csv", delimiter=",")
        x, y = data[:, 1:], data[:, 0].astype(int)
        if data.shape != (500, 6) or set(data[:, 0]) != {0, 1}:
            raise ValueError(f"Unexpected {name} schema")
        for seed in BOOST_SEEDS:
            train, test = train_test_split(
                np.arange(len(data)), test_size=0.30, stratify=y, random_state=seed,
            )
            splits.extend(
                {"dataset": name, "seed": seed, "row": int(i), "split": split}
                for split, indices in (("train", train), ("test", test))
                for i in indices
            )
            model = GradientBoostingClassifier(
                loss="log_loss", learning_rate=0.1, max_depth=3,
                n_estimators=STAGES[-1], random_state=seed,
            ).fit(x[train], y[train])
            for split, indices in (("train", train), ("test", test)):
                for stage, probs in enumerate(model.staged_predict_proba(x[indices]), start=1):
                    if stage in STAGES:
                        scores.append({
                            "dataset": name, "seed": seed, "trees": stage, "split": split,
                            "error": float(np.mean(model.classes_[probs.argmax(axis=1)] != y[indices])),
                            "log_loss": log_loss(y[indices], probs),
                        })
            linear = make_pipeline(StandardScaler(), LogisticRegression()).fit(x[train], y[train])
            baselines.append({
                "dataset": name, "seed": seed,
                "majority_error": float(np.mean(y[test] != int(y[train].mean() > 0.5))),
                "logistic_error": float(np.mean(linear.predict(x[test]) != y[test])),
                "logistic_auc": roc_auc_score(y[test], linear.predict_proba(x[test])[:, 1]),
            })
    write_csv("boosting_scores.csv", scores)
    write_csv("boosting_baselines.csv", baselines)
    write_csv("boosting_splits.csv", splits)

    summary = []
    for name, stage, split in product(("dset0", "dset1"), STAGES, ("train", "test")):
        records = [
            row for row in scores
            if (row["dataset"], row["trees"], row["split"]) == (name, stage, split)
        ]
        row = {"dataset": name, "trees": stage, "split": split}
        for metric in ("error", "log_loss"):
            values = [record[metric] for record in records]
            row[f"{metric}_seed42"] = values[0]
            row[f"{metric}_mean"] = float(np.mean(values))
        summary.append(row)
    write_csv("boosting_summary.csv", summary)
    plot_boosting(summary)


def plot_boosting(summary):
    os.environ.setdefault("MPLCONFIGDIR", str(RESULTS / "mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in ("dset0", "dset1"):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), layout="constrained")
        for axis, metric, ylabel in zip(
            axes, ("error", "log_loss"), ("Classification error (%)", "Log loss"), strict=True
        ):
            scale = 100 if metric == "error" else 1
            for split, color, marker in (("train", "#265e9e", "o"), ("test", "#ac4a16", "s")):
                values = [
                    scale * row[f"{metric}_mean"] for row in summary
                    if row["dataset"] == name and row["split"] == split
                ]
                axis.plot(STAGES, values, color=color, marker=marker, label=split.capitalize())
            axis.set(xscale="log", xlabel="Trees", ylabel=ylabel, ylim=(0, None))
            axis.set_xticks(STAGES, labels=[str(stage) for stage in STAGES])
            axis.minorticks_off()
            axis.grid(axis="y", alpha=0.25)
        axes[0].legend(frameon=False)
        fig.savefig(FIGURES / f"{name}_boosting.pdf")
        plt.close(fig)


def main():
    for path in (DATA, RESULTS, FIGURES):
        path.mkdir(exist_ok=True)
    input_files()
    with threadpool_limits(limits=1):
        wine_search()
        boosting_study()
    print("Wrote wine and boosting results")


if __name__ == "__main__":
    main()
