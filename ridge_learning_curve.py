#!/usr/bin/env python3
import re
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import ShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

CASE_RE = re.compile(r"^(?P<runID>[A-Z]\d{2,3})_(?P<placement>T|S|SZ|TS)_N(?P<N>\d+)_Q(?P<Q>\d+)$")

def parse_case_name(name: str):
    m = CASE_RE.match(name)
    if not m:
        raise ValueError(f"Cannot parse case name: {name}")
    return {"runID": m.group("runID")}

def build_dataset(dead_csv="dead_volume.csv", runs_csv="runs.csv"):
    dead = pd.read_csv(dead_csv)
    runs = pd.read_csv(runs_csv)

    meta = dead["case"].apply(parse_case_name).apply(pd.Series)
    data = pd.concat([dead, meta], axis=1).merge(runs, on="runID", how="left")

    if data["pipeHeight_mm"].isna().any():
        bad = data.loc[data["pipeHeight_mm"].isna(), "case"].head(10).tolist()
        raise RuntimeError(f"Some cases did not match runs.csv. Examples: {bad}")

    y = data["deadFraction"].values

    feature_cols = [
        "pipeHeight_mm",
        "D",
        "placement",
        "N",
        "Qtot_uLmin",
        "weightMode",
        "cosA",
        "cosPhase",
    ]
    feature_cols = [c for c in feature_cols if c in data.columns]
    X = data[feature_cols].copy()

    return data, X, y

def make_model(feature_cols):
    cat_cols = [c for c in ["placement", "weightMode"] if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop"
    )

    alphas = np.logspace(-6, 6, 61)

    return Pipeline(steps=[
        ("pre", pre),
        ("ridge", RidgeCV(alphas=alphas, cv=5))
    ])

def eval_repeated_holdout(X, y, n_train_requested, n_repeats=30, test_size=0.1, seed=0):
    maes, rmses, alphas = [], [], []
    n_eff_list = []

    rs = ShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=seed)

    for rep, (tr_all, te) in enumerate(rs.split(X, y), start=1):
        n_train_eff = min(n_train_requested, len(tr_all))
        n_eff_list.append(n_train_eff)

        rng = np.random.RandomState(seed + 1000*rep + n_train_eff)
        tr = rng.choice(tr_all, size=n_train_eff, replace=False)

        model = make_model(list(X.columns))
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[te])

        maes.append(mean_absolute_error(y[te], pred))
        rmses.append(mean_squared_error(y[te], pred, squared=False))
        alphas.append(model.named_steps["ridge"].alpha_)

    return {
        "method": "repeated_holdout",
        "n_repeats": int(n_repeats),
        "test_size": float(test_size),
        "n_train_effective_mean": float(np.mean(n_eff_list)),
        "n_train_effective_std": float(np.std(n_eff_list)),
        "MAE_mean": float(np.mean(maes)),
        "MAE_std": float(np.std(maes)),
        "RMSE_mean": float(np.mean(rmses)),
        "RMSE_std": float(np.std(rmses)),
        "alpha_median": float(np.median(alphas)),
    }

def try_plot(df, out_png="ridge_learning_curve.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot. Use ridge_learning_curve.csv to plot locally.")
        return

    x = df["n_train_effective_mean"].values

    plt.figure(figsize=(6.5,4.5), dpi=160)
    plt.errorbar(x, df["MAE_mean"], yerr=df["MAE_std"], marker="o", capsize=3, label="MAE")
    plt.errorbar(x, df["RMSE_mean"], yerr=df["RMSE_std"], marker="o", capsize=3, label="RMSE")
    plt.xlabel("Effective number of training samples (mean over splits)")
    plt.ylabel("Error")
    plt.title("Ridge learning curve (consistent repeated holdout)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Wrote {out_png}")

def main():
    train_sizes = [10, 25, 50, 75, 90, 104]

    _, X, y = build_dataset("dead_volume.csv", "runs.csv")

    rows = []
    for n_train in train_sizes:
        res = eval_repeated_holdout(X, y, n_train, n_repeats=30, test_size=0.2, seed=0)
        row = {"n_train_requested": int(n_train), **res}
        rows.append(row)

        print(f"n_train_requested={n_train:3d} (effective≈{res['n_train_effective_mean']:.1f}): "
              f"MAE={res['MAE_mean']:.4g}±{res['MAE_std']:.3g}, "
              f"RMSE={res['RMSE_mean']:.4g}±{res['RMSE_std']:.3g}")

    df = pd.DataFrame(rows)
    df.to_csv("ridge_learning_curve.csv", index=False)
    print("Wrote ridge_learning_curve.csv")

    try_plot(df, "ridge_learning_curve.png")

if __name__ == "__main__":
    main()
