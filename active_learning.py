#!/usr/bin/env python3
import re
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import RidgeCV

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
    return data, X, y, feature_cols

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

def build_candidate_grid_inbounds():
    # Fixed baseline geometry
    H = 8.0
    D = 1.88e9

    # Expanded scheme grid to get more candidates
    placements = ["T", "S", "SZ", "TS"]
    
    # Add more N values
    Ns = [2, 6, 10, 16, 20, 30, 40, 50]
    
    # Add intermediate Qtot values
    Qtots = [10, 15, 20, 25, 35, 50, 75, 100, 125, 150]

    weightMode = "cosine"
    cosA = 0.6
    cosPhase = 0.0

    rows = []
    for plc in placements:
        for N in Ns:
            for Q in Qtots:
                rows.append({
                    "pipeHeight_mm": H,
                    "D": D,
                    "placement": plc,
                    "N": int(N),
                    "Qtot_uLmin": float(Q),
                    "weightMode": weightMode,
                    "cosA": cosA,
                    "cosPhase": cosPhase
                })
    return pd.DataFrame(rows)

def mark_already_run(cand: pd.DataFrame, data_runs: pd.DataFrame):
    key_cols = ["pipeHeight_mm","D","placement","N","Qtot_uLmin","weightMode","cosA","cosPhase"]
    existing = set(tuple(row) for row in data_runs[key_cols].itertuples(index=False, name=None))
    cand["isAlreadyRun"] = [tuple(row) in existing for row in cand[key_cols].itertuples(index=False, name=None)]
    return cand

def bootstrap_predict(Xtrain, ytrain, Xcand, feature_cols, B=200, seed=0):
    rng = np.random.RandomState(seed)
    preds = np.zeros((B, len(Xcand)), dtype=float)
    alphas = np.zeros(B, dtype=float)

    n = len(Xtrain)
    for b in range(B):
        idx = rng.randint(0, n, size=n)
        model = make_model(feature_cols)
        model.fit(Xtrain.iloc[idx], ytrain[idx])
        preds[b, :] = model.predict(Xcand)
        alphas[b] = model.named_steps["ridge"].alpha_

    mu = preds.mean(axis=0)
    sigma = preds.std(axis=0)
    return mu, sigma, alphas

def main():
    K = 20
    B = 200

    if not Path("dead_volume.csv").exists():
        raise FileNotFoundError("dead_volume.csv not found")
    if not Path("runs.csv").exists():
        raise FileNotFoundError("runs.csv not found")

    data, Xtrain, ytrain, feature_cols = build_dataset("dead_volume.csv", "runs.csv")

    cand = build_candidate_grid_inbounds()
    print(f"Built candidate grid with {len(cand)} total combinations")

    data_runs = data[["pipeHeight_mm","D","placement","N","Qtot_uLmin","weightMode","cosA","cosPhase"]].copy()
    cand = mark_already_run(cand, data_runs)

    n_new = (~cand["isAlreadyRun"]).sum()
    print(f"After filtering already-run: {n_new} new candidates")

    Xcand = cand[feature_cols].copy()
    mu, sigma, alpha_samples = bootstrap_predict(Xtrain, ytrain, Xcand, feature_cols, B=B, seed=0)

    cand["mu_pred"] = mu
    cand["sigma_pred"] = sigma
    cand["LCB_k1"] = cand["mu_pred"] - 1.0*cand["sigma_pred"]
    cand["LCB_k2"] = cand["mu_pred"] - 2.0*cand["sigma_pred"]

    cand.to_csv("active_learning_candidates_inbounds_v2.csv", index=False)
    print("Wrote active_learning_candidates_inbounds_v2.csv")

    newcand = cand.loc[~cand["isAlreadyRun"]].copy()

    explore = newcand.sort_values("sigma_pred", ascending=False).head(K)
    explore.to_csv(f"suggest_next_explore_inbounds_v2_top{K}.csv", index=False)
    print(f"Wrote suggest_next_explore_inbounds_v2_top{K}.csv with {len(explore)} rows")

    lcb1 = newcand.sort_values("LCB_k1", ascending=True).head(K)
    lcb1.to_csv(f"suggest_next_lcb_k1_inbounds_v2_top{K}.csv", index=False)
    print(f"Wrote suggest_next_lcb_k1_inbounds_v2_top{K}.csv with {len(lcb1)} rows")

    lcb2 = newcand.sort_values("LCB_k2", ascending=True).head(K)
    lcb2.to_csv(f"suggest_next_lcb_k2_inbounds_v2_top{K}.csv", index=False)
    print(f"Wrote suggest_next_lcb_k2_inbounds_v2_top{K}.csv with {len(lcb2)} rows")

    print("\nBootstrap ridge alpha samples summary:")
    print(f"  median alpha = {np.median(alpha_samples):g}")

if __name__ == "__main__":
    main()
