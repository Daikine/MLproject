"""Evaluate baseline models (Milestone 2)."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from .features import make_supervised
from .baselines import naive_last_value, moving_average, RidgeMultiStep

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sales.csv")
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--test-days", type=int, default=120)
    args = ap.parse_args()

    df = pd.read_csv(args.data, parse_dates=["date"])
    df = df.sort_values(["sku","date"]).reset_index(drop=True)

    rows = []
    for sku, g in df.groupby("sku", sort=False):
        g = g.sort_values("date")
        cutoff = g["date"].max() - pd.Timedelta(days=args.test_days)
        train = g[g["date"] <= cutoff]
        test  = g[g["date"] > cutoff]

        # History for naive/MA
        history = train["sales"].to_numpy(dtype=float)

        # Evaluate on each test date using last known history only (simple, fair baseline)
        for t in range(len(test)):
            y_true = test["sales"].iloc[t : t + args.horizon].to_numpy(dtype=float)
            if len(y_true) < 1:
                continue
            y_true = np.pad(y_true, (0, max(0, args.horizon-len(y_true))), constant_values=y_true[-1] if len(y_true)>0 else 0.0)

            y_naive = naive_last_value(history, args.horizon)
            y_ma = moving_average(history, args.horizon, window=7)

            rows.append((sku, "naive", y_true, y_naive))
            rows.append((sku, "moving_avg_7", y_true, y_ma))

            # update history with true next day (walk-forward)
            history = np.append(history, float(test["sales"].iloc[t]))

        # Ridge on supervised frame
        sup = make_supervised(g, lookback=args.lookback, horizon=args.horizon)
        X = sup.X
        y = sup.y
        # Train on rows where date <= cutoff and predict where date > cutoff (last available row before cutoff for recursive)
        train_mask = X["date"] <= cutoff
        X_train = X.loc[train_mask].copy()
        y_train = y.loc[train_mask].copy()
        X_test = X.loc[~train_mask].copy()
        y_test = y.loc[~train_mask].copy()

        if len(X_train) > 50 and len(X_test) > 0:
            model = RidgeMultiStep(alpha=1.0).fit(X_train, y_train["y_1"], sup.feature_cols)
            # Use each test row as starting point (already contains lags)
            for i in range(len(X_test)):
                row = X_test.iloc[i]
                y_true = y_test.iloc[i].to_numpy(dtype=float)
                y_pred = model.predict_recursive(row, args.horizon, args.lookback)
                rows.append((sku, "ridge_recursive", y_true, y_pred))

    # Aggregate metrics
    metrics = {}
    for sku, model_name, y_true, y_pred in rows:
        key = (sku, model_name)
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        m = metrics.setdefault(key, {"y_true": [], "y_pred": []})
        m["y_true"].append(y_true)
        m["y_pred"].append(y_pred)

    out_rows = []
    for (sku, model_name), d in metrics.items():
        yt = np.concatenate(d["y_true"])
        yp = np.concatenate(d["y_pred"])
        out_rows.append({
            "sku": sku,
            "model": model_name,
            "mae": float(mean_absolute_error(yt, yp)),
            "mse": float(mean_squared_error(yt, yp)),
            "rmse": rmse(yt, yp),
            "mape": mape(yt, yp),
        })

    out_df = pd.DataFrame(out_rows).sort_values(["sku","model"]).reset_index(drop=True)

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    out_df.to_csv("artifacts/baseline_metrics.csv", index=False)

    with open("artifacts/metrics_baselines.json", "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    print("Saved artifacts/baseline_metrics.csv and artifacts/metrics_baselines.json")

if __name__ == "__main__":
    main()
