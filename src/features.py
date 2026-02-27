"""Feature engineering for baseline models.

Turns a time series per SKU into a supervised learning table:
  X(t) = [sales lags, rolling stats, exogenous vars, calendar]
  y(t) = sales at t+1..t+horizon
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

EXOG_COLS = ["price", "promo_flag", "discount_pct", "is_weekend", "is_holiday"]

@dataclass
class SupervisedFrame:
    X: pd.DataFrame
    y: pd.DataFrame
    feature_cols: list[str]
    target_cols: list[str]

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["dow"] = out["date"].dt.dayofweek.astype(int)
    out["month"] = out["date"].dt.month.astype(int)
    out["dow_sin"] = np.sin(2*np.pi*out["dow"]/7.0)
    out["dow_cos"] = np.cos(2*np.pi*out["dow"]/7.0)
    out["month_sin"] = np.sin(2*np.pi*out["month"]/12.0)
    out["month_cos"] = np.cos(2*np.pi*out["month"]/12.0)

    if "is_weekend" not in out.columns:
        out["is_weekend"] = (out["dow"] >= 5).astype(int)
    if "is_holiday" not in out.columns:
        out["is_holiday"] = 0
    if "promo_flag" not in out.columns:
        out["promo_flag"] = 0
    if "discount_pct" not in out.columns:
        out["discount_pct"] = 0.0
    if "price" not in out.columns:
        out["price"] = 1.0
    return out

def make_supervised(df: pd.DataFrame, lookback: int = 30, horizon: int = 14) -> SupervisedFrame:
    df = add_calendar(df).sort_values(["sku","date"]).reset_index(drop=True)

    frames = []
    for sku, g in df.groupby("sku", sort=False):
        gg = g.copy()
        for k in range(1, lookback+1):
            gg[f"sales_lag_{k}"] = gg["sales"].shift(k)
        gg["roll_mean_7"] = gg["sales"].shift(1).rolling(7).mean()
        gg["roll_std_7"]  = gg["sales"].shift(1).rolling(7).std()

        for h in range(1, horizon+1):
            gg[f"y_{h}"] = gg["sales"].shift(-h)

        frames.append(gg)

    sup = pd.concat(frames, ignore_index=True)
    need = [f"sales_lag_{k}" for k in range(1, lookback+1)] + [f"y_{h}" for h in range(1, horizon+1)]
    sup = sup.dropna(subset=need).reset_index(drop=True)

    feature_cols = (
        [f"sales_lag_{k}" for k in range(1, lookback+1)]
        + ["roll_mean_7","roll_std_7"]
        + EXOG_COLS
        + ["dow_sin","dow_cos","month_sin","month_cos"]
    )
    target_cols = [f"y_{h}" for h in range(1, horizon+1)]

    X = sup[feature_cols + ["date","sku"]].copy()
    y = sup[target_cols].copy()
    return SupervisedFrame(X=X, y=y, feature_cols=feature_cols, target_cols=target_cols)
