"""PyTorch dataset helpers for multi-step forecasting."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "sales", "price", "promo_flag", "discount_pct", "is_weekend", "is_holiday",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
]

@dataclass
class TSConfig:
    lookback: int = 28
    horizon: int = 14

def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
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

def build_sequences_by_sku(df: pd.DataFrame, cfg: TSConfig):
    """Return dict sku -> (X, y, dates). X shape (N, lookback, F), y shape (N, horizon)."""
    df = _add_calendar(df).sort_values(["sku","date"]).reset_index(drop=True)
    out = {}
    for sku, g in df.groupby("sku", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        arr = g[FEATURE_COLS].to_numpy(dtype=float)
        sales = g["sales"].to_numpy(dtype=float)
        dates = g["date"].to_numpy()
        Xs, ys, ds = [], [], []
        for i in range(cfg.lookback, len(g) - cfg.horizon):
            Xs.append(arr[i-cfg.lookback:i, :])
            ys.append(sales[i:i+cfg.horizon])
            ds.append(dates[i])
        if Xs:
            out[sku] = (np.stack(Xs), np.stack(ys), np.array(ds))
    return out
