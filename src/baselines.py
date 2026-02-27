"""Baseline models."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

def naive_last_value(history: np.ndarray, horizon: int) -> np.ndarray:
    last = float(history[-1])
    return np.full((horizon,), last, dtype=float)

def moving_average(history: np.ndarray, horizon: int, window: int = 7) -> np.ndarray:
    w = min(window, len(history))
    v = float(np.mean(history[-w:]))
    return np.full((horizon,), v, dtype=float)

class RidgeMultiStep:
    """Train ridge to predict next step and roll forward."""

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        self.feature_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y_next: pd.Series, feature_cols: list[str]) -> "RidgeMultiStep":
        self.feature_cols = feature_cols
        self.model.fit(X[feature_cols].to_numpy(), y_next.to_numpy())
        return self

    def predict_recursive(self, row: pd.Series, horizon: int, lookback: int) -> np.ndarray:
        row = row.copy()
        preds = []
        for _ in range(horizon):
            x = row[self.feature_cols].to_numpy(dtype=float).reshape(1, -1)
            yhat = float(self.model.predict(x)[0])
            preds.append(yhat)
            for k in range(lookback, 1, -1):
                row[f"sales_lag_{k}"] = row.get(f"sales_lag_{k-1}")
            row["sales_lag_1"] = yhat
            row["roll_mean_7"] = np.mean([row.get(f"sales_lag_{i}") for i in range(1, min(7, lookback)+1)])
            row["roll_std_7"]  = np.std([row.get(f"sales_lag_{i}") for i in range(1, min(7, lookback)+1)])
        return np.array(preds, dtype=float)
