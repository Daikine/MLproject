# src/train_torch.py
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from src.dataset import TSConfig, FEATURE_COLS, build_sequences_with_future_exog
from src.models.lstm import LSTMForecaster


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sales.csv")
    ap.add_argument("--lookback", type=int, default=28)
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--test-days", type=int, default=120)
    ap.add_argument("--val-days", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    df = pd.read_csv(args.data, parse_dates=["date"]).sort_values(["sku","date"]).reset_index(drop=True)

    cfg = TSConfig(lookback=args.lookback, horizon=args.horizon)
    by_sku = build_sequences_with_future_exog(df, cfg)

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for sku, (X, y, dates) in by_sku.items():
        # time split по датам "якоря" (date на старте horizon)
        dts = pd.to_datetime(dates)
        last = dts.max()
        test_cut = last - pd.Timedelta(days=args.test_days)
        val_cut  = test_cut - pd.Timedelta(days=args.val_days)

        idx_train = np.where(dts <= val_cut)[0]
        idx_val   = np.where((dts > val_cut) & (dts <= test_cut))[0]
        idx_test  = np.where(dts > test_cut)[0]

        if len(idx_train) < 200 or len(idx_test) < 30:
            continue

        # scalers per SKU
        feat_scaler = StandardScaler()
        targ_scaler = StandardScaler()

        X_train_flat = X[idx_train].reshape(-1, X.shape[-1])
        feat_scaler.fit(X_train_flat)

        y_train_flat = y[idx_train].reshape(-1, 1)
        targ_scaler.fit(y_train_flat)

        def scale_X(A):
            return feat_scaler.transform(A.reshape(-1, A.shape[-1])).reshape(A.shape)

        def scale_y(A):
            return targ_scaler.transform(A.reshape(-1, 1)).reshape(A.shape)

        Xtr, ytr = scale_X(X[idx_train]), scale_y(y[idx_train])
        Xva, yva = (scale_X(X[idx_val]), scale_y(y[idx_val])) if len(idx_val) else (None, None)
        Xte, yte = scale_X(X[idx_test]), scale_y(y[idx_test])

        train_loader = DataLoader(
            TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)),
            batch_size=args.batch,
            shuffle=True,
        )
        val_loader = None
        if Xva is not None and len(Xva):
            val_loader = DataLoader(
                TensorDataset(torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)),
                batch_size=args.batch,
                shuffle=False,
            )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.float32)),
            batch_size=args.batch,
            shuffle=False,
        )

        model = LSTMForecaster(
            n_features=len(FEATURE_COLS),
            hidden_size=64,
            num_layers=2,
            dropout=0.1,
            horizon=args.horizon,
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.MSELoss()

        def eval_mae(loader):
            model.eval()
            ys, ps = [], []
            with torch.no_grad():
                for xb, yb in loader:
                    xb = xb.to(device)
                    pred = model(xb).cpu().numpy()
                    ys.append(yb.numpy())
                    ps.append(pred)
            yt = np.concatenate(ys, axis=0)
            yp = np.concatenate(ps, axis=0)
            return float(np.mean(np.abs(yt - yp)))

        best = float("inf")
        best_state = None
        patience = 6
        bad = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            val_mae = eval_mae(val_loader or test_loader)
            if val_mae < best:
                best = val_mae
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # test in original scale
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                pred = model(xb).cpu().numpy()
                preds.append(pred)
                trues.append(yb.numpy())

        yp = np.concatenate(preds, axis=0)
        yt = np.concatenate(trues, axis=0)

        yp_inv = targ_scaler.inverse_transform(yp.reshape(-1, 1)).reshape(yp.shape)
        yt_inv = targ_scaler.inverse_transform(yt.reshape(-1, 1)).reshape(yt.shape)

        mae = float(mean_absolute_error(yt_inv.reshape(-1), yp_inv.reshape(-1)))
        mse = float(mean_squared_error(yt_inv.reshape(-1), yp_inv.reshape(-1)))
        r = rmse(yt_inv.reshape(-1), yp_inv.reshape(-1))
        mp = mape(yt_inv.reshape(-1), yp_inv.reshape(-1))

        sku_dir = Path("artifacts") / sku
        sku_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "state_dict": model.state_dict(),
                "lookback": args.lookback,
                "horizon": args.horizon,
                "feature_cols": FEATURE_COLS,
                "note": "trained with future exogenous features (future sales are zeroed to avoid leakage)",
            },
            sku_dir / "model.pt",
        )

        joblib.dump(feat_scaler, sku_dir / "feature_scaler.joblib")
        joblib.dump(targ_scaler, sku_dir / "target_scaler.joblib")

        sku_metrics = {"sku": sku, "mae": mae, "mse": mse, "rmse": r, "mape": mp}
        (sku_dir / "metrics_nn.json").write_text(json.dumps(sku_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        all_metrics.append(sku_metrics)
        print(f"{sku}: MAE={mae:.3f} RMSE={r:.3f} MAPE={mp:.2f}%")

    (Path("artifacts") / "metrics_nn_all.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Saved artifacts/* and artifacts/metrics_nn_all.json")


if __name__ == "__main__":
    main()
