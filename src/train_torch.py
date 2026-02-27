"""Train PyTorch LSTM per-SKU (Milestone 3)."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dataset import TSConfig, build_sequences_by_sku, FEATURE_COLS
from .models.lstm import LSTMForecaster

def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _mape(y_true, y_pred):
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
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    df = pd.read_csv(args.data, parse_dates=["date"]).sort_values(["sku","date"]).reset_index(drop=True)

    cfg = TSConfig(lookback=args.lookback, horizon=args.horizon)
    by_sku = build_sequences_by_sku(df, cfg)

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for sku, (X, y, dates) in by_sku.items():
        # time split by last days
        last_date = pd.to_datetime(dates[-1])
        test_cut = last_date - pd.Timedelta(days=args.test_days)
        val_cut  = test_cut - pd.Timedelta(days=args.val_days)

        date_series = pd.to_datetime(dates)

        idx_train = np.where(date_series <= val_cut)[0]
        idx_val   = np.where((date_series > val_cut) & (date_series <= test_cut))[0]
        idx_test  = np.where(date_series > test_cut)[0]

        if len(idx_train) < 50 or len(idx_test) < 10:
            continue

        # scale features and target per SKU
        feat_scaler = StandardScaler()
        targ_scaler = StandardScaler()

        X_train_flat = X[idx_train].reshape(-1, X.shape[-1])
        feat_scaler.fit(X_train_flat)

        y_train_flat = y[idx_train].reshape(-1, 1)
        targ_scaler.fit(y_train_flat)

        def scale_X(A):
            B = feat_scaler.transform(A.reshape(-1, A.shape[-1])).reshape(A.shape)
            return B

        def scale_y(A):
            B = targ_scaler.transform(A.reshape(-1, 1)).reshape(A.shape)
            return B

        Xtr = scale_X(X[idx_train]); ytr = scale_y(y[idx_train])
        Xva = scale_X(X[idx_val]) if len(idx_val) else None
        yva = scale_y(y[idx_val]) if len(idx_val) else None
        Xte = scale_X(X[idx_test]); yte = scale_y(y[idx_test])

        train_loader = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)),
                                  batch_size=args.batch, shuffle=True)
        val_loader = None
        if Xva is not None and len(Xva):
            val_loader = DataLoader(TensorDataset(torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)),
                                    batch_size=args.batch, shuffle=False)
        test_loader = DataLoader(TensorDataset(torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.float32)),
                                 batch_size=args.batch, shuffle=False)

        model = LSTMForecaster(n_features=len(FEATURE_COLS), hidden_size=64, num_layers=2, dropout=0.1, horizon=args.horizon).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.MSELoss()

        best = float("inf")
        best_state = None
        patience = 5
        bad = 0

        def eval_loader(loader):
            model.eval()
            ys, ps = [], []
            with torch.no_grad():
                for xb, yb in loader:
                    xb = xb.to(device); yb = yb.to(device)
                    pred = model(xb)
                    ys.append(yb.cpu().numpy()); ps.append(pred.cpu().numpy())
            yt = np.concatenate(ys, axis=0); yp = np.concatenate(ps, axis=0)
            return float(np.mean(np.abs(yt-yp)))

        for epoch in range(1, args.epochs+1):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            val_mae = eval_loader(val_loader or test_loader)
            if val_mae < best:
                best = val_mae
                best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # test metrics in original scale
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                pred = model(xb).cpu().numpy()
                preds.append(pred); trues.append(yb.numpy())
        yp = np.concatenate(preds, axis=0)
        yt = np.concatenate(trues, axis=0)

        # inverse scale
        yp_inv = targ_scaler.inverse_transform(yp.reshape(-1,1)).reshape(yp.shape)
        yt_inv = targ_scaler.inverse_transform(yt.reshape(-1,1)).reshape(yt.shape)

        mae = float(mean_absolute_error(yt_inv.reshape(-1), yp_inv.reshape(-1)))
        mse = float(mean_squared_error(yt_inv.reshape(-1), yp_inv.reshape(-1)))
        rmse = _rmse(yt_inv.reshape(-1), yp_inv.reshape(-1))
        mape = _mape(yt_inv.reshape(-1), yp_inv.reshape(-1))

        sku_dir = Path("artifacts")/sku
        sku_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            "state_dict": model.state_dict(),
            "lookback": args.lookback,
            "horizon": args.horizon,
            "feature_cols": FEATURE_COLS,
        }, sku_dir/"model.pt")

        import joblib
        joblib.dump(feat_scaler, sku_dir/"feature_scaler.joblib")
        joblib.dump(targ_scaler, sku_dir/"target_scaler.joblib")

        sku_metrics = {"sku": sku, "mae": mae, "mse": mse, "rmse": rmse, "mape": mape}
        with open(sku_dir/"metrics_nn.json", "w", encoding="utf-8") as f:
            json.dump(sku_metrics, f, ensure_ascii=False, indent=2)

        all_metrics.append(sku_metrics)
        print(f"{sku}: MAE={mae:.3f} RMSE={rmse:.3f}")

    with open("artifacts/metrics_nn_all.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print("Saved artifacts per SKU and artifacts/metrics_nn_all.json")

if __name__ == "__main__":
    main()
