import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import dump
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import FEATURE_COLS, TSConfig, build_sequences_with_future_exog
from src.models.lstm import LSTMForecaster


def wape(y_true, y_pred):
    return float(
        np.sum(np.abs(y_true - y_pred))
        / np.maximum(np.sum(np.abs(y_true)), 1e-6)
        * 100
    )


def to_log_target(y: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(y, 0.0))


def from_log_target(y_log: np.ndarray) -> np.ndarray:
    return np.expm1(y_log)


def main(args):
    device = torch.device(args.device)

    df = pd.read_csv(args.data)

    cfg = TSConfig(
        lookback=args.lookback,
        horizon=args.horizon,
    )

    data = build_sequences_with_future_exog(df, cfg)

    os.makedirs("artifacts", exist_ok=True)

    all_metrics = []

    for sku, (X, y, dates) in data.items():
        print(f"\nTraining SKU: {sku}")

        dates = pd.to_datetime(dates)
        last = dates.max()

        test_cut = last - pd.Timedelta(days=args.test_days)
        val_cut = test_cut - pd.Timedelta(days=args.val_days)

        end_dts = dates + pd.Timedelta(days=args.horizon - 1)

        idx_train = np.where(end_dts <= val_cut)[0]
        idx_val = np.where((dates > val_cut) & (end_dts <= test_cut))[0]
        idx_test = np.where((dates > test_cut) & (end_dts <= last))[0]

        if len(idx_train) == 0:
            print("skip: no train data")
            continue

        X_past = X[:, : args.lookback, :]
        X_future = X[:, args.lookback :, :]

        Xp_train = X_past[idx_train]
        Xf_train = X_future[idx_train]
        y_train = y[idx_train]

        Xp_val = X_past[idx_val]
        Xf_val = X_future[idx_val]
        y_val = y[idx_val]

        Xp_test = X_past[idx_test]
        Xf_test = X_future[idx_test]
        y_test = y[idx_test]

        feat_scaler = StandardScaler()
        feat_scaler.fit(
            np.vstack(
                [
                    Xp_train.reshape(-1, Xp_train.shape[-1]),
                    Xf_train.reshape(-1, Xf_train.shape[-1]),
                ]
            )
        )

        target_transform = "log1p"

        y_train_t = to_log_target(y_train)
        y_val_t = to_log_target(y_val)

        targ_scaler = StandardScaler()
        targ_scaler.fit(y_train_t.reshape(-1, 1))

        def scale_feats(xp, xf):
            xp_s = feat_scaler.transform(
                xp.reshape(-1, xp.shape[-1])
            ).reshape(xp.shape)

            xf_s = feat_scaler.transform(
                xf.reshape(-1, xf.shape[-1])
            ).reshape(xf.shape)

            return xp_s, xf_s

        Xp_train, Xf_train = scale_feats(Xp_train, Xf_train)
        Xp_val, Xf_val = scale_feats(Xp_val, Xf_val)
        Xp_test, Xf_test = scale_feats(Xp_test, Xf_test)

        y_train_s = targ_scaler.transform(
            y_train_t.reshape(-1, 1)
        ).reshape(y_train.shape)

        y_val_s = targ_scaler.transform(
            y_val_t.reshape(-1, 1)
        ).reshape(y_val.shape)

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(Xp_train, dtype=torch.float32),
                torch.tensor(Xf_train, dtype=torch.float32),
                torch.tensor(y_train_s, dtype=torch.float32),
            ),
            batch_size=args.batch,
            shuffle=True,
        )

        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(Xp_val, dtype=torch.float32),
                torch.tensor(Xf_val, dtype=torch.float32),
                torch.tensor(y_val_s, dtype=torch.float32),
            ),
            batch_size=args.batch,
        )

        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(Xp_test, dtype=torch.float32),
                torch.tensor(Xf_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32),
            ),
            batch_size=args.batch,
        )

        effective_hidden_size = args.hidden_size

        model = LSTMForecaster(
            n_features=X.shape[2],
            hidden_size=effective_hidden_size,
            num_layers=args.num_layers,
            dropout=0.2,
            horizon=args.horizon,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
        )

        # SmoothL1Loss стабильнее для стандартизированного log1p(sales).
        # WAPE нельзя использовать как loss после StandardScaler:
        # сумма target может быть около нуля, и обучение становится нестабильным.
        loss_fn = nn.SmoothL1Loss()

        best_state = None
        best_val_wape = float("inf")
        patience_left = args.patience

        for epoch in range(args.epochs):
            model.train()
            train_losses = []

            for xb_p, xb_f, yb in train_loader:
                xb_p = xb_p.to(device)
                xb_f = xb_f.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()

                yp = model(xb_p, xb_f)
                loss = loss_fn(yp, yb)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                optimizer.step()

                train_losses.append(loss.item())

            model.eval()

            val_losses = []
            val_true_all = []
            val_pred_all = []

            with torch.no_grad():
                for xb_p, xb_f, yb in val_loader:
                    xb_p = xb_p.to(device)
                    xb_f = xb_f.to(device)
                    yb = yb.to(device)

                    yp = model(xb_p, xb_f)

                    val_losses.append(
                        loss_fn(yp, yb).item()
                    )

                    yp_s = yp.cpu().numpy()

                    yp_log = targ_scaler.inverse_transform(
                        yp_s.reshape(-1, 1)
                    ).reshape(yp_s.shape)

                    yp_orig = np.maximum(
                        from_log_target(yp_log),
                        0.0,
                    )

                    yb_np = yb.cpu().numpy()

                    yt_log = targ_scaler.inverse_transform(
                        yb_np.reshape(-1, 1)
                    ).reshape(yb_np.shape)

                    yt_orig = np.maximum(
                        from_log_target(yt_log),
                        0.0,
                    )

                    val_pred_all.append(yp_orig)
                    val_true_all.append(yt_orig)

            val_pred_all = (
                np.concatenate(val_pred_all)
                if val_pred_all
                else np.empty((0, args.horizon))
            )

            val_true_all = (
                np.concatenate(val_true_all)
                if val_true_all
                else np.empty((0, args.horizon))
            )

            val_w = (
                wape(val_true_all, val_pred_all)
                if len(val_true_all)
                else float("inf")
            )

            print(
                f"epoch {epoch + 1}: "
                f"train={np.mean(train_losses):.2f}, "
                f"val={np.mean(val_losses):.2f}, "
                f"val_wape={val_w:.2f}%"
            )

            if val_w < best_val_wape:
                best_val_wape = val_w
                best_state = copy.deepcopy(model.state_dict())
                patience_left = args.patience
            else:
                patience_left -= 1

                if patience_left <= 0:
                    print("early stop")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Калибровка только по validation, без утечки test.
        val_pred_cal = []
        val_true_cal = []

        model.eval()

        with torch.no_grad():
            for xb_p, xb_f, yb in val_loader:
                xb_p = xb_p.to(device)
                xb_f = xb_f.to(device)

                yp_s = model(xb_p, xb_f).cpu().numpy()

                yp_log = targ_scaler.inverse_transform(
                    yp_s.reshape(-1, 1)
                ).reshape(yp_s.shape)

                yp = np.maximum(
                    from_log_target(yp_log),
                    0.0,
                )

                yt_log = targ_scaler.inverse_transform(
                    yb.numpy().reshape(-1, 1)
                ).reshape(yb.shape)

                yt = np.maximum(
                    from_log_target(yt_log),
                    0.0,
                )

                val_pred_cal.append(yp)
                val_true_cal.append(yt)

        val_pred_cal = (
            np.concatenate(val_pred_cal)
            if val_pred_cal
            else np.empty((0, args.horizon))
        )

        val_true_cal = (
            np.concatenate(val_true_cal)
            if val_true_cal
            else np.empty((0, args.horizon))
        )

        if len(val_pred_cal):
            scale = np.sum(val_true_cal) / np.maximum(
                np.sum(val_pred_cal),
                1e-6,
            )
            bias = np.mean(val_true_cal - val_pred_cal * scale)
        else:
            scale = 1.0
            bias = 0.0

        yp_all = []
        yt_all = []

        with torch.no_grad():
            for xb_p, xb_f, yb in test_loader:
                xb_p = xb_p.to(device)
                xb_f = xb_f.to(device)

                yp_s = model(xb_p, xb_f).cpu().numpy()

                yp_log = targ_scaler.inverse_transform(
                    yp_s.reshape(-1, 1)
                ).reshape(yp_s.shape)

                yp = np.maximum(
                    from_log_target(yp_log),
                    0.0,
                )

                yt = np.maximum(
                    yb.numpy(),
                    0.0,
                )

                yp_all.append(yp)
                yt_all.append(yt)

        yp_all = (
            np.concatenate(yp_all)
            if yp_all
            else np.empty((0, args.horizon))
        )

        yt_all = (
            np.concatenate(yt_all)
            if yt_all
            else np.empty((0, args.horizon))
        )

        yp_all = np.maximum(
            yp_all * scale + bias,
            0.0,
        )

        mae = np.mean(np.abs(yt_all - yp_all))
        rmse = np.sqrt(np.mean((yt_all - yp_all) ** 2))
        w = wape(yt_all, yp_all)

        print(
            f"{sku} → MAE={mae:.2f} "
            f"RMSE={rmse:.2f} "
            f"WAPE={w:.2f}%"
        )

        all_metrics.append(
            {
                "sku": sku,
                "mae": float(mae),
                "rmse": float(rmse),
                "wape": float(w),
            }
        )

        sku_dir = os.path.join("artifacts", sku)
        os.makedirs(sku_dir, exist_ok=True)

        torch.save(
            {
                "state_dict": model.state_dict(),
                "calibration": {
                    "scale": float(scale),
                    "bias": float(bias),
                },
                "lookback": args.lookback,
                "horizon": args.horizon,
                "hidden_size": effective_hidden_size,
                "num_layers": args.num_layers,
                "target_transform": target_transform,
                "feature_cols": list(FEATURE_COLS),
            },
            os.path.join(sku_dir, "model.pt"),
        )

        dump(
            feat_scaler,
            os.path.join(sku_dir, "feature_scaler.joblib"),
        )

        dump(
            targ_scaler,
            os.path.join(sku_dir, "target_scaler.joblib"),
        )

        with open(
            os.path.join(sku_dir, "metrics_nn.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "wape": float(w),
                },
                f,
                indent=2,
            )

    with open(
        "artifacts/metrics_nn_all.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            all_metrics,
            f,
            indent=2,
        )

    print("Saved artifacts/metrics_nn_all.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default="data/sales.csv",
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=28,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=14,
    )

    parser.add_argument(
        "--test-days",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--val-days",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    args = parser.parse_args()
    main(args)
