"""
streamlit_app.py

MVP ML-сервиса прогнозирования спроса (строго по кейсу).

Функциональность:
1) Выбор SKU из синтетических данных → прогноз на 7-14 дней
2) Ввод сценария (множитель цены + длительность промо) → прогноз
3) Загрузка CSV (date,sales + опционально price,promo_flag,discount_pct,...) → прогноз

Важно:
- НЕ показываем «сырой датасет» таблицей по умолчанию.
- Показываем KPI и график "история + прогноз".
- Для NN используем PyTorch LSTM (артефакты в artifacts/SKU_xx/...).

Запуск локально:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Деплой Streamlit Cloud:
- выбрать этот файл как entrypoint (Main file path = streamlit_app.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import torch
import joblib

# --- Импорты из проекта ---
from src.models.lstm import LSTMForecaster

# FEATURE_COLS нужен 100%, add_calendar_feats может отсутствовать по имени, поэтому делаем fallback
try:
    from src.dataset import add_calendar_feats, FEATURE_COLS  # type: ignore
except Exception:
    from src.dataset import FEATURE_COLS  # type: ignore

    def add_calendar_feats(df: pd.DataFrame) -> pd.DataFrame:
        """Fallback: добавление календарных признаков (если нет add_calendar_feats в src.dataset)."""
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])
        out["dow"] = out["date"].dt.dayofweek.astype(int)
        out["month"] = out["date"].dt.month.astype(int)
        out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
        out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)
        out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
        out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)

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


DATA_PATH = Path("data/sales.csv")
ART_DIR = Path("artifacts")


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def load_nn_for_sku(sku: str):
    """
    Загружаем модель и скейлеры для конкретного SKU из artifacts/{sku}/

    Ожидаемые файлы:
      - model.pt (torch.save({state_dict, horizon, feature_cols, lookback, ...}))
      - feature_scaler.joblib
      - target_scaler.joblib
      - metrics_nn.json (опционально)
    """
    sku_dir = ART_DIR / sku
    model_path = sku_dir / "model.pt"
    fs_path = sku_dir / "feature_scaler.joblib"
    ts_path = sku_dir / "target_scaler.joblib"
    metrics_path = sku_dir / "metrics_nn.json"

    if not (model_path.exists() and fs_path.exists() and ts_path.exists()):
        return None

    feature_scaler = joblib.load(fs_path)
    target_scaler = joblib.load(ts_path)

    ckpt = torch.load(model_path, map_location="cpu")

    # Правильный формат (как в train_torch.py): dict со state_dict + метаданные
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        horizon_ckpt = int(ckpt.get("horizon", 14))
        feature_cols_ckpt = ckpt.get("feature_cols", FEATURE_COLS)

        model = LSTMForecaster(
            n_features=len(feature_cols_ckpt),
            hidden_size=64,
            num_layers=2,
            dropout=0.1,
            horizon=horizon_ckpt,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
    else:
        # Запасной вариант: если в model.pt лежит чистый state_dict
        horizon_ckpt = 14
        feature_cols_ckpt = FEATURE_COLS

        model = LSTMForecaster(
            n_features=len(FEATURE_COLS),
            hidden_size=64,
            num_layers=2,
            dropout=0.1,
            horizon=14,
        )
        model.load_state_dict(ckpt)
        model.eval()

    nn_metrics = None
    if metrics_path.exists():
        nn_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    return model, feature_scaler, target_scaler, nn_metrics, feature_cols_ckpt, horizon_ckpt


def make_future_frame(
    history: pd.DataFrame,
    horizon: int,
    price_mult: float,
    promo_days: int,
    promo_where: str,
) -> pd.DataFrame:
    """Делаем будущий фрейм на horizon дней для отображения и сценарных экзогенных фич."""
    last = history.sort_values("date").iloc[-1]
    start = last["date"] + pd.Timedelta(days=1)
    dates = pd.date_range(start=start, periods=horizon, freq="D")

    base_price = float(last.get("price", 10.0))
    price = np.full(horizon, base_price * price_mult)

    promo_flag = np.zeros(horizon, dtype=int)
    discount = np.zeros(horizon, dtype=float)
    promo_days = int(max(0, min(horizon, promo_days)))

    if promo_days > 0:
        if promo_where == "В начале горизонта":
            idx = slice(0, promo_days)
        else:
            idx = slice(horizon - promo_days, horizon)
        promo_flag[idx] = 1
        discount[idx] = 0.20  # фиксируем скидку для сценария

    df_fut = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "sku": last.get("sku", "SKU"),
            "sales": np.nan,
            "price": np.round(price, 2),
            "promo_flag": promo_flag,
            "discount_pct": np.round(discount, 3),
            "is_weekend": (pd.Series(dates).dt.weekday >= 5).astype(int).values,
            "is_holiday": np.zeros(horizon, dtype=int),
        }
    )
    return df_fut


@torch.no_grad()
def lstm_forecast(
    model: LSTMForecaster,
    feature_scaler,
    target_scaler,
    history: pd.DataFrame,
    lookback: int,
    horizon: int,
    feature_cols: list[str],
) -> np.ndarray:
    """
    Прогноз на horizon дней (multi-step).

    Защита от ошибок:
    - history приводим к DataFrame
    - add_calendar_feats может вернуть None → тогда оставляем исходный df
    - гарантируем наличие нужных feature_cols
    """
    # 1) гарантируем DataFrame
    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame(history)

    h = history.sort_values("date").tail(int(lookback)).copy()

    # 2) календарные фичи
    try:
        h2 = add_calendar_feats(h)
        if isinstance(h2, pd.DataFrame):
            h = h2
    except Exception:
        pass  # просто работаем с тем, что есть

    # 3) гарантируем экзогенные колонки
    defaults = {
        "price": 1.0,
        "promo_flag": 0,
        "discount_pct": 0.0,
        "is_weekend": (pd.to_datetime(h["date"]).dt.weekday >= 5).astype(int) if "date" in h.columns else 0,
        "is_holiday": 0,
        "dow_sin": 0.0,
        "dow_cos": 0.0,
        "month_sin": 0.0,
        "month_cos": 0.0,
        "sales": 0.0,
    }

    for col in feature_cols:
        if col not in h.columns:
            val = defaults.get(col, 0.0)
            # если val серия (как is_weekend), то длина должна совпадать
            h[col] = val

    # 4) формируем окно
    X = h[feature_cols].to_numpy(dtype=float)
    Xs = feature_scaler.transform(X)

    x_tensor = torch.tensor(Xs, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
    pred_scaled = model(x_tensor).squeeze(0).cpu().numpy()  # (horizon_ckpt,)

    pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    pred = np.maximum(pred, 0.0)
    return pred[: int(horizon)]


def moving_average_future(history_sales: np.ndarray, horizon: int, window: int = 7) -> np.ndarray:
    hist = list(history_sales.astype(float))
    preds = []
    for _ in range(horizon):
        w = hist[-window:] if len(hist) >= window else hist
        p = float(np.mean(w))
        preds.append(p)
        hist.append(p)
    return np.array(preds)


def plot_history_forecast(dates_hist, sales_hist, dates_fut, pred_nn, pred_base):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates_hist, sales_hist, label="История продаж")
    if pred_base is not None:
        ax.plot(dates_fut, pred_base, linestyle="--", marker="o", label="Baseline MA(7)")
    if pred_nn is not None:
        ax.plot(dates_fut, pred_nn, linestyle="--", marker="o", label="LSTM прогноз")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Продажи")
    ax.grid(True)
    ax.legend()
    fig.autofmt_xdate()
    return fig


def main():
    st.set_page_config(page_title="Прогноз спроса", layout="wide")
    st.title("Нейросетевая модель для прогнозирования спроса на товары")

    if not DATA_PATH.exists():
        st.error(
            "Нет данных data/sales.csv.\n\n"
            "Сначала сгенерируй их: `python data/generate_data.py --output data/sales.csv ...`"
        )
        return

    df = load_data(DATA_PATH)
    df = df.sort_values(["sku", "date"]).reset_index(drop=True)

    if "promo" in df.columns and "promo_flag" not in df.columns:
        df = df.rename(columns={"promo": "promo_flag"})

    skus = sorted(df["sku"].unique())

    with st.sidebar:
        st.header("Панель управления")
        mode = st.selectbox(
            "Режим",
            ["Демо (выбор товара)", "Сценарий (ввод параметров)", "Загрузка CSV"],
            index=0,
        )
        horizon = st.slider("Горизонт прогноза (дней)", 7, 14, 14)

    # =======================
    # 1) ДЕМО / СЦЕНАРИЙ
    # =======================
    if mode in {"Демо (выбор товара)", "Сценарий (ввод параметров)"}:
        with st.sidebar:
            sku = st.selectbox("Товар (SKU)", skus)
            lookback = st.slider("Окно истории (lookback)", 14, 60, 28)

            price_mult = 1.0
            promo_days = 0
            promo_where = "В начале горизонта"

            if mode == "Сценарий (ввод параметров)":
                st.subheader("Сценарий")
                price_mult = st.number_input("Множитель цены (например 0.9 = дешевле)", 0.5, 2.0, 1.0, 0.05)
                promo_days = st.slider("Сколько дней промо", 0, 14, 0)
                promo_where = st.selectbox("Где промо", ["В начале горизонта", "В конце горизонта"], index=0)

            run = st.button("Рассчитать прогноз")

        if not run:
            st.info("Выбери параметры и нажми «Рассчитать прогноз».")
            return

        hist = df[df["sku"] == sku].sort_values("date").reset_index(drop=True)
        fut = make_future_frame(hist, horizon, price_mult, promo_days, promo_where)

        base_pred = moving_average_future(hist["sales"].values, horizon=horizon, window=7)

        nn_pack = load_nn_for_sku(sku)
        nn_pred = None
        nn_metrics = None

        if nn_pack is not None:
            model, fs, ts, nn_metrics, feature_cols_ckpt, horizon_ckpt = nn_pack

            # прогноз на выбранный horizon (не больше horizon из чекпоинта)
            horizon_used = min(int(horizon), int(horizon_ckpt))
            nn_pred = lstm_forecast(
                model=model,
                feature_scaler=fs,
                target_scaler=ts,
                history=hist,
                lookback=int(lookback),
                horizon=horizon_used,
                feature_cols=list(feature_cols_ckpt),
            )

            # если пользователь выбрал horizon больше, дополним хвост последним значением
            if horizon_used < horizon:
                nn_pred = np.pad(nn_pred, (0, horizon - horizon_used), constant_values=float(nn_pred[-1]))

        # Layout
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = plot_history_forecast(
                hist["date"].tail(180),
                hist["sales"].tail(180),
                fut["date"],
                nn_pred,
                base_pred,
            )
            st.pyplot(fig)

        with c2:
            st.subheader("Метрики")

            # baseline metrics: artifacts/metrics_baselines.json это LIST, найдём строку для sku+moving_avg_7
            base_metrics_file = ART_DIR / "metrics_baselines.json"
            if base_metrics_file.exists():
                bm = json.loads(base_metrics_file.read_text(encoding="utf-8"))
                row = next((r for r in bm if r.get("sku") == sku and r.get("model") == "moving_avg_7"), None)
                if row:
                    st.write("**Baseline MA(7)**")
                    st.json(row)
                else:
                    st.info("Baseline метрики не найдены для выбранного SKU.")
            else:
                st.info("Файл baseline метрик ещё не создан (запусти evaluate_baselines/train workflow).")

            if nn_metrics is not None:
                st.write("**LSTM**")
                st.json(nn_metrics)
            else:
                st.warning(
                    "Артефакты LSTM не найдены для этого SKU.\n\n"
                    "Запусти обучение: `python -m src.train_torch --data data/sales.csv ...` "
                    "или дождись завершения GitHub Actions."
                )

            st.subheader("Сценарий")
            st.write(f"SKU: **{sku}**")
            st.write(f"Горизонт: **{horizon}** дней")
            if mode == "Сценарий (ввод параметров)":
                st.write(f"Множитель цены: **{price_mult:.2f}**")
                st.write(f"Промо дней: **{promo_days}** ({promo_where})")

    # =======================
    # 2) ЗАГРУЗКА CSV
    # =======================
    else:
        with st.sidebar:
            uploaded = st.file_uploader("CSV (date,sales + optional price,promo_flag,discount_pct,...)", type=["csv"])
            run = st.button("Рассчитать прогноз")

        if uploaded is None:
            st.info("Загрузи CSV в панели слева.")
            return
        if not run:
            st.info("Нажми «Рассчитать прогноз».")
            return

        user_df = pd.read_csv(uploaded)
        cols = {c.lower(): c for c in user_df.columns}

        if "date" not in cols or "sales" not in cols:
            st.error("CSV должен содержать столбцы date и sales")
            return

        user_df[cols["date"]] = pd.to_datetime(user_df[cols["date"]])
        user_df = user_df.sort_values(cols["date"]).rename(columns={cols["date"]: "date", cols["sales"]: "sales"})

        # нормализуем возможное имя promo -> promo_flag
        if "promo" in cols and "promo_flag" not in user_df.columns:
            user_df = user_df.rename(columns={cols["promo"]: "promo_flag"})
        if "promo_flag" not in user_df.columns:
            user_df["promo_flag"] = 0

        # optional columns
        if "price" not in user_df.columns:
            user_df["price"] = 10.0
        if "discount_pct" not in user_df.columns:
            user_df["discount_pct"] = 0.0
        if "is_weekend" not in user_df.columns:
            user_df["is_weekend"] = (user_df["date"].dt.weekday >= 5).astype(int)
        if "is_holiday" not in user_df.columns:
            user_df["is_holiday"] = 0

        last_date = user_df["date"].iloc[-1]
        fut_dates = [last_date + timedelta(days=i + 1) for i in range(int(horizon))]
        base_pred = moving_average_future(user_df["sales"].values, horizon=int(horizon), window=7)

        fig = plot_history_forecast(
            user_df["date"].tail(180),
            user_df["sales"].tail(180),
            pd.to_datetime(fut_dates),
            pred_nn=None,
            pred_base=base_pred,
        )
        st.pyplot(fig)
        st.success(
            "Для пользовательского CSV в MVP используется baseline.\n\n"
            "Чтобы включить NN для вашего CSV, нужно обучать модель на ваших фичах (или привести фичи к нашему формату)."
        )


if __name__ == "__main__":
    main()
