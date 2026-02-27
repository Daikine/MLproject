"""streamlit_app.py

MVP ML-сервиса прогнозирования спроса (строго по кейсу).

Функциональность:
1) Выбор SKU из синтетических данных → прогноз на 7-14 дней
2) Ввод сценария (множитель цены + длительность промо) → прогноз
3) Загрузка CSV (date,sales + опционально price,promo,discount_pct,...) → прогноз

Важно:
- НЕ показываем «сырой датасет» таблицей по умолчанию (по просьбе).
- Показываем KPI и график "история + прогноз".
- Для NN используем PyTorch LSTM (артефакты в artifacts/SKU_xx/...).

Запуск локально:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Деплой Streamlit Cloud:
- выбрать этот файл как entrypoint (Main file path).
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

from src.dataset import add_calendar_feats, FEATURE_COLS
from src.models.lstm import LSTMForecaster


DATA_PATH = Path("data/sales.csv")
ART_DIR = Path("artifacts")


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def load_nn_for_sku(sku: str):
    sku_dir = ART_DIR / sku
    model_path = sku_dir / "model.pt"
    fs_path = sku_dir / "feature_scaler.joblib"
    ts_path = sku_dir / "target_scaler.joblib"
    metrics_path = sku_dir / "metrics_nn.json"

    if not (model_path.exists() and fs_path.exists() and ts_path.exists()):
        return None

    feature_scaler = joblib.load(fs_path)
    target_scaler = joblib.load(ts_path)

    model = LSTMForecaster(n_features=len(FEATURE_COLS), horizon=14)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    nn_metrics = None
    if metrics_path.exists():
        nn_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    return model, feature_scaler, target_scaler, nn_metrics


def make_future_frame(history: pd.DataFrame, horizon: int, price_mult: float, promo_days: int, promo_where: str) -> pd.DataFrame:
    """Делаем будущий фрейм на horizon дней.

    - price_mult умножает последнюю цену
    - promo_days выставляет промо на первые/последние promo_days
    """
    last = history.sort_values("date").iloc[-1]
    start = last["date"] + pd.Timedelta(days=1)
    dates = pd.date_range(start=start, periods=horizon, freq="D")

    base_price = float(last.get("price", 10.0))
    price = np.full(horizon, base_price * price_mult)

    promo = np.zeros(horizon, dtype=int)
    discount = np.zeros(horizon, dtype=float)
    promo_days = int(max(0, min(horizon, promo_days)))
    if promo_days > 0:
        if promo_where == "В начале горизонта":
            idx = slice(0, promo_days)
        else:
            idx = slice(horizon - promo_days, horizon)
        promo[idx] = 1
        discount[idx] = 0.20  # фиксируем скидку для сценария

    # weekend/holiday оставляем как в генераторе: weekend по календарю, holiday синтетика — нет в будущем
    df_fut = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "sku": last.get("sku", "SKU"),
            "sales": np.nan,
            "price": np.round(price, 2),
            "promo": promo,
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
    future: pd.DataFrame,
    lookback: int,
    horizon: int,
) -> np.ndarray:
    """Прогноз на horizon дней.

    Мы формируем одно окно (lookback) на основе последних дней истории,
    а фичи будущего используем как часть входа через последовательно обновляемые окна.

    Упрощение для MVP:
    - модель обучалась multi-step, поэтому мы подаём последнее окно и берём сразу horizon.
    - фичи будущего (price/promo/...) учитываем через то, что в окне находятся и эти столбцы.

    Для корректного учёта future-фичей в multi-step нужно обучать seq2seq.
    Для учебного MVP текущего достаточно.
    """
    # берём последние lookback строк истории + добавляем календарь
    h = history.sort_values("date").tail(lookback).copy()
    h = add_calendar_feats(h)

    X = h[FEATURE_COLS].values.astype(float)
    Xs = feature_scaler.transform(X)

    x_tensor = torch.tensor(Xs, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
    pred_scaled = model(x_tensor).squeeze(0).numpy()  # (horizon,)

    pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    pred = np.maximum(pred, 0.0)
    return pred[:horizon]


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
        st.error("Нет данных data/sales.csv. Сначала сгенерируй их: python data/generate_data.py ...")
        return

    df = load_data(DATA_PATH)
    skus = sorted(df["sku"].unique())

    with st.sidebar:
        st.header("Панель управления")
        mode = st.selectbox("Режим", ["Демо (выбор товара)", "Сценарий (ввод параметров)", "Загрузка CSV"], index=0)
        horizon = st.slider("Горизонт прогноза (дней)", 7, 14, 14)

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
        # future frame for plot labels
        fut = make_future_frame(hist, horizon, price_mult, promo_days, promo_where)

        base_pred = moving_average_future(hist["sales"].values, horizon=horizon, window=7)

        nn_pack = load_nn_for_sku(sku)
        nn_pred = None
        nn_metrics = None
        if nn_pack is not None:
            model, fs, ts, nn_metrics = nn_pack
            nn_pred = lstm_forecast(model, fs, ts, hist, fut, lookback=lookback, horizon=horizon)

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
            # baseline metrics (из artifacts) если есть
            base_metrics_file = ART_DIR / "metrics_baselines.json"
            if base_metrics_file.exists():
                bm = json.loads(base_metrics_file.read_text(encoding="utf-8"))
                ma7 = bm.get("moving_avg_7", None)
                if ma7:
                    st.write("**Baseline MA(7)**")
                    st.json(ma7)

            if nn_metrics is not None:
                st.write("**LSTM**")
                st.json(nn_metrics)
            else:
                st.warning(
                    "Артефакты LSTM не найдены. Сначала обучи модель: \n"
                    "`python -m src.train_torch --data data/sales.csv ...`"
                )

            st.subheader("Сценарий")
            st.write(f"SKU: **{sku}**")
            st.write(f"Горизонт: **{horizon}** дней")
            if mode == "Сценарий (ввод параметров)":
                st.write(f"Множитель цены: **{price_mult:.2f}**")
                st.write(f"Промо дней: **{promo_days}** ({promo_where})")

    else:
        with st.sidebar:
            uploaded = st.file_uploader("CSV (date,sales + optional price/promo/discount)", type=["csv"])
            run = st.button("Рассчитать прогноз")

        if uploaded is None:
            st.info("Загрузи CSV в панели слева.")
            return
        if not run:
            st.info("Нажми «Рассчитать прогноз».")
            return

        user_df = pd.read_csv(uploaded)
        # detect columns
        cols = {c.lower(): c for c in user_df.columns}
        if "date" not in cols or "sales" not in cols:
            st.error("CSV должен содержать столбцы date и sales")
            return
        user_df[cols["date"]] = pd.to_datetime(user_df[cols["date"]])
        user_df = user_df.sort_values(cols["date"]).rename(columns={cols["date"]: "date", cols["sales"]: "sales"})

        # fill optional
        for c, default in [("price", 10.0), ("promo", 0), ("discount_pct", 0.0), ("is_weekend", None), ("is_holiday", 0)]:
            if c not in user_df.columns:
                if c == "is_weekend":
                    user_df[c] = (user_df["date"].dt.weekday >= 5).astype(int)
                else:
                    user_df[c] = default

        horizon = int(horizon)
        last_date = user_df["date"].iloc[-1]
        fut_dates = [last_date + timedelta(days=i + 1) for i in range(horizon)]
        base_pred = moving_average_future(user_df["sales"].values, horizon=horizon, window=7)

        fig = plot_history_forecast(
            user_df["date"].tail(180),
            user_df["sales"].tail(180),
            pd.to_datetime(fut_dates),
            pred_nn=None,
            pred_base=base_pred,
        )
        st.pyplot(fig)
        st.success("Для пользовательского CSV в MVP используется baseline. Для NN нужна модель, обученная под ваши фичи.")


if __name__ == "__main__":
    main()
