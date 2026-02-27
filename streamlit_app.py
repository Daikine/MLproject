"""
streamlit_app.py

MVP ML-сервиса прогнозирования спроса (строго по кейсу).

Функциональность:
1) Выбор SKU → прогноз на 7-14 дней
2) Сценарий (множитель цены + длительность промо) → прогноз
3) Загрузка CSV (date,sales + опционально price,promo_flag,discount_pct,...) → прогноз

Важно:
- Не показываем «сырой датасет» таблицей по умолчанию.
- Показываем KPI и график "история + прогноз".
- Для NN используем PyTorch LSTM (артефакты в artifacts/SKU_xx/...).
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


# ---------------------------
# UI helpers
# ---------------------------
def inject_css():
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
          #MainMenu {visibility: hidden;}
          footer {visibility: hidden;}

          /* Mobile-only button container */
          .mobile-only { display: none; }
          @media (max-width: 768px) {
            .mobile-only { display: block; }
          }

          /* Slightly nicer button */
          div.stButton>button {
            border-radius: 12px;
            padding: 0.65rem 1rem;
            font-weight: 650;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_params_ui(prefix: str, skus: list[str]):
    """
    Рисует набор параметров. prefix нужен, чтобы ключи не конфликтовали.
    Возвращает dict с параметрами.
    """
    sku = st.selectbox("Товар (SKU)", skus, key=f"{prefix}_sku")
    horizon = st.slider("Горизонт прогноза (дней)", 7, 14, 14, key=f"{prefix}_horizon")
    lookback = st.slider("Окно истории (lookback)", 14, 60, 28, key=f"{prefix}_lookback")

    st.markdown("---")
    show_base = st.toggle("Показывать Baseline", value=True, key=f"{prefix}_show_base")
    show_nn = st.toggle("Показывать LSTM", value=True, key=f"{prefix}_show_nn")

    st.markdown("---")
    st.subheader("Сценарий (A)")
    price_mult_a = st.number_input("Множитель цены (A)", 0.5, 2.0, 1.0, 0.05, key=f"{prefix}_price_mult_a")
    promo_days_a = st.slider("Промо дней (A)", 0, 14, 0, key=f"{prefix}_promo_days_a")
    promo_where_a = st.radio("Где промо (A)", ["В начале", "В конце"], horizontal=True, key=f"{prefix}_promo_where_a")
    promo_where_a_key = "start" if promo_where_a == "В начале" else "end"

    st.markdown("---")
    st.subheader("Сценарий (B) — сравнение")
    enable_b = st.toggle("Включить сценарий B", value=False, key=f"{prefix}_enable_b")
    price_mult_b = st.number_input("Множитель цены (B)", 0.5, 2.0, 1.1, 0.05, disabled=not enable_b, key=f"{prefix}_price_mult_b")
    promo_days_b = st.slider("Промо дней (B)", 0, 14, 7, disabled=not enable_b, key=f"{prefix}_promo_days_b")
    promo_where_b = st.radio("Где промо (B)", ["В начале", "В конце"], horizontal=True, disabled=not enable_b, key=f"{prefix}_promo_where_b")
    promo_where_b_key = "start" if promo_where_b == "В начале" else "end"

    return {
        "sku": sku,
        "horizon": int(horizon),
        "lookback": int(lookback),
        "show_base": bool(show_base),
        "show_nn": bool(show_nn),
        "price_mult_a": float(price_mult_a),
        "promo_days_a": int(promo_days_a),
        "promo_where_a_key": promo_where_a_key,
        "enable_b": bool(enable_b),
        "price_mult_b": float(price_mult_b),
        "promo_days_b": int(promo_days_b),
        "promo_where_b_key": promo_where_b_key,
    }


def sync_mobile_to_sidebar(m: dict):
    """
    Копируем значения мобильной панели в ключи sidebar (sb_*),
    чтобы основной код использовал одни параметры.
    """
    st.session_state["sb_sku"] = m["sku"]
    st.session_state["sb_horizon"] = m["horizon"]
    st.session_state["sb_lookback"] = m["lookback"]
    st.session_state["sb_show_base"] = m["show_base"]
    st.session_state["sb_show_nn"] = m["show_nn"]

    st.session_state["sb_price_mult_a"] = m["price_mult_a"]
    st.session_state["sb_promo_days_a"] = m["promo_days_a"]
    st.session_state["sb_promo_where_a"] = "В начале" if m["promo_where_a_key"] == "start" else "В конце"

    st.session_state["sb_enable_b"] = m["enable_b"]
    st.session_state["sb_price_mult_b"] = m["price_mult_b"]
    st.session_state["sb_promo_days_b"] = m["promo_days_b"]
    st.session_state["sb_promo_where_b"] = "В начале" if m["promo_where_b_key"] == "start" else "В конце"


# ---------------------------
# Data / model
# ---------------------------
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values(["sku", "date"]).reset_index(drop=True)

    # normalize columns
    if "promo" in df.columns and "promo_flag" not in df.columns:
        df = df.rename(columns={"promo": "promo_flag"})

    # fill optional columns
    for col, default in [("price", 10.0), ("promo_flag", 0), ("discount_pct", 0.0), ("is_holiday", 0)]:
        if col not in df.columns:
            df[col] = default

    df["is_weekend"] = (df["date"].dt.weekday >= 5).astype(int)
    return df


def load_nn_for_sku(sku: str):
    sku_dir = ART_DIR / sku
    model_path = sku_dir / "model.pt"
    fs_path = sku_dir / "feature_scaler.joblib"
    ts_path = sku_dir / "target_scaler.joblib"
    metrics_path = sku_dir / "metrics_nn.json"

    if not (model_path.exists() and fs_path.exists() and ts_path.exists()):
        return None

    ckpt = torch.load(model_path, map_location="cpu")
    feature_cols = ckpt.get("feature_cols", FEATURE_COLS)
    horizon_ckpt = int(ckpt.get("horizon", 14))

    model = LSTMForecaster(
        n_features=len(feature_cols),
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        horizon=horizon_ckpt,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    fs = joblib.load(fs_path)
    ts = joblib.load(ts_path)

    nn_metrics = None
    if metrics_path.exists():
        nn_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    return model, fs, ts, list(feature_cols), horizon_ckpt, nn_metrics


def load_baseline_metrics_for_sku(sku: str):
    p = ART_DIR / "metrics_baselines.json"
    if not p.exists():
        return None
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
        row = next((r for r in rows if r.get("sku") == sku and r.get("model") == "moving_avg_7"), None)
        return row
    except Exception:
        return None


# ---------------------------
# Forecast utils
# ---------------------------
def make_future_frame(history: pd.DataFrame, horizon: int, price_mult: float, promo_days: int, promo_where: str) -> pd.DataFrame:
    last = history.sort_values("date").iloc[-1]
    start = last["date"] + pd.Timedelta(days=1)
    dates = pd.date_range(start=start, periods=horizon, freq="D")

    base_price = float(last.get("price", 10.0))
    price = np.full(horizon, base_price * float(price_mult))

    promo_flag = np.zeros(horizon, dtype=int)
    discount = np.zeros(horizon, dtype=float)

    promo_days = int(max(0, min(horizon, promo_days)))
    if promo_days > 0:
        sl = slice(0, promo_days) if promo_where == "start" else slice(horizon - promo_days, horizon)
        promo_flag[sl] = 1
        discount[sl] = 0.20

    fut = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "sku": last.get("sku", "SKU"),
            "sales": 0.0,  # future sales unknown
            "price": np.round(price, 2),
            "promo_flag": promo_flag,
            "discount_pct": np.round(discount, 3),
            "is_weekend": (pd.Series(dates).dt.weekday >= 5).astype(int).values,
            "is_holiday": np.zeros(horizon, dtype=int),
        }
    )
    return fut


@torch.no_grad()
def lstm_forecast(model, fs, ts, feature_cols, hist: pd.DataFrame, fut: pd.DataFrame, lookback: int, horizon: int) -> np.ndarray:
    h = add_calendar_feats(hist.sort_values("date").tail(int(lookback)).copy())
    f = add_calendar_feats(fut.copy())

    for col in feature_cols:
        if col not in h.columns:
            h[col] = 0.0
        if col not in f.columns:
            f[col] = 0.0

    past_X = h[feature_cols].to_numpy(dtype=float)
    fut_X = f[feature_cols].to_numpy(dtype=float)

    fut_X[:, 0] = 0.0  # no leakage: future sales unknown

    X = np.vstack([past_X, fut_X])
    Xs = fs.transform(X)

    x_tensor = torch.tensor(Xs, dtype=torch.float32).unsqueeze(0)
    pred_scaled = model(x_tensor).squeeze(0).cpu().numpy()
    pred = ts.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    return np.maximum(pred[:horizon], 0.0)


def moving_average_future(history_sales: np.ndarray, horizon: int, window: int = 7) -> np.ndarray:
    hist = list(history_sales.astype(float))
    preds = []
    for _ in range(horizon):
        w = hist[-window:] if len(hist) >= window else hist
        p = float(np.mean(w))
        preds.append(p)
        hist.append(p)
    return np.array(preds)


def plot_history_forecast(dates_hist, sales_hist, dates_fut, pred_nn, pred_base, show_base=True, show_nn=True):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates_hist, sales_hist, label="История продаж", linewidth=2)
    if show_base and pred_base is not None:
        ax.plot(dates_fut, pred_base, linestyle="--", marker="o", label="Baseline MA(7)")
    if show_nn and pred_nn is not None:
        ax.plot(dates_fut, pred_nn, linestyle="--", marker="o", label="LSTM прогноз")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Продажи")
    ax.grid(True)
    ax.legend()
    fig.autofmt_xdate()
    return fig


def main():
    st.set_page_config(page_title="Прогноз спроса", layout="wide")
    inject_css()

    st.title("Нейросетевая модель для прогнозирования спроса на товары")
    st.write(
        "Эта модель — **LSTM нейросеть**, обученная на временных рядах продаж и внешних факторах "
        "(цена/промо/календарь). Она прогнозирует спрос на **7–14 дней** вперёд и позволяет "
        "проверять сценарии (например, скидка или изменение цены)."
    )

    # Mobile-only button (top-right) to open params panel
    top_l, top_r = st.columns([6, 2])
    with top_r:
        st.markdown('<div class="mobile-only">', unsafe_allow_html=True)
        if st.button("⚙️ Параметры", use_container_width=True, key="open_mobile"):
            st.session_state["mobile_panel_open"] = True
        st.markdown("</div>", unsafe_allow_html=True)

    if not DATA_PATH.exists():
        st.error("Нет данных data/sales.csv. Сначала сгенерируй их или дождись GitHub Actions.")
        return

    df = load_data(DATA_PATH)
    skus = sorted(df["sku"].unique())

    # Sidebar (как основной контроль)
    with st.sidebar:
        st.header("⚙️ Панель параметров")
        sb_vals = render_params_ui("sb", skus)
        run = st.button("🚀 Рассчитать прогноз", use_container_width=True, key="sb_run")

    # Mobile panel overlay (only on phone, opened by button)
    if st.session_state.get("mobile_panel_open", False):
        with st.expander("⚙️ Параметры (мобильная панель)", expanded=True):
            mob_vals = render_params_ui("mob", skus)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Применить", use_container_width=True, key="mob_apply"):
                    sync_mobile_to_sidebar(mob_vals)
                    st.session_state["mobile_panel_open"] = False
                    st.rerun()
            with c2:
                if st.button("✖ Закрыть", use_container_width=True, key="mob_close"):
                    st.session_state["mobile_panel_open"] = False
                    st.rerun()

    # Используем значения из sidebar (sb_vals) — так “ничего больше не меняем”
    sku = sb_vals["sku"]
    horizon = sb_vals["horizon"]
    lookback = sb_vals["lookback"]
    show_base = sb_vals["show_base"]
    show_nn = sb_vals["show_nn"]

    price_mult_a = sb_vals["price_mult_a"]
    promo_days_a = sb_vals["promo_days_a"]
    promo_where_a_key = sb_vals["promo_where_a_key"]

    enable_b = sb_vals["enable_b"]
    price_mult_b = sb_vals["price_mult_b"]
    promo_days_b = sb_vals["promo_days_b"]
    promo_where_b_key = sb_vals["promo_where_b_key"]

    if not run:
        st.info("Выбери параметры и нажми «Рассчитать прогноз».")
        return

    hist = df[df["sku"] == sku].sort_values("date").reset_index(drop=True)
    fut = make_future_frame(hist, horizon, price_mult_a, promo_days_a, promo_where_a_key)

    base_pred = moving_average_future(hist["sales"].values, horizon=horizon, window=7)

    nn_pack = load_nn_for_sku(sku)
    nn_pred = None
    nn_metrics = None

    if nn_pack is not None:
        model, fs, ts, fcols, h_ckpt, nn_metrics = nn_pack
        horizon_used = min(horizon, h_ckpt)
        nn_pred = lstm_forecast(model, fs, ts, fcols, hist, fut, lookback=lookback, horizon=horizon_used)
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
            show_base=show_base,
            show_nn=show_nn,
        )
        st.pyplot(fig)

    with c2:
        st.subheader("Метрики")
        bm = load_baseline_metrics_for_sku(sku)
        if bm:
            st.write("**Baseline MA(7)**")
            st.json(bm)

        if nn_metrics is not None:
            st.write("**LSTM**")
            st.json(nn_metrics)
        else:
            st.warning("Артефакты LSTM не найдены для этого SKU. Дождись Actions или обучи модель.")

        st.subheader("Сценарий")
        st.write(f"SKU: **{sku}**")
        st.write(f"Горизонт: **{horizon}** дней")
        st.write(f"Lookback: **{lookback}** дней")
        st.write(f"Цена x: **{price_mult_a:.2f}**")
        st.write(f"Промо дней: **{promo_days_a}** ({'в начале' if promo_where_a_key=='start' else 'в конце'})")

    # Scenario B compare (если включено)
    if enable_b and nn_pack is not None and nn_pred is not None:
        st.divider()
        st.subheader("Сравнение сценариев A vs B (LSTM)")

        model, fs, ts, fcols, h_ckpt, _ = nn_pack
        fut_b = make_future_frame(hist, horizon, price_mult_b, promo_days_b, promo_where_b_key)
        horizon_used = min(horizon, h_ckpt)
        nn_b = lstm_forecast(model, fs, ts, fcols, hist, fut_b, lookback=lookback, horizon=horizon_used)
        if horizon_used < horizon:
            nn_b = np.pad(nn_b, (0, horizon - horizon_used), constant_values=float(nn_b[-1]))

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(hist["date"].tail(180), hist["sales"].tail(180), label="История", linewidth=2)
        ax2.plot(fut["date"], nn_pred, "--", marker="o", label="LSTM (A)")
        ax2.plot(fut_b["date"], nn_b, "--", marker="o", label="LSTM (B)")
        ax2.grid(True)
        ax2.legend()
        fig2.autofmt_xdate()
        st.pyplot(fig2)

    # Download
    st.divider()
    out = pd.DataFrame({"date": fut["date"], "baseline_ma7": base_pred})
    if nn_pred is not None:
        out["lstm_forecast"] = nn_pred
    st.download_button(
        "Скачать CSV прогноза",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_{sku}.csv",
        mime="text/csv",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
