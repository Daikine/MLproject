# streamlit_app.py
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
# UI polish (light, clean)
# ---------------------------
def inject_css():
    st.markdown(
        """
        <style>
          /* Page width */
          .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

          /* Hide Streamlit chrome */
          #MainMenu {visibility: hidden;}
          footer {visibility: hidden;}
          header {visibility: hidden;}

          /* Cards */
          .card {
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-radius: 16px;
            padding: 14px 14px;
            background: white;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
          }
          .card h4 { margin: 0 0 0.25rem 0; font-size: 0.95rem; color: rgba(49, 51, 63, 0.85); }
          .card .big { font-size: 1.45rem; font-weight: 700; margin: 0; }
          .muted { color: rgba(49, 51, 63, 0.65); font-size: 0.9rem; }

          /* Buttons */
          div.stButton>button {
            border-radius: 12px;
            padding: 0.6rem 1rem;
            font-weight: 600;
          }

          /* Tabs spacing */
          button[data-baseweb="tab"] { font-size: 0.95rem; }

          /* Nice separators */
          hr { margin: 1rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------
# Data / model loading
# ---------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values(["sku", "date"]).reset_index(drop=True)
    # normalize column naming
    if "promo" in df.columns and "promo_flag" not in df.columns:
        df = df.rename(columns={"promo": "promo_flag"})
    for col, default in [("price", 10.0), ("promo_flag", 0), ("discount_pct", 0.0), ("is_holiday", 0)]:
        if col not in df.columns:
            df[col] = default
    df["is_weekend"] = (df["date"].dt.weekday >= 5).astype(int)
    return df


@st.cache_resource
def load_nn_for_sku_cached(sku: str):
    return load_nn_for_sku(sku)


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
# Forecast logic
# ---------------------------
def make_future_frame(
    history: pd.DataFrame,
    horizon: int,
    price_mult: float,
    promo_days: int,
    promo_where: str,  # "start"|"end"
    discount: float = 0.20,
) -> pd.DataFrame:
    last = history.sort_values("date").iloc[-1]
    start = last["date"] + pd.Timedelta(days=1)
    dates = pd.date_range(start=start, periods=horizon, freq="D")

    base_price = float(last.get("price", 10.0))
    price = np.full(horizon, base_price * float(price_mult))

    promo_flag = np.zeros(horizon, dtype=int)
    discount_pct = np.zeros(horizon, dtype=float)

    promo_days = int(max(0, min(horizon, promo_days)))
    if promo_days > 0:
        sl = slice(0, promo_days) if promo_where == "start" else slice(horizon - promo_days, horizon)
        promo_flag[sl] = 1
        discount_pct[sl] = float(discount)

    fut = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "sku": last.get("sku", "SKU"),
            "sales": 0.0,  # unknown future sales
            "price": np.round(price, 2),
            "promo_flag": promo_flag,
            "discount_pct": np.round(discount_pct, 3),
            "is_weekend": (pd.Series(dates).dt.weekday >= 5).astype(int).values,
            "is_holiday": np.zeros(horizon, dtype=int),
        }
    )
    return fut


@torch.no_grad()
def lstm_forecast(
    model,
    fs,
    ts,
    feature_cols: list[str],
    hist: pd.DataFrame,
    fut: pd.DataFrame,
    lookback: int,
    horizon: int,
) -> np.ndarray:
    # past
    h = add_calendar_feats(hist.sort_values("date").tail(int(lookback)).copy())
    # future
    f = add_calendar_feats(fut.copy())

    # ensure all columns exist
    for col in feature_cols:
        if col not in h.columns:
            h[col] = 0.0
        if col not in f.columns:
            f[col] = 0.0

    past_X = h[feature_cols].to_numpy(dtype=float)
    fut_X = f[feature_cols].to_numpy(dtype=float)

    # no leakage: future sales must be 0
    fut_X[:, 0] = 0.0

    X = np.vstack([past_X, fut_X])  # (L+H, F)
    Xs = fs.transform(X)

    x_tensor = torch.tensor(Xs, dtype=torch.float32).unsqueeze(0)
    pred_scaled = model(x_tensor).squeeze(0).cpu().numpy()
    pred = ts.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    return np.maximum(pred[:horizon], 0.0)


def baseline_ma(history_sales: np.ndarray, horizon: int, window: int = 7) -> np.ndarray:
    hist = list(history_sales.astype(float))
    preds = []
    for _ in range(horizon):
        w = hist[-window:] if len(hist) >= window else hist
        p = float(np.mean(w))
        preds.append(p)
        hist.append(p)
    return np.array(preds)


def compute_kpis(pred: np.ndarray):
    total = float(np.sum(pred))
    avg = float(np.mean(pred))
    peak = float(np.max(pred))
    peak_day = int(np.argmax(pred) + 1)
    return total, avg, peak, peak_day


def plot_forecast(d_hist, y_hist, d_fut, y_base, y_nn, band=None, show_base=True, show_nn=True):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(d_hist, y_hist, label="История", linewidth=2)

    if show_base and y_base is not None:
        ax.plot(d_fut, y_base, "--", marker="o", label="Baseline MA(7)")

    if show_nn and y_nn is not None:
        ax.plot(d_fut, y_nn, "--", marker="o", label="LSTM прогноз")

    if band is not None and y_nn is not None:
        lo, hi = band
        ax.fill_between(d_fut, lo, hi, alpha=0.15, label="Интервал (оценка)")

    ax.set_xlabel("Дата")
    ax.set_ylabel("Продажи")
    ax.grid(True)
    ax.legend()
    fig.autofmt_xdate()
    return fig


# ---------------------------
# App
# ---------------------------
def card(title: str, big: str, sub: str = ""):
    st.markdown(
        f"""
        <div class="card">
          <h4>{title}</h4>
          <div class="big">{big}</div>
          <div class="muted">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Demand Forecasting", layout="wide")
    inject_css()

    st.markdown("## 🔮 Прогноз спроса на товары (MVP)")
    st.markdown(
        "<div class='muted'>Сервис прогнозирует спрос на 7–14 дней вперёд. "
        "Сценарий цены/промо влияет на прогноз (LSTM обучена с будущими экзогенными признаками).</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    if not DATA_PATH.exists():
        st.error("Нет `data/sales.csv`. Сначала сгенерируй данные или дождись GitHub Actions.")
        st.stop()

    df = load_data()
    skus = sorted(df["sku"].unique())

    # Sidebar controls
    # --- Responsive controls (works on phone + desktop) ---
if "run_clicked" not in st.session_state:
    st.session_state.run_clicked = False

# Кнопка всегда видна (и на телефоне тоже)
top_left, top_right = st.columns([1, 1])
with top_left:
    st.markdown("### ⚙️ Управление")
with top_right:
    st.session_state.run_clicked = st.button("🚀 Рассчитать прогноз", use_container_width=True)

# Параметры прячем в expander — он отлично работает на телефоне
with st.expander("⚙️ Параметры (нажми, чтобы открыть)", expanded=True):
    sku = st.selectbox("SKU", skus, key="sku")
    horizon = st.slider("Горизонт (дней)", 7, 14, 14, key="horizon")
    lookback = st.slider("Окно истории (lookback)", 14, 60, 28, key="lookback")

    st.markdown("---")
    st.subheader("Отображение")
    show_base = st.toggle("Показывать Baseline", value=True, key="show_base")
    show_nn = st.toggle("Показывать LSTM", value=True, key="show_nn")

    st.markdown("---")
    st.subheader("Сценарий (A)")
    price_mult_a = st.number_input("Цена x (A)", 0.5, 2.0, 1.0, 0.05, key="price_mult_a")
    promo_days_a = st.slider("Промо дней (A)", 0, 14, 0, key="promo_days_a")
    promo_where_a = st.radio("Промо где (A)", ["В начале", "В конце"], horizontal=True, key="promo_where_a")
    promo_where_a_key = "start" if promo_where_a == "В начале" else "end"

    st.markdown("---")
    st.subheader("Сценарий (B) — сравнение")
    enable_b = st.toggle("Включить сценарий B", value=False, key="enable_b")
    price_mult_b = st.number_input("Цена x (B)", 0.5, 2.0, 1.1, 0.05, disabled=not enable_b, key="price_mult_b")
    promo_days_b = st.slider("Промо дней (B)", 0, 14, 7, disabled=not enable_b, key="promo_days_b")
    promo_where_b = st.radio("Промо где (B)", ["В начале", "В конце"], horizontal=True, disabled=not enable_b, key="promo_where_b")
    promo_where_b_key = "start" if promo_where_b == "В начале" else "end"

# Run button state
run = bool(st.session_state.run_clicked)

    # History
    hist = df[df["sku"] == sku].sort_values("date").reset_index(drop=True)
    d_hist = hist["date"].tail(180)
    y_hist = hist["sales"].tail(180)

    # Baseline
    base = baseline_ma(hist["sales"].values, horizon=horizon, window=7)

    # Load NN
    pack = load_nn_for_sku_cached(sku)
    nn = None
    nn_metrics = None
    band = None

    fut_a = make_future_frame(hist, horizon, price_mult_a, promo_days_a, promo_where_a_key)
    d_fut = fut_a["date"]

    if pack is not None:
        model, fs, ts, fcols, h_ckpt, nn_metrics = pack
        horizon_used = min(horizon, h_ckpt)

        nn_a = lstm_forecast(model, fs, ts, fcols, hist, fut_a, lookback, horizon_used)
        if horizon_used < horizon:
            nn_a = np.pad(nn_a, (0, horizon - horizon_used), constant_values=float(nn_a[-1]))
        nn = nn_a

        # “интервал” — очень грубая оценка: используем RMSE модели из metrics_nn.json, если есть
        if nn_metrics and "rmse" in nn_metrics:
            sigma = float(nn_metrics["rmse"])
            lo = np.maximum(nn - 1.0 * sigma, 0.0)
            hi = np.maximum(nn + 1.0 * sigma, 0.0)
            band = (lo, hi)

    # KPI blocks
    col1, col2, col3, col4 = st.columns(4)
    total_b, avg_b, peak_b, peak_day_b = compute_kpis(base)
    with col1:
        card("Baseline: суммарный спрос", f"{total_b:.0f}", f"Среднее/день: {avg_b:.1f}")
    with col2:
        card("Baseline: пик", f"{peak_b:.0f}", f"День пика: {peak_day_b}")

    if nn is not None:
        total_n, avg_n, peak_n, peak_day_n = compute_kpis(nn)
        with col3:
            card("LSTM: суммарный спрос", f"{total_n:.0f}", f"Среднее/день: {avg_n:.1f}")
        with col4:
            card("LSTM: пик", f"{peak_n:.0f}", f"День пика: {peak_day_n}")
    else:
        with col3:
            card("LSTM", "нет модели", "Нет artifacts для этого SKU")
        with col4:
            card("Подсказка", "Запусти Actions", "Или обучи локально и закоммить artifacts/")

    st.divider()

    # Plot + scenario summary
    left, right = st.columns([2, 1])

    with left:
        st.subheader("📈 История + прогноз")
        fig = plot_forecast(d_hist, y_hist, d_fut, base, nn, band=band, show_base=show_base, show_nn=show_nn)
        st.pyplot(fig)

    with right:
        st.subheader("🧾 Параметры")
        st.write(f"**SKU:** {sku}")
        st.write(f"**Horizon:** {horizon} дней")
        st.write(f"**Lookback:** {lookback} дней")
        st.write("**Сценарий A:**")
        st.write(f"- Цена x: **{price_mult_a:.2f}**")
        st.write(f"- Промо дней: **{promo_days_a}** ({'в начале' if promo_where_a_key=='start' else 'в конце'})")

        # Baseline & NN metrics
        st.divider()
        st.subheader("📊 Метрики (на тесте)")
        bm = load_baseline_metrics_for_sku(sku)
        if bm:
            st.caption("Baseline MA(7)")
            st.json(bm)

        if nn_metrics:
            st.caption("LSTM")
            st.json(nn_metrics)

        # Anomaly / sanity checks
        if nn is not None:
            if np.any(np.isnan(nn)) or np.any(np.isinf(nn)):
                st.warning("В прогнозе есть NaN/Inf — проверь обучение/артефакты.")
            if float(np.max(nn)) > float(np.max(hist["sales"].tail(180))) * 3.0:
                st.warning("Прогноз слишком высок по сравнению с историей — возможно сценарий/данные дают всплеск.")

    # Scenario B comparison
    if enable_b and pack is not None:
        st.divider()
        st.subheader("🆚 Сравнение сценариев A vs B (LSTM)")

        model, fs, ts, fcols, h_ckpt, _ = pack
        fut_b = make_future_frame(hist, horizon, price_mult_b, promo_days_b, promo_where_b_key)
        horizon_used = min(horizon, h_ckpt)
        nn_b = lstm_forecast(model, fs, ts, fcols, hist, fut_b, lookback, horizon_used)
        if horizon_used < horizon:
            nn_b = np.pad(nn_b, (0, horizon - horizon_used), constant_values=float(nn_b[-1]))

        cA, cB, cD = st.columns(3)
        tA, aA, pA, _ = compute_kpis(nn)
        tB, aB, pB, _ = compute_kpis(nn_b)
        with cA:
            card("Сценарий A — сумма", f"{tA:.0f}", f"среднее/день: {aA:.1f}")
        with cB:
            card("Сценарий B — сумма", f"{tB:.0f}", f"среднее/день: {aB:.1f}")
        with cD:
            delta = tB - tA
            pct = (delta / max(tA, 1e-6)) * 100.0
            card("Разница B − A", f"{delta:+.0f}", f"{pct:+.1f}% к сумме горизонта")

        # Plot two scenario lines
        fig2, ax2 = plt.subplots(figsize=(11, 4))
        ax2.plot(d_hist, y_hist, label="История", linewidth=2)
        ax2.plot(fut_a["date"], nn, "--", marker="o", label="LSTM (A)")
        ax2.plot(fut_b["date"], nn_b, "--", marker="o", label="LSTM (B)")
        ax2.grid(True)
        ax2.legend()
        ax2.set_xlabel("Дата")
        ax2.set_ylabel("Продажи")
        fig2.autofmt_xdate()
        st.pyplot(fig2)

    # Download forecast CSV
    st.divider()
    st.subheader("⬇️ Скачать прогноз")
    out = pd.DataFrame({"date": d_fut})
    out["baseline_ma7"] = base
    if nn is not None:
        out["lstm_forecast"] = nn

    st.download_button(
        "Скачать CSV прогноза",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_{sku}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("ℹ️ Как это работает (коротко)", expanded=False):
        st.write(
            "- Baseline: скользящее среднее MA(7).\n"
            "- LSTM: нейросеть обучена на входе **прошлое + будущие экзогенные фичи** (price/promo/calendar).\n"
            "- Поэтому сценарий (цена/промо) меняет вход → меняется прогноз.\n"
            "- Метрики считаются на тестовом периоде (time split по датам)."
        )


if __name__ == "__main__":
    main()
