"""Generate figures for presentation (Milestone 5)."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sales.csv")
    ap.add_argument("--sku", default=None)
    args = ap.parse_args()

    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # Compare MAE baseline (moving_avg_7) vs NN
    if Path("artifacts/metrics_baselines.json").exists() and Path("artifacts/metrics_nn_all.json").exists():
        with open("artifacts/metrics_baselines.json","r",encoding="utf-8") as f:
            bas = json.load(f)
        with open("artifacts/metrics_nn_all.json","r",encoding="utf-8") as f:
            nn = json.load(f)

        # map sku -> baseline MAE
        b_mae = {}
        for r in bas:
            if r["model"] == "moving_avg_7":
                b_mae[r["sku"]] = r["mae"]

        skus = [r["sku"] for r in nn]
        nn_mae = [r["mae"] for r in nn]
        base_mae = [b_mae.get(s, np.nan) for s in skus]

        x = np.arange(len(skus))
        plt.figure()
        plt.bar(x - 0.2, base_mae, width=0.4, label="Baseline MA(7)")
        plt.bar(x + 0.2, nn_mae, width=0.4, label="LSTM")
        plt.xticks(x, skus, rotation=30, ha="right")
        plt.ylabel("MAE")
        plt.title("Сравнение MAE: Baseline vs LSTM")
        plt.legend()
        plt.tight_layout()
        plt.savefig("reports/figures/compare_mae.png", dpi=200)
        plt.close()

    # Example series
    df = pd.read_csv(args.data, parse_dates=["date"]).sort_values(["sku","date"])
    sku = args.sku or df["sku"].iloc[0]
    g = df[df["sku"] == sku].tail(180)
    plt.figure()
    plt.plot(g["date"], g["sales"])
    plt.title(f"Продажи (последние 180 дней): {sku}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("reports/figures/example_series.png", dpi=200)
    plt.close()

    print("Saved reports/figures/*.png")

if __name__ == "__main__":
    main()
