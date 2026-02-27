# data/generate_data.py
"""
Генерация синтетических данных продаж для задачи прогнозирования спроса.

Выход: CSV с колонками:
date, sku, price, promo_flag, discount_pct, is_weekend, is_holiday, sales

Особенности синтетики:
- недельная сезонность + годовая сезонность
- тренд
- влияние промо/скидки на спрос
- weekend/holiday эффект
- шум
- редкие "spikes" (всплески спроса)
- редкие "stockout" (почти нулевые продажи из-за отсутствия товара)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
import math
import numpy as np
import pandas as pd


@dataclass
class SkuParams:
    base_demand: float
    trend_per_day: float
    weekly_amp: float
    yearly_amp: float
    noise_std: float
    price_base: float
    price_vol: float
    promo_lift: float          # мультипликатор спроса при promo
    discount_elasticity: float # насколько скидка увеличивает спрос


def _simple_holidays(year: int) -> set[date]:
    """
    Упрощённый набор праздников (без привязки к стране).
    Нужен, чтобы в данных был эффект "праздников".
    """
    return {
        date(year, 1, 1),   # New Year
        date(year, 1, 7),   # example holiday
        date(year, 3, 8),
        date(year, 5, 1),
        date(year, 5, 9),
        date(year, 12, 25),
    }


def generate(
    output: str,
    days: int = 1100,
    skus: int = 5,
    seed: int = 42,
    start: date | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    if start is None:
        # последние N дней до "сегодня"
        start = date.today() - timedelta(days=days)

    dates = [start + timedelta(days=i) for i in range(days)]

    # зададим параметры на SKU (слегка разные)
    params: list[SkuParams] = []
    for i in range(skus):
        p = SkuParams(
            base_demand=float(rng.integers(20, 120)),
            trend_per_day=float(rng.normal(0.02, 0.02)),     # небольшой тренд
            weekly_amp=float(rng.uniform(0.05, 0.25)),
            yearly_amp=float(rng.uniform(0.05, 0.35)),
            noise_std=float(rng.uniform(2.0, 12.0)),
            price_base=float(rng.uniform(5.0, 25.0)),
            price_vol=float(rng.uniform(0.02, 0.08)),        # дневная волатильность цены
            promo_lift=float(rng.uniform(1.10, 1.60)),
            discount_elasticity=float(rng.uniform(0.8, 2.0)),
        )
        params.append(p)

    records = []
    for sku_idx in range(skus):
        sku = f"SKU_{sku_idx+1:02d}"
        p = params[sku_idx]

        # индивидуальная "любимая" фаза сезонности
        weekly_phase = rng.uniform(0, 2 * math.pi)
        yearly_phase = rng.uniform(0, 2 * math.pi)

        price = p.price_base

        for t, d in enumerate(dates):
            dow = d.weekday()  # 0..6
            is_weekend = int(dow >= 5)

            holidays = _simple_holidays(d.year)
            is_holiday = int(d in holidays)

            # цена слегка гуляет (как random walk)
            price_shock = rng.normal(0.0, p.price_vol)
            price = max(0.5, price * (1.0 + price_shock))

            # промо события: редкие блоки по 3-10 дней
            # вероятность старта промо
            promo_start_prob = 0.012
            if rng.random() < promo_start_prob:
                promo_len = int(rng.integers(3, 11))
            else:
                promo_len = 0

            # применяем промо на текущий день, если оно "активно"
            # для простоты — промо считается активным только в день старта,
            # а эффект скидки задаём через discount_pct как функцию дня.
            promo_flag = int(promo_len > 0)

            # скидка: если промо, то скидка 5-35%
            discount_pct = float(rng.uniform(0.05, 0.35)) if promo_flag else 0.0

            # недельная сезонность (синус + "пятница/сб" часто выше)
            weekly = 1.0 + p.weekly_amp * math.sin(2 * math.pi * (dow / 7.0) + weekly_phase)
            if dow in (4, 5):  # пятница/суббота
                weekly *= 1.05

            # годовая сезонность
            day_of_year = (d - date(d.year, 1, 1)).days
            yearly = 1.0 + p.yearly_amp * math.sin(2 * math.pi * (day_of_year / 365.25) + yearly_phase)

            # тренд
            trend = 1.0 + p.trend_per_day * t

            # эффект выходных и праздников
            cal_mult = 1.0
            if is_weekend:
                cal_mult *= 1.08
            if is_holiday:
                cal_mult *= 1.20

            # эффект промо и скидки
            promo_mult = 1.0
            if promo_flag:
                promo_mult *= p.promo_lift
                promo_mult *= (1.0 + p.discount_elasticity * discount_pct)

            # эффект цены (чем выше цена — тем ниже спрос), простая эластичность
            # нормируем на базовую цену
            price_mult = (p.price_base / price) ** 0.35

            # базовый спрос
            demand_mean = p.base_demand * weekly * yearly * trend * cal_mult * promo_mult * price_mult

            # редкие всплески спроса (PR-кампании)
            if rng.random() < 0.01:
                demand_mean *= float(rng.uniform(1.3, 2.2))

            # редкие stockout (нет товара на складе)
            if rng.random() < 0.006:
                demand_mean *= float(rng.uniform(0.0, 0.15))

            # шум
            noise = rng.normal(0.0, p.noise_std)
            sales = max(0.0, demand_mean + noise)

            records.append(
                {
                    "date": d.isoformat(),
                    "sku": sku,
                    "price": round(float(price), 2),
                    "promo_flag": promo_flag,
                    "discount_pct": round(float(discount_pct), 3),
                    "is_weekend": is_weekend,
                    "is_holiday": is_holiday,
                    "sales": int(round(sales)),
                }
            )

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sku", "date"]).reset_index(drop=True)
    df.to_csv(output, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/sales.csv")
    parser.add_argument("--days", type=int, default=1100)
    parser.add_argument("--skus", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate(
        output=args.output,
        days=args.days,
        skus=args.skus,
        seed=args.seed,
    )
    print(f"Saved: {args.output}")
    print(df.head())


if __name__ == "__main__":
    main()