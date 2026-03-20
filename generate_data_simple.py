import argparse
import numpy as np
import pandas as pd
from datetime import date, timedelta
import math

def generate_sales(output='sales.csv', days=1100, seed=42, base_demand=100, trend=0.01,
                   weekly_amp=20, yearly_amp=30, noise_std=10):
    """
    Генерирует простой временной ряд продаж с трендом, сезонностью и шумом.
    """
    rng = np.random.default_rng(seed)
    start = date.today() - timedelta(days=days)
    dates = [start + timedelta(days=i) for i in range(days)]

    weekly_phase = rng.uniform(0, 2 * math.pi)
    yearly_phase = rng.uniform(0, 2 * math.pi)

    sales = []
    for t, d in enumerate(dates):
        dow = d.weekday()
        # недельная сезонность (относительное отклонение)
        weekly = 1.0 + weekly_amp * math.sin(2 * math.pi * (dow / 7.0) + weekly_phase) / base_demand
        # годовая сезонность
        day_of_year = (d - date(d.year, 1, 1)).days
        yearly = 1.0 + yearly_amp * math.sin(2 * math.pi * (day_of_year / 365.25) + yearly_phase) / base_demand
        # тренд
        trend_factor = 1.0 + trend * t
        # базовый спрос
        demand = base_demand * trend_factor * weekly * yearly
        # шум
        noise = rng.normal(0, noise_std)
        s = max(0, demand + noise)
        sales.append(round(s))

    df = pd.DataFrame({'date': dates, 'sales': sales})
    df['date'] = pd.to_datetime(df['date'])
    df.to_csv(output, index=False)
    print(f"Готово: {days} дней продаж сохранено в {output}")
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='sales.csv')
    parser.add_argument('--days', type=int, default=1100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base', type=float, default=100)
    parser.add_argument('--trend', type=float, default=0.01)
    parser.add_argument('--weekly_amp', type=float, default=20)
    parser.add_argument('--yearly_amp', type=float, default=30)
    parser.add_argument('--noise', type=float, default=10)
    args = parser.parse_args()
    generate_sales(args.output, args.days, args.seed, args.base,
                   args.trend, args.weekly_amp, args.yearly_amp, args.noise)