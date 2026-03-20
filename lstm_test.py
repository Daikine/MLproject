import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==================== ЗАГРУЗКА ====================
def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

# ==================== BASELINE ====================
def baseline_ma(history, horizon, window=7):
    """Простое скользящее среднее (итеративный прогноз)"""
    hist = list(history)
    preds = []
    for _ in range(horizon):
        w = hist[-window:] if len(hist) >= window else hist
        p = float(np.mean(w))
        preds.append(p)
        hist.append(p)          # подставляем предсказанное для следующего шага
    return np.array(preds)

# ==================== ПОДГОТОВКА ПОСЛЕДОВАТЕЛЬНОСТЕЙ ====================
def create_sequences(data, seq_length, horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length : i + seq_length + horizon])
    return np.array(X), np.array(y)

# ==================== МОДЕЛЬ LSTM ====================
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, horizon=14):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.linear = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        out = out[:, -1, :]             # последний шаг
        out = self.linear(out)          # (batch, horizon)
        return out

# ==================== ОБУЧЕНИЕ ====================
def train_lstm(X_train, y_train, X_val, y_val, horizon,
               hidden_size=64, num_layers=2, epochs=50, batch_size=32, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используется устройство: {device}')

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = LSTMPredictor(
        input_size=X_train.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        horizon=horizon
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience = 10
    counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Ранняя остановка на эпохе {epoch+1}')
                break

    model.load_state_dict(best_state)
    return model, device

# ==================== ОСНОВНОЙ ЭКСПЕРИМЕНТ ====================
def main():
    # Параметры
    data_file = 'sales.csv'
    lookback = 28
    horizon = 14
    test_size = 120
    val_size = 60

    # Загрузка данных
    df = load_data(data_file)
    sales = df['sales'].values.astype(float)
    dates = df['date'].values

    # Масштабирование (Z-score)
    scaler = StandardScaler()
    sales_scaled = scaler.fit_transform(sales.reshape(-1, 1)).flatten()

    # Создание последовательностей (lookback -> horizon)
    X, y = create_sequences(sales_scaled, lookback, horizon)

    # Разделение по времени
    total = len(X)
    train_end = total - test_size - val_size
    val_end = total - test_size

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f'Обучающих примеров: {len(X_train)}, Валидация: {len(X_val)}, Тест: {len(X_test)}')

    # Обучение LSTM (добавляем размерность признака = 1)
    model, device = train_lstm(
        X_train.reshape(X_train.shape[0], X_train.shape[1], 1),
        y_train,
        X_val.reshape(X_val.shape[0], X_val.shape[1], 1),
        y_val,
        horizon=horizon,
        hidden_size=64,
        num_layers=2,
        epochs=50,
        batch_size=32,
        lr=0.001
    )

    # Предсказание на тесте
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(
            X_test.reshape(X_test.shape[0], X_test.shape[1], 1),
            dtype=torch.float32
        ).to(device)
        y_pred_scaled = model(X_test_t).cpu().numpy()

    # Обратное масштабирование
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    # Baseline на тех же тестовых окнах
    base_preds = []
    for i in range(len(X_test)):
        # Индекс начала тестового окна в исходном ряду
        start_idx = train_end + val_size + i
        # Берём историю до этого момента (фактические продажи, не масштабированные)
        hist_sales = sales[:start_idx + lookback]   # до последнего дня окна
        if len(hist_sales) > 200:
            hist_part = hist_sales[-200:]
        else:
            hist_part = hist_sales
        pred = baseline_ma(hist_part, horizon, window=7)
        base_preds.append(pred)
    base_preds = np.array(base_preds)

    # Метрики (по всем предсказанным точкам)
    all_base = base_preds.reshape(-1)
    all_lstm = y_pred.reshape(-1)
    all_fact = y_test_actual.reshape(-1)

    mae_base = mean_absolute_error(all_fact, all_base)
    mae_lstm = mean_absolute_error(all_fact, all_lstm)
    rmse_base = np.sqrt(mean_squared_error(all_fact, all_base))
    rmse_lstm = np.sqrt(mean_squared_error(all_fact, all_lstm))

    print("\n=== МЕТРИКИ НА ТЕСТЕ ===")
    print(f"Baseline MA(7): MAE = {mae_base:.2f}, RMSE = {rmse_base:.2f}")
    print(f"LSTM:          MAE = {mae_lstm:.2f}, RMSE = {rmse_lstm:.2f}")

    # Графики нескольких тестовых окон
    n_show = 5
    plt.figure(figsize=(15, 10))
    for i in range(min(n_show, len(X_test))):
        plt.subplot(n_show, 1, i+1)

        start_idx = train_end + val_size + i
        hist_vals = sales[start_idx : start_idx + lookback]
        hist_dates = dates[start_idx : start_idx + lookback]

        fact_vals = y_test_actual[i]
        fact_dates = dates[start_idx + lookback : start_idx + lookback + horizon]

        base_vals = base_preds[i]
        lstm_vals = y_pred[i]

        plt.plot(hist_dates, hist_vals, 'b-', label='История')
        plt.plot(fact_dates, fact_vals, 'g-', marker='o', label='Факт')
        plt.plot(fact_dates, base_vals, 'r--', marker='s', label='Baseline MA(7)')
        plt.plot(fact_dates, lstm_vals, 'm--', marker='^', label='LSTM')
        plt.legend()
        plt.grid(True)
        plt.title(f'Тестовое окно {i+1}')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()