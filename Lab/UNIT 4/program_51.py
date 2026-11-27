# Program 51: Stock forecasting RNN
# Question: RNN-based time-series forecasting for stock prices, evaluate prediction accuracy.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 2. Generate Synthetic Daily Stock Price Time-Series (1200 days)
t = np.linspace(0, 60, 1200)
# Trend + multiple cyclic frequencies + Gaussian stochastic fluctuations
trend = 100 + 0.9 * t
cycles = 10 * np.sin(0.4 * t) + 5 * np.cos(0.8 * t)
noise = np.random.normal(0, 1.2, 1200)
prices = (trend + cycles + noise).reshape(-1, 1).astype(np.float32)

# 3. Normalization using MinMaxScaler
scaler_51 = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler_51.fit_transform(prices)

# 4. Sliding Window Dataset Creation (Window Size = 30 days)
window_size = 30
X_list, y_list = [], []
for i in range(len(scaled_prices) - window_size):
    X_list.append(scaled_prices[i:i + window_size])
    y_list.append(scaled_prices[i + window_size])

X_arr = np.array(X_list, dtype=np.float32)
y_arr = np.array(y_list, dtype=np.float32)

# Chronological 80/20 train/test split
train_size = int(len(X_arr) * 0.8)
X_train_51 = torch.tensor(X_arr[:train_size])
y_train_51 = torch.tensor(y_arr[:train_size])
X_test_51  = torch.tensor(X_arr[train_size:])
y_test_51  = torch.tensor(y_arr[train_size:])

train_loader_51 = DataLoader(TensorDataset(X_train_51, y_train_51), batch_size=32, shuffle=True)

# 5. Multi-Layer RNN Architecture for Forecasting
class StockForecastingRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)          # (batch, seq_len, hidden_size)
        return self.fc(out[:, -1, :]) # Predict next-step price

model_51 = StockForecastingRNN(input_size=1, hidden_size=64, num_layers=2)
criterion_51 = nn.MSELoss()
optimizer_51 = optim.Adam(model_51.parameters(), lr=0.005)

# 6. Training Loop
print("\n--- Training Stock Forecasting RNN ---")
epochs_51 = 25
for epoch in range(1, epochs_51 + 1):
    model_51.train()
    running_loss = 0.0
    for bx, by in train_loader_51:
        optimizer_51.zero_grad()
        preds = model_51(bx)
        loss = criterion_51(preds, by)
        loss.backward()
        optimizer_51.step()
        running_loss += loss.item()
    if epoch % 5 == 0 or epoch == epochs_51:
        print(f"Epoch [{epoch:2d}/{epochs_51}], MSE Loss: {running_loss / len(train_loader_51):.6f}")

# 7. Evaluation on Test Set
model_51.eval()
with torch.no_grad():
    y_pred_scaled = model_51(X_test_51).numpy()

# Invert normalization back to original USD ($) scale
y_pred_actual = scaler_51.inverse_transform(y_pred_scaled)
y_test_actual = scaler_51.inverse_transform(y_test_51.numpy())

mae_51 = mean_absolute_error(y_test_actual, y_pred_actual)
rmse_51 = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mape_51 = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100.0

print("\n" + "=" * 42)
print("       STOCK RNN EVALUATION METRICS       ")
print("=" * 42)
print(f"Mean Absolute Error (MAE)        : ${mae_51:.2f}")
print(f"Root Mean Squared Error (RMSE)   : ${rmse_51:.2f}")
print(f"Mean Absolute % Error (MAPE)     : {mape_51:.2f}%")
print("=" * 42)

# 8. Visualization
plt.figure(figsize=(10, 4))
plt.plot(y_test_actual, label='Actual Stock Price', color='navy', linewidth=1.8)
plt.plot(y_pred_actual, label='RNN Forecast', color='crimson', linestyle='--', linewidth=1.8)
plt.title('Program 51: Stock Price Time-Series Forecasting (Test Set)')
plt.xlabel('Trading Day')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('program51_stock_forecasting_rnn.png')
plt.show()

# Output comments:
# --- Training Stock Forecasting RNN ---
# Epoch [ 5/25], MSE Loss: 0.001699
# Epoch [10/25], MSE Loss: 0.000933
# Epoch [15/25], MSE Loss: 0.000760
# Epoch [20/25], MSE Loss: 0.000722
# Epoch [25/25], MSE Loss: 0.000674
# 
# ==========================================
#        STOCK RNN EVALUATION METRICS       
# ==========================================
# Mean Absolute Error (MAE)        : $1.22
# Root Mean Squared Error (RMSE)   : $1.53
# Mean Absolute % Error (MAPE)     : 0.83%
# ==========================================
# Conclusion: The RNN time-series model achieves high forecast accuracy
# with an average percentage error of ~0.83%, reliably tracking the underlying
# price trends and cyclic fluctuations.

# ======================================================================
