# Program 55: LSTM stock prediction
# Question: LSTM model for stock-price trends, evaluate with regression metrics (MAE, RMSE, R2).

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 2. Generate Synthetic Stock Price Series (1200 time steps)
t_55 = np.linspace(0, 60, 1200)
# Trend + multiple cyclical harmonics + noise
stock_trend = 150 + 1.2 * t_55
harmonics = 12 * np.sin(0.3 * t_55) + 6 * np.cos(0.7 * t_55)
stochastic_noise = np.random.normal(0, 1.5, 1200)
stock_series = (stock_trend + harmonics + stochastic_noise).reshape(-1, 1).astype(np.float32)

# 3. Normalization
scaler_55 = MinMaxScaler(feature_range=(0, 1))
scaled_stock_55 = scaler_55.fit_transform(stock_series)

# 4. 30-Day Sliding Window
win_size_55 = 30
X_55_list, y_55_list = [], []
for i in range(len(scaled_stock_55) - win_size_55):
    X_55_list.append(scaled_stock_55[i:i + win_size_55])
    y_55_list.append(scaled_stock_55[i + win_size_55])

X_55_arr = np.array(X_55_list, dtype=np.float32)
y_55_arr = np.array(y_55_list, dtype=np.float32)

split_55 = int(len(X_55_arr) * 0.8)
X_tr_55, X_te_55 = torch.tensor(X_55_arr[:split_55]), torch.tensor(X_55_arr[split_55:])
y_tr_55, y_te_55 = torch.tensor(y_55_arr[:split_55]), torch.tensor(y_55_arr[split_55:])

train_loader_55 = DataLoader(TensorDataset(X_tr_55, y_tr_55), batch_size=32, shuffle=True)

# 5. Stacked LSTM Architecture
class StackedLSTMPricePredictor(nn.Module):
    def __init__(self, in_dim=1, hid_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.regressor = nn.Sequential(
            nn.Linear(hid_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.regressor(out[:, -1, :])

model_55 = StackedLSTMPricePredictor(in_dim=1, hid_dim=64)
criterion_55 = nn.MSELoss()
optimizer_55 = optim.Adam(model_55.parameters(), lr=0.005)

# 6. Training Loop
print("\n--- Training Stacked LSTM Stock Predictor ---")
epochs_55 = 30
for epoch in range(1, epochs_55 + 1):
    model_55.train()
    tot_loss = 0.0
    for bx, by in train_loader_55:
        optimizer_55.zero_grad()
        loss = criterion_55(model_55(bx), by)
        loss.backward()
        optimizer_55.step()
        tot_loss += loss.item()
    if epoch % 10 == 0 or epoch == epochs_55:
        print(f"Epoch [{epoch:2d}/{epochs_55}], MSE Loss: {tot_loss / len(train_loader_55):.6f}")

# 7. Test Evaluation & Regression Metrics
model_55.eval()
with torch.no_grad():
    preds_scaled_55 = model_55(X_te_55).numpy()

preds_usd = scaler_55.inverse_transform(preds_scaled_55)
actuals_usd = scaler_55.inverse_transform(y_te_55.numpy())

mae_55 = mean_absolute_error(actuals_usd, preds_usd)
mse_55 = mean_squared_error(actuals_usd, preds_usd)
rmse_55 = np.sqrt(mse_55)
r2_55 = r2_score(actuals_usd, preds_usd)
mape_55 = np.mean(np.abs((actuals_usd - preds_usd) / actuals_usd)) * 100.0

print("\n" + "=" * 48)
print("     LSTM STOCK PREDICTION REGRESSION METRICS   ")
print("=" * 48)
print(f"Mean Absolute Error (MAE)        : ${mae_55:.2f}")
print(f"Mean Squared Error (MSE)         : {mse_55:.2f}")
print(f"Root Mean Squared Error (RMSE)   : ${rmse_55:.2f}")
print(f"Coefficient of Determination (R2): {r2_55:.4f}")
print(f"Mean Absolute % Error (MAPE)     : {mape_55:.2f}%")
print("=" * 48)

# 8. Visualization
plt.figure(figsize=(11, 6))

plt.subplot(2, 1, 1)
plt.plot(actuals_usd, label='Actual Price ($)', color='royalblue', linewidth=1.8)
plt.plot(preds_usd, label='LSTM Forecast ($)', color='darkorange', linestyle='--', linewidth=1.8)
plt.title(f'Program 55: LSTM Stock Trend Prediction (R2 Score = {r2_55:.4f})')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
residuals = actuals_usd - preds_usd
plt.plot(residuals, color='crimson', linewidth=1.2, label='Residual Error (Actual - Pred)')
plt.axhline(0, color='black', linestyle=':')
plt.title(f'Residual Analysis (MAE = ${mae_55:.2f})')
plt.xlabel('Trading Day')
plt.ylabel('Residual ($)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('program55_lstm_stock_prediction.png')
plt.show()

# Output comments:
# --- Training Stacked LSTM Stock Predictor ---
# Epoch [10/30], MSE Loss: 0.000754
# Epoch [20/30], MSE Loss: 0.000974
# Epoch [30/30], MSE Loss: 0.000741
# 
# ================================================
#      LSTM STOCK PREDICTION REGRESSION METRICS   
# ================================================
# Mean Absolute Error (MAE)        : $1.44
# Mean Squared Error (MSE)         : 3.50
# Root Mean Squared Error (RMSE)   : $1.87
# Coefficient of Determination (R2): 0.8659
# Mean Absolute % Error (MAPE)     : 0.68%
# ================================================
# Conclusion: The stacked LSTM model demonstrates high predictive power
# with an R2 score of 0.8659 and a MAPE of under 1%, accurately capturing
# long-term upward price trends and short-term volatility.

# ======================================================================
