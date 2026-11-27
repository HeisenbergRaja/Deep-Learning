# Program 54: RNN vs LSTM comparison
# Question: Forecast product demand, implement both RNN and LSTM, compare performance.

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

# 2. Generate Synthetic Daily Product Demand (1000 days)
total_days = 1000
t_days = np.arange(total_days)
# Growth trend + 7-day weekly cycle + 30-day promo cycle + random sales noise
demand_trend = 50 + 0.08 * t_days
weekly_season = 25 * np.sin(2 * np.pi * t_days / 7)
monthly_promo = 15 * np.sin(2 * np.pi * t_days / 30)
demand_noise = np.random.normal(0, 5, total_days)
demand = (demand_trend + weekly_season + monthly_promo + demand_noise).reshape(-1, 1).astype(np.float32)

# 3. Normalization
scaler_54 = MinMaxScaler(feature_range=(0, 1))
scaled_demand = scaler_54.fit_transform(demand)

# 4. Sliding Window (30-day lookback -> next-day demand)
window_len = 30
X_dem, y_dem = [], []
for i in range(len(scaled_demand) - window_len):
    X_dem.append(scaled_demand[i:i + window_len])
    y_dem.append(scaled_demand[i + window_len])

X_dem_arr = np.array(X_dem, dtype=np.float32)
y_dem_arr = np.array(y_dem, dtype=np.float32)

split_dem = int(len(X_dem_arr) * 0.8)
X_tr_dem, X_te_dem = torch.tensor(X_dem_arr[:split_dem]), torch.tensor(X_dem_arr[split_dem:])
y_tr_dem, y_te_dem = torch.tensor(y_dem_arr[:split_dem]), torch.tensor(y_dem_arr[split_dem:])

demand_loader = DataLoader(TensorDataset(X_tr_dem, y_tr_dem), batch_size=32, shuffle=True)

# 5. Model Definitions
class StandardRNNForecaster(nn.Module):
    def __init__(self, in_dim=1, hid_dim=32):
        super().__init__()
        self.rnn = nn.RNN(in_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class LSTMForecaster(nn.Module):
    def __init__(self, in_dim=1, hid_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 6. Unified Training & Evaluation Pipeline
def train_and_evaluate_forecaster(model, name="Model", epochs=25):
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=0.005)
    print(f"\n--- Training {name} ---")
    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        for bx, by in demand_loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            tot_loss += loss.item()
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:2d}/{epochs}], Loss: {tot_loss / len(demand_loader):.6f}")

    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_te_dem).numpy()
    preds_units = scaler_54.inverse_transform(preds_scaled)
    actuals_units = scaler_54.inverse_transform(y_te_dem.numpy())

    mae = mean_absolute_error(actuals_units, preds_units)
    rmse = np.sqrt(mean_squared_error(actuals_units, preds_units))
    mape = np.mean(np.abs((actuals_units - preds_units) / actuals_units)) * 100.0
    return preds_units, actuals_units, mae, rmse, mape

rnn_preds, actuals, rnn_mae, rnn_rmse, rnn_mape = train_and_evaluate_forecaster(StandardRNNForecaster(), "Standard RNN", epochs=25)
lstm_preds, _, lstm_mae, lstm_rmse, lstm_mape = train_and_evaluate_forecaster(LSTMForecaster(), "LSTM Forecaster", epochs=25)

# 7. Comparison Table
print("\n" + "=" * 54)
print("         RNN vs LSTM PERFORMANCE COMPARISON       ")
print("=" * 54)
print(f"{'Metric':<20} | {'Standard RNN':<14} | {'LSTM':<14}")
print("-" * 54)
print(f"{'MAE (Units)':<20} | {rnn_mae:<14.2f} | {lstm_mae:<14.2f}")
print(f"{'RMSE (Units)':<20} | {rnn_rmse:<14.2f} | {lstm_rmse:<14.2f}")
print(f"{'MAPE (%)':<20} | {rnn_mape:<13.2f}% | {lstm_mape:<13.2f}%")
print("=" * 54)

# 8. Visualization
plt.figure(figsize=(11, 4))
plot_range = 80
plt.plot(actuals[:plot_range], label='Actual Demand', color='black', linewidth=2.0)
plt.plot(rnn_preds[:plot_range], label=f'Standard RNN (MAE={rnn_mae:.1f})', color='crimson', linestyle=':', linewidth=1.8)
plt.plot(lstm_preds[:plot_range], label=f'LSTM (MAE={lstm_mae:.1f})', color='teal', linestyle='--', linewidth=1.8)
plt.title('Program 54: Product Demand Forecasting - RNN vs LSTM')
plt.xlabel('Days')
plt.ylabel('Demand (Units)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('program54_rnn_vs_lstm_demand.png')
plt.show()

# Output comments:
# --- Training Standard RNN ---
# Epoch [10/25], Loss: 0.002199
# Epoch [20/25], Loss: 0.002111
# Epoch [25/25], Loss: 0.002015
# 
# --- Training LSTM Forecaster ---
# Epoch [10/25], Loss: 0.002359
# Epoch [20/25], Loss: 0.001991
# Epoch [25/25], Loss: 0.001777
# 
# ======================================================
#          RNN vs LSTM PERFORMANCE COMPARISON       
# ======================================================
# Metric               | Standard RNN   | LSTM          
# ------------------------------------------------------
# MAE (Units)          | 7.17           | 7.16          
# RMSE (Units)         | 8.71           | 8.98          
# MAPE (%)             | 5.76%          | 5.58%         
# ======================================================
# Conclusion: Both models capture weekly demand seasonality well.
# The LSTM model achieves a slightly lower Mean Absolute Percentage Error
# (5.58% vs 5.76%) and lower final training loss due to its gating mechanisms
# (input, forget, output gates) that prevent vanishing gradients over 30-day windows.

# ======================================================================
