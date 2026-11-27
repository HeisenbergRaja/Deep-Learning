# Program 56: GRU weather forecasting
# Question: GRU model for multivariate time-series weather forecasting (temperature, humidity, pressure).

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

# 2. Generate Synthetic Multivariate Weather Time-Series (2000 hours)
n_hours = 2000
t_w = np.arange(n_hours)

# Temperature (°C): Diurnal solar heating cycle + weather front variations
temperature = 22 + 8 * np.sin(2 * np.pi * t_w / 24) + 4 * np.sin(2 * np.pi * t_w / 365) + np.random.normal(0, 0.8, n_hours)

# Relative Humidity (%): Inversely correlated with temperature
humidity = 65 - 18 * np.sin(2 * np.pi * t_w / 24) + np.random.normal(0, 2.5, n_hours)

# Atmospheric Pressure (hPa): Moving synoptic systems
pressure = 1013 + 5 * np.cos(2 * np.pi * t_w / 72) - 0.2 * (temperature - 22) + np.random.normal(0, 0.5, n_hours)

multivariate_weather = np.column_stack([temperature, humidity, pressure]).astype(np.float32)

# 3. Multivariate Scaling
scaler_56 = MinMaxScaler(feature_range=(0, 1))
scaled_weather = scaler_56.fit_transform(multivariate_weather)

# 4. Multivariate Sliding Window (24h history -> next hour 3 features)
lookback_56 = 24
X_56_list, y_56_list = [], []
for i in range(len(scaled_weather) - lookback_56):
    X_56_list.append(scaled_weather[i:i + lookback_56])
    y_56_list.append(scaled_weather[i + lookback_56])

X_56_arr = np.array(X_56_list, dtype=np.float32)
y_56_arr = np.array(y_56_list, dtype=np.float32)

# Chronological 80/20 train/test split
split_56 = int(len(X_56_arr) * 0.8)
X_tr_56 = torch.tensor(X_56_arr[:split_56])
y_tr_56 = torch.tensor(y_56_arr[:split_56])
X_te_56 = torch.tensor(X_56_arr[split_56:])
y_te_56 = torch.tensor(y_56_arr[split_56:])

weather_loader = DataLoader(TensorDataset(X_tr_56, y_tr_56), batch_size=32, shuffle=True)

# 5. Multivariate GRU Model Architecture
class MultivariateWeatherGRU(nn.Module):
    def __init__(self, in_features=3, hid_dim=64, out_features=3):
        super().__init__()
        self.gru = nn.GRU(in_features, hid_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.regressor = nn.Sequential(
            nn.Linear(hid_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_features)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.regressor(out[:, -1, :])

model_56 = MultivariateWeatherGRU(in_features=3, hid_dim=64, out_features=3)
criterion_56 = nn.MSELoss()
optimizer_56 = optim.Adam(model_56.parameters(), lr=0.005)

# 6. Training Loop
print("\n--- Training Multivariate Weather GRU ---")
epochs_56 = 25
for epoch in range(1, epochs_56 + 1):
    model_56.train()
    tot_loss = 0.0
    for bx, by in weather_loader:
        optimizer_56.zero_grad()
        loss = criterion_56(model_56(bx), by)
        loss.backward()
        optimizer_56.step()
        tot_loss += loss.item()
    if epoch % 5 == 0 or epoch == epochs_56:
        print(f"Epoch [{epoch:2d}/{epochs_56}], MSE Loss: {tot_loss / len(weather_loader):.6f}")

# 7. Evaluation & Metrics Calculation
model_56.eval()
with torch.no_grad():
    preds_scaled_56 = model_56(X_te_56).numpy()

preds_actual_56 = scaler_56.inverse_transform(preds_scaled_56)
actuals_56 = scaler_56.inverse_transform(y_te_56.numpy())

feature_names = ["Temperature (°C)", "Relative Humidity (%)", "Pressure (hPa)"]
units = ["°C", "%", "hPa"]

print("\n" + "=" * 56)
print("     MULTIVARIATE WEATHER FORECAST EVALUATION     ")
print("=" * 56)
for i in range(3):
    v_mae = mean_absolute_error(actuals_56[:, i], preds_actual_56[:, i])
    v_rmse = np.sqrt(mean_squared_error(actuals_56[:, i], preds_actual_56[:, i]))
    print(f"{feature_names[i]:<25} | MAE: {v_mae:6.3f} {units[i]} | RMSE: {v_rmse:6.3f} {units[i]}")
print("=" * 56)

# 8. Visualization: 3-panel subplot comparing actual vs forecasted series
plt.figure(figsize=(11, 7))
plot_hours = 72

plt.subplot(3, 1, 1)
plt.plot(actuals_56[:plot_hours, 0], label='Actual Temp (°C)', color='firebrick')
plt.plot(preds_actual_56[:plot_hours, 0], label='GRU Temp (°C)', color='salmon', linestyle='--')
plt.title('Program 56: Multivariate Weather Forecast (72-Hour Horizon)')
plt.ylabel('Temp (°C)')
plt.legend(loc='upper right')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(actuals_56[:plot_hours, 1], label='Actual Humidity (%)', color='teal')
plt.plot(preds_actual_56[:plot_hours, 1], label='GRU Humidity (%)', color='mediumturquoise', linestyle='--')
plt.ylabel('Humidity (%)')
plt.legend(loc='upper right')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(actuals_56[:plot_hours, 2], label='Actual Pressure (hPa)', color='indigo')
plt.plot(preds_actual_56[:plot_hours, 2], label='GRU Pressure (hPa)', color='orchid', linestyle='--')
plt.xlabel('Hour')
plt.ylabel('Pressure (hPa)')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.savefig('program56_gru_weather_forecast.png')
plt.show()

# Output comments:
# --- Training Multivariate Weather GRU ---
# Epoch [ 5/25], MSE Loss: 0.002866
# Epoch [10/25], MSE Loss: 0.002412
# Epoch [15/25], MSE Loss: 0.002468
# Epoch [20/25], MSE Loss: 0.002208
# Epoch [25/25], MSE Loss: 0.002165
# 
# ========================================================
#      MULTIVARIATE WEATHER FORECAST EVALUATION     
# ========================================================
# Temperature (°C)          | MAE:  0.952 °C | RMSE:  1.188 °C
# Relative Humidity (%)     | MAE:  2.320 % | RMSE:  2.863 %
# Pressure (hPa)            | MAE:  0.505 hPa | RMSE:  0.629 hPa
# ========================================================
# Conclusion: The multivariate GRU network captures the cross-variable
# correlations (e.g. inverse temperature-humidity dynamics) and temporal
# weather patterns, predicting all three atmospheric variables with low error.




# ======================================================================
# CO4 PROGRAMS 57 TO 64
# ======================================================================

# ======================================================================
