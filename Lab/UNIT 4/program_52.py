# Program 52: Electricity consumption RNN
# Question: RNN model for electricity consumption forecasting from historical data.

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

# 2. Generate Synthetic Hourly Electricity Consumption (2400 hours = 100 days)
total_hours = 2400
t_h = np.arange(total_hours)
# Diurnal 24-hour cycle + Weekly 168-hour pattern + Base load + Random noise
daily_cycle = 120 * np.sin(2 * np.pi * t_h / 24 - np.pi / 2)
weekly_cycle = 40 * np.sin(2 * np.pi * t_h / 168)
base_load = 550 + 0.04 * t_h
load_noise = np.random.normal(0, 15, total_hours)
consumption = (base_load + daily_cycle + weekly_cycle + load_noise).reshape(-1, 1).astype(np.float32)

# 3. Normalization
scaler_52 = MinMaxScaler(feature_range=(0, 1))
scaled_consumption = scaler_52.fit_transform(consumption)

# 4. Create 24-hour lookback sequences to predict the next hour
seq_len_52 = 24
X_52_list, y_52_list = [], []
for i in range(len(scaled_consumption) - seq_len_52):
    X_52_list.append(scaled_consumption[i:i + seq_len_52])
    y_52_list.append(scaled_consumption[i + seq_len_52])

X_52_arr = np.array(X_52_list, dtype=np.float32)
y_52_arr = np.array(y_52_list, dtype=np.float32)

# Chronological 80/20 Train/Test split
split_idx = int(len(X_52_arr) * 0.8)
X_train_52 = torch.tensor(X_52_arr[:split_idx])
y_train_52 = torch.tensor(y_52_arr[:split_idx])
X_test_52  = torch.tensor(X_52_arr[split_idx:])
y_test_52  = torch.tensor(y_52_arr[split_idx:])

train_loader_52 = DataLoader(TensorDataset(X_train_52, y_train_52), batch_size=32, shuffle=True)

# 5. Deep RNN Forecasting Model
class ElectricityConsumptionRNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

model_52 = ElectricityConsumptionRNN(input_dim=1, hidden_dim=64, num_layers=2)
criterion_52 = nn.MSELoss()
optimizer_52 = optim.Adam(model_52.parameters(), lr=0.005)

# 6. Training
print("\n--- Training Electricity Consumption RNN ---")
epochs_52 = 20
for epoch in range(1, epochs_52 + 1):
    model_52.train()
    loss_acc = 0.0
    for bx, by in train_loader_52:
        optimizer_52.zero_grad()
        out = model_52(bx)
        loss = criterion_52(out, by)
        loss.backward()
        optimizer_52.step()
        loss_acc += loss.item()
    if epoch % 5 == 0 or epoch == epochs_52:
        print(f"Epoch [{epoch:2d}/{epochs_52}], MSE Loss: {loss_acc / len(train_loader_52):.6f}")

# 7. Evaluation
model_52.eval()
with torch.no_grad():
    pred_test_scaled = model_52(X_test_52).numpy()

pred_test_actual = scaler_52.inverse_transform(pred_test_scaled)
y_test_actual = scaler_52.inverse_transform(y_test_52.numpy())

mae_52 = mean_absolute_error(y_test_actual, pred_test_actual)
rmse_52 = np.sqrt(mean_squared_error(y_test_actual, pred_test_actual))
r2_52 = r2_score(y_test_actual, pred_test_actual)

print("\n" + "=" * 46)
print("     ELECTRICITY CONSUMPTION FORECAST METRICS  ")
print("=" * 46)
print(f"Mean Absolute Error (MAE)      : {mae_52:.2f} MW")
print(f"Root Mean Squared Error (RMSE) : {rmse_52:.2f} MW")
print(f"R-squared (R2 Score)           : {r2_52:.4f}")
print("=" * 46)

# 8. Visualization: 7-day (168 hours) snapshot of forecast vs actual
plt.figure(figsize=(11, 4))
plot_hours = 168
plt.plot(y_test_actual[:plot_hours], label='Actual Consumption (MW)', color='forestgreen', linewidth=1.8)
plt.plot(pred_test_actual[:plot_hours], label='RNN Forecast (MW)', color='darkorange', linestyle='--', linewidth=1.8)
plt.title('Program 52: Hourly Electricity Consumption Forecast (1-Week Profile)')
plt.xlabel('Hour of Test Period')
plt.ylabel('Electricity Load (Megawatts)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('program52_electricity_rnn.png')
plt.show()

# Output comments:
# --- Training Electricity Consumption RNN ---
# Epoch [ 5/20], MSE Loss: 0.002795
# Epoch [10/20], MSE Loss: 0.002803
# Epoch [15/20], MSE Loss: 0.002746
# Epoch [20/20], MSE Loss: 0.002916
# 
# ==============================================
#      ELECTRICITY CONSUMPTION FORECAST METRICS  
# ==============================================
# Mean Absolute Error (MAE)      : 30.25 MW
# Root Mean Squared Error (RMSE) : 35.50 MW
# R-squared (R2 Score)           : 0.8465
# ==============================================
# Conclusion: The RNN model captures diurnal daily demand peaks and
# weekly cyclical load variations with an R2 score of ~0.847, demonstrating
# effective utility for grid load management and scheduling.

# ======================================================================
