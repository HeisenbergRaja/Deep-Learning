# Program 58: Stacked RNN for Multivariate Industrial Machine Condition Prediction
# Question: A smart factory collects temperature, pressure, and vibration readings from industrial machines. Develop a stacked RNN model to predict future machine conditions using multivariate time-series data. (BL6 – Create)
# ======================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic multivariate sensor readings (1200 time steps)
N = 1200
t = np.linspace(0, 100, N)

# Multi-channel industrial sensors
temp = 65.0 + 10.0 * np.sin(0.15 * t) + 3.0 * np.cos(0.3 * t) + np.random.normal(0, 0.3, N)    # Temperature (°C)
pressure = 4.0 + 1.2 * np.cos(0.12 * t) + 0.5 * np.sin(0.25 * t) + np.random.normal(0, 0.1, N) # Pressure (bar)
vibration = 1.2 + 0.6 * np.sin(0.18 * t) + 0.3 * np.cos(0.22 * t) + np.random.normal(0, 0.05, N) # Vibration (mm/s)

# Target: Machine Condition / Health Index (0-100, 100 = pristine, lower = degraded)
condition_index = 100.0 - (0.4 * (temp - 55.0) + 5.0 * (pressure - 3.0) + 15.0 * (vibration - 0.6)) + np.random.normal(0, 0.2, N)

# Combine multivariate features (N, 3)
sensors = np.stack([temp, pressure, vibration], axis=1)

# Normalization parameters based on training distribution
WINDOW_LEN = 15
X_multi, y_cond = [], []
for i in range(len(sensors) - WINDOW_LEN):
    X_multi.append(sensors[i:i + WINDOW_LEN])
    y_cond.append(condition_index[i + WINDOW_LEN])

X_multi = np.array(X_multi, dtype=np.float32)
y_cond = np.array(y_cond, dtype=np.float32)

split_idx = int(0.8 * len(X_multi))
X_train_raw, X_test_raw = X_multi[:split_idx], X_multi[split_idx:]
y_train_raw, y_test_raw = y_cond[:split_idx], y_cond[split_idx:]

feat_mean = X_train_raw.mean(axis=(0, 1), keepdims=True)
feat_std = X_train_raw.std(axis=(0, 1), keepdims=True) + 1e-8
y_mean = y_train_raw.mean()
y_std = y_train_raw.std() + 1e-8

X_train = torch.tensor((X_train_raw - feat_mean) / feat_std)
X_test = torch.tensor((X_test_raw - feat_mean) / feat_std)
y_train = torch.tensor((y_train_raw - y_mean) / y_std).unsqueeze(-1)
y_test = torch.tensor((y_test_raw - y_mean) / y_std).unsqueeze(-1)

# Multi-layer Stacked RNN Architecture
class StackedMachineConditionRNN(nn.Module):
    def __init__(self, num_features=3, hidden_dim=48, num_layers=3, dropout=0.2):
        super(StackedMachineConditionRNN, self).__init__()
        self.stacked_rnn = nn.RNN(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        rnn_out, _ = self.stacked_rnn(x)
        # Use final time step hidden representation
        last_hidden = rnn_out[:, -1, :]
        return self.regressor(last_hidden)

stacked_model = StackedMachineConditionRNN(num_features=3, hidden_dim=48, num_layers=3, dropout=0.2)
criterion = nn.MSELoss()
optimizer = optim.Adam(stacked_model.parameters(), lr=0.008)

print("Training 3-Layer Stacked RNN on Multivariate Sensor Streams...")
for epoch in range(35):
    stacked_model.train()
    optimizer.zero_grad()
    predictions = stacked_model(X_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch [{epoch+1:2d}/35] - Training MSE Loss: {loss.item():.5f}")

# Evaluation on Unseen Test Sequences
stacked_model.eval()
with torch.no_grad():
    test_preds_norm = stacked_model(X_test).numpy().flatten()
    # Denormalize to original condition index scale
    test_preds = test_preds_norm * y_std + y_mean
    actual_condition = y_test_raw

mse_58 = float(np.mean((actual_condition - test_preds) ** 2))
rmse_58 = float(np.sqrt(mse_58))
mae_58 = float(np.mean(np.abs(actual_condition - test_preds)))
ss_tot = np.sum((actual_condition - np.mean(actual_condition)) ** 2)
ss_res = np.sum((actual_condition - test_preds) ** 2)
r2_58 = float(1.0 - (ss_res / (ss_tot + 1e-9)))

print("\n--- Stacked RNN Machine Condition Prediction Results ---")
print(f"Mean Squared Error (MSE)      : {mse_58:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_58:.4f}")
print(f"Mean Absolute Error (MAE)     : {mae_58:.4f}")
print(f"Coefficient of Determination R²: {r2_58:.4f}")

print("\nSample Predicted vs Actual Machine Health Index:")
print(f"{'Sample #':<10} | {'Predicted Index':<18} | {'Actual Index':<18} | {'Absolute Error':<15}")
print("-" * 68)
for i in range(5):
    err = abs(actual_condition[i] - test_preds[i])
    print(f"Sample {i+1:<3} | {test_preds[i]:<18.2f} | {actual_condition[i]:<18.2f} | {err:<15.2f}")

# Output comments:
# Training 3-Layer Stacked RNN on Multivariate Sensor Streams...
#   Epoch [10/35] - Training MSE Loss: 0.10003
#   Epoch [20/35] - Training MSE Loss: 0.03513
#   Epoch [30/35] - Training MSE Loss: 0.02200
# 
# --- Stacked RNN Machine Condition Prediction Results ---
# Mean Squared Error (MSE)      : 1.4568
# Root Mean Squared Error (RMSE): 1.2070
# Mean Absolute Error (MAE)     : 0.9689
# Coefficient of Determination R²: 0.9634
# 
# Sample Predicted vs Actual Machine Health Index:
# Sample #   | Predicted Index    | Actual Index       | Absolute Error 
# --------------------------------------------------------------------
# Sample 1   | 76.56              | 77.03              | 0.47           
# Sample 2   | 76.33              | 76.40              | 0.07           
# Sample 3   | 75.92              | 75.12              | 0.80           
# Sample 4   | 75.29              | 77.29              | 1.99           
# Sample 5   | 75.31              | 76.15              | 0.84           

# ======================================================================
