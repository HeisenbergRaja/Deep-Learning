# Program 62: RNN Evaluation Metrics (MAE, MSE, RMSE) for Electricity Consumption
# Question: An energy management company develops an RNN to forecast electricity consumption. Calculate MAE, MSE, and RMSE and evaluate the model's prediction errors. (BL5 – Evaluate)
# ======================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(42)
torch.manual_seed(42)

# Generate hourly electricity load data (1000 hours in Megawatts, MW)
# Daily load cycle (24h) + Weekly cycle (168h) + Noise + Outlier Peaks
hours = np.linspace(0, 200, 1000)
elec_load = 250.0 + 75.0 * np.sin(2 * np.pi * hours / 24) + 25.0 * np.sin(2 * np.pi * hours / 168) + np.random.normal(0, 6.0, len(hours))
# Simulate 20 high-load demand spikes (peak air conditioning / industrial surges)
spike_indices = np.random.choice(len(elec_load), 20, replace=False)
elec_load[spike_indices] += 40.0

# 24-hour historical window forecasting next hour demand
WINDOW_SIZE = 24
X_all, y_all = [], []
for i in range(len(elec_load) - WINDOW_SIZE):
    X_all.append(elec_load[i:i + WINDOW_SIZE])
    y_all.append(elec_load[i + WINDOW_SIZE])

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.float32)

split = int(0.8 * len(X_all))
X_train_raw, X_test_raw = X_all[:split], X_all[split:]
y_train_raw, y_test_raw = y_all[:split], y_all[split:]

load_mean = float(X_train_raw.mean())
load_std = float(X_train_raw.std())

X_train = torch.tensor((X_train_raw - load_mean) / load_std).unsqueeze(-1)
y_train = torch.tensor((y_train_raw - load_mean) / load_std).unsqueeze(-1)
X_test = torch.tensor((X_test_raw - load_mean) / load_std).unsqueeze(-1)
y_test = torch.tensor((y_test_raw - load_mean) / load_std).unsqueeze(-1)

# 2-Layer Recurrent Forecasting Network
class ElectricityDemandRNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2):
        super(ElectricityDemandRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

model = ElectricityDemandRNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training Electricity Consumption Forecasting RNN...")
for epoch in range(25):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()

# Evaluation on Test Data
model.eval()
with torch.no_grad():
    preds_norm = model(X_test).numpy().flatten()
    # Denormalize to Megawatts (MW)
    preds_mw = preds_norm * load_std + load_mean
    actual_mw = y_test_raw

# Metric Calculations
errors = actual_mw - preds_mw
mae = float(np.mean(np.abs(errors)))
mse = float(np.mean(errors ** 2))
rmse = float(np.sqrt(mse))
mape = float(np.mean(np.abs(errors / actual_mw))) * 100.0
max_error = float(np.max(np.abs(errors)))
ss_tot = np.sum((actual_mw - np.mean(actual_mw)) ** 2)
ss_res = np.sum(errors ** 2)
r2 = float(1.0 - ss_res / ss_tot)

print("\n" + "=" * 55)
print("      ELECTRICITY FORECAST EVALUATION REPORT")
print("=" * 55)
print(f"Mean Absolute Error (MAE)        : {mae:.4f} MW")
print(f"Mean Squared Error (MSE)         : {mse:.4f} MW²")
print(f"Root Mean Squared Error (RMSE)   : {rmse:.4f} MW")
print(f"Mean Absolute Percentage (MAPE)  : {mape:.2f} %")
print(f"Maximum Absolute Prediction Error: {max_error:.4f} MW")
print(f"Coefficient of Determination (R²): {r2:.4f}")
print("=" * 55)

print("\nSample Forecast Comparison (Next-Hour Electricity Load):")
print(f"{'Hour':<6} | {'Actual Load (MW)':<18} | {'Forecast (MW)':<16} | {'Error (MW)':<12}")
print("-" * 58)
for i in range(5):
    print(f"H+{i+1:<4} | {actual_mw[i]:<18.2f} | {preds_mw[i]:<16.2f} | {errors[i]:<+12.2f}")

print("\nEvaluation Commentary (BL5 - Evaluate):")
print(f"1. MAE ({mae:.2f} MW) reflects expected error magnitude across typical operating conditions.")
print(f"2. RMSE ({rmse:.2f} MW) > MAE ({mae:.2f} MW) indicates the presence of large error outliers,")
print("   specifically caused by simulated unexpected demand spikes.")
print(f"3. High R² ({r2:.4f}) and low MAPE ({mape:.2f}%) confirm excellent practical utility for grid management.")

# Output comments:
# Training Electricity Consumption Forecasting RNN...
# 
# =======================================================
#       ELECTRICITY FORECAST EVALUATION REPORT
# =======================================================
# Mean Absolute Error (MAE)        : 8.2854 MW
# Mean Squared Error (MSE)         : 123.7207 MW²
# Root Mean Squared Error (RMSE)   : 11.1230 MW
# Mean Absolute Percentage (MAPE)  : 3.31 %
# Maximum Absolute Prediction Error: 54.1304 MW
# Coefficient of Determination (R²): 0.9630
# =======================================================
# 
# Sample Forecast Comparison (Next-Hour Electricity Load):
# Hour   | Actual Load (MW)   | Forecast (MW)    | Error (MW)  
# ----------------------------------------------------------
# H+1    | 168.65             | 171.04           | -2.40       
# H+2    | 168.62             | 169.68           | -1.06       
# H+3    | 171.24             | 169.50           | +1.74       
# H+4    | 166.66             | 170.67           | -4.01       
# H+5    | 176.90             | 169.61           | +7.29       
# 
# Evaluation Commentary (BL5 - Evaluate):
# 1. MAE (8.29 MW) reflects expected error magnitude across typical operating conditions.
# 2. RMSE (11.12 MW) > MAE (8.29 MW) indicates the presence of large error outliers,
#    specifically caused by simulated unexpected demand spikes.
# 3. High R² (0.9630) and low MAPE (3.31%) confirm excellent practical utility for grid management.

# ======================================================================
