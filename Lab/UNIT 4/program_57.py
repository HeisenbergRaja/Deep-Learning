# Program 57: RNN vs LSTM vs GRU Sequence Prediction Comparison
# Question: A research organization wants to identify the most effective recurrent architecture for sequence prediction. Implement RNN, LSTM, and GRU models using a common dataset and evaluate their performance. (BL5 – Evaluate)
# ======================================================================

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate common synthetic sequential dataset: composite wave with noise
time_steps = np.linspace(0, 60, 1000)
series = np.sin(0.2 * time_steps) + 0.5 * np.cos(0.05 * time_steps) + 0.05 * np.random.randn(len(time_steps))

# Sliding window sequence creation
def create_sequences(data, window_size=15):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

WINDOW_SIZE = 15
X, y = create_sequences(series, window_size=WINDOW_SIZE)

# Train/Test Split (80% train, 20% test)
split = int(0.8 * len(X))
X_train = torch.tensor(X[:split]).unsqueeze(-1)
y_train = torch.tensor(y[:split]).unsqueeze(-1)
X_test = torch.tensor(X[split:]).unsqueeze(-1)
y_test = torch.tensor(y[split:]).unsqueeze(-1)

# Unified Recurrent Model Wrapper
class SequencePredictor(nn.Module):
    def __init__(self, cell_type='RNN', input_dim=1, hidden_dim=32, output_dim=1):
        super(SequencePredictor, self).__init__()
        self.cell_type = cell_type.upper()
        if self.cell_type == 'RNN':
            self.recurrent = nn.RNN(input_dim, hidden_dim, batch_first=True)
        elif self.cell_type == 'LSTM':
            self.recurrent = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        elif self.cell_type == 'GRU':
            self.recurrent = nn.GRU(input_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.recurrent(x)
        return self.fc(out[:, -1, :])

# Train and evaluate RNN, LSTM, and GRU
models = ['RNN', 'LSTM', 'GRU']
performance_metrics = {}

print("Training and evaluating recurrent architectures on common dataset...")
for model_name in models:
    torch.manual_seed(42)
    model = SequencePredictor(cell_type=model_name, hidden_dim=32)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    start_time = time.time()
    for epoch in range(25):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
    elapsed_time = time.time() - start_time

    # Evaluation on unseen test set
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test).numpy()
        test_actual = y_test.numpy()
        mse = float(np.mean((test_actual - test_preds) ** 2))
        mae = float(np.mean(np.abs(test_actual - test_preds)))
        rmse = float(np.sqrt(mse))

    performance_metrics[model_name] = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'Time (s)': elapsed_time
    }

print("\n" + "=" * 65)
print(f"{'Architecture':<14} | {'Test MSE':<12} | {'Test MAE':<12} | {'Test RMSE':<12} | {'Time (s)':<10}")
print("=" * 65)
for arch, m in performance_metrics.items():
    print(f"{arch:<14} | {m['MSE']:<12.5f} | {m['MAE']:<12.5f} | {m['RMSE']:<12.5f} | {m['Time (s)']:<10.3f}")
print("=" * 65)

# Architectural Evaluation Insight
best_arch = min(performance_metrics, key=lambda k: performance_metrics[k]['MSE'])
print(f"\nEvaluation Conclusion: {best_arch} achieved the lowest prediction error.")
print("Simple RNN converges fast on smooth, short-dependency series.")
print("GRU provides a streamlined gating mechanism with lower parameter count than LSTM.")
print("LSTM maintains long-term cell memory via dedicated forget, input, and output gates.")

# Output comments:
# Training and evaluating recurrent architectures on common dataset...
# 
# =================================================================
# Architecture   | Test MSE     | Test MAE     | Test RMSE    | Time (s)  
# =================================================================
# RNN            | 0.06705      | 0.24210      | 0.25894      | 0.138     
# LSTM           | 0.24968      | 0.47844      | 0.49968      | 0.260     
# GRU            | 0.04372      | 0.19722      | 0.20910      | 0.571     
# =================================================================
# 
# Evaluation Conclusion: GRU achieved the lowest prediction error.
# Simple RNN converges fast on smooth, short-dependency series.
# GRU provides a streamlined gating mechanism with lower parameter count than LSTM.
# LSTM maintains long-term cell memory via dedicated forget, input, and output gates.

# ======================================================================
