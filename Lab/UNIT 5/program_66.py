# Program 66: RNN vs LSTM long sequences
# Question: A research organization observes that a simple RNN performs poorly when predicting sequences containing long-term dependencies. Compare RNN and LSTM models using long-sequence prediction and evaluate their ability to retain historical information. (BL5 – Evaluate)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# 1. Generate Synthetic Long Sequence Time-Series
# Sequence length T=40: combines a low-frequency wave (historical context) and high-frequency oscillation
T = 40
t = np.linspace(0, 80, 1000)
signal = np.sin(t / 8.0) + 0.4 * np.cos(t * 1.5)

X_list, y_list = [], []
for i in range(len(signal) - T):
    X_list.append(signal[i:i+T])
    y_list.append(signal[i+T])

X = np.array(X_list, dtype=np.float32)[:, :, np.newaxis]
y = np.array(y_list, dtype=np.float32)[:, np.newaxis]

train_sz = 750
X_tr, y_tr = torch.tensor(X[:train_sz]), torch.tensor(y[:train_sz])
X_te, y_te = torch.tensor(X[train_sz:]), torch.tensor(y[train_sz:])

# 2. Define Simple RNN and LSTM Architectures
class SimpleRNN(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_dim, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class LSTMModel(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 3. Training and Evaluation Harness
def train_and_eval(model, epochs=60):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    losses = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr), y_tr)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        preds = model(X_te)
        mse = criterion(preds, y_te).item()
        mae = torch.mean(torch.abs(preds - y_te)).item()
    return losses, mse, mae

rnn_m = SimpleRNN()
lstm_m = LSTMModel()

rnn_losses, rnn_mse, rnn_mae = train_and_eval(rnn_m)
lstm_losses, lstm_mse, lstm_mae = train_and_eval(lstm_m)

# 4. Comparative Evaluation
print(f"{'Model Architecture':<20} | {'Test MSE':<12} | {'Test MAE':<12}")
print("-" * 50)
print(f"{'Simple RNN (Tanh)':<20} | {rnn_mse:<12.6f} | {rnn_mae:<12.6f}")
print(f"{'LSTM (Gated Memory)':<20} | {lstm_mse:<12.6f} | {lstm_mae:<12.6f}")
error_diff = ((rnn_mse - lstm_mse) / rnn_mse) * 100
print(f"Error Reduction: {error_diff:.2f}% MSE improvement with LSTM.")
print("Evaluation: LSTM's constant error carousel and gating mechanism retain long-range historical information.")

# 5. Visualization: Learning Curves
plt.figure(figsize=(8, 4))
plt.plot(rnn_losses, label="Simple RNN Loss", color="red", linestyle="--")
plt.plot(lstm_losses, label="LSTM Loss", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Training MSE Loss")
plt.title("Long-Sequence Learning: Simple RNN vs LSTM Convergence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Output:
# Model Architecture   | Test MSE     | Test MAE    
# --------------------------------------------------
# Simple RNN (Tanh)    | 0.000745     | 0.022974    
# LSTM (Gated Memory)  | 0.000673     | 0.022187    
# Error Reduction: 9.72% MSE improvement with LSTM.
# Evaluation: LSTM's constant error carousel and gating mechanism retain long-range historical information.

# ======================================================================
