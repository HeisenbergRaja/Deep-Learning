# Program 33: Optimizer Comparison for Defective Product Detection
# Question: A manufacturing company develops a neural network to identify defective products. Train the model using SGD, RMSprop, and Adam optimizers and compare their convergence speed and classification accuracy. (BL5 – Evaluate)

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic manufacturing dataset for defective product detection
# 1000 product samples with 8 sensor features (vibration, heat, pressure, etc.)
num_samples = 1000
num_features = 8
X_raw = np.random.randn(num_samples, num_features).astype(np.float32)

# Defect condition: complex non-linear boundary on sensor features
defect_score = (
    1.8 * X_raw[:, 0]
    - 1.4 * X_raw[:, 1]
    + 1.2 * (X_raw[:, 2] ** 2)
    - 1.5 * X_raw[:, 3]
    + 0.5 * np.random.randn(num_samples)
)
y_raw = (defect_score > 1.2).astype(np.int64)

# 80/20 Train/Test split
train_size = 800
X_train = torch.tensor(X_raw[:train_size])
y_train = torch.tensor(y_raw[:train_size])
X_test = torch.tensor(X_raw[train_size:])
y_test = torch.tensor(y_raw[train_size:])

# Multi-layer neural network for defect classification
class DefectClassifier(nn.Module):
    def __init__(self, in_features=8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.network(x)

# Base model to ensure identical initial weights across all optimizers
base_model = DefectClassifier()
criterion = nn.CrossEntropyLoss()
epochs = 30

optimizers_to_compare = [
    ("SGD (with Momentum)", optim.SGD, {"lr": 0.05, "momentum": 0.9}),
    ("RMSprop", optim.RMSprop, {"lr": 0.005}),
    ("Adam", optim.Adam, {"lr": 0.005})
]

results = {}

print("=== Training Defective Product Detection Model across Optimizers ===")
for name, opt_class, opt_params in optimizers_to_compare:
    model = copy.deepcopy(base_model)
    optimizer = opt_class(model.parameters(), **opt_params)
    loss_history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test).argmax(dim=1)
        test_acc = (test_preds == y_test).float().mean().item() * 100
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        # Epochs to reach loss below 0.35 threshold (convergence speed)
        conv_epoch = next((i + 1 for i, l in enumerate(loss_history) if l < 0.35), epochs)

    results[name] = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "test_acc": test_acc,
        "conv_epoch": conv_epoch,
        "loss_history": loss_history
    }
    print(f"[{name:20s}] Epoch 10: {loss_history[9]:.4f} | Epoch 20: {loss_history[19]:.4f} | Final Loss: {final_loss:.4f} | Test Acc: {test_acc:.2f}%")

print("\n=== Convergence and Accuracy Comparison Table ===")
print(f"{'Optimizer':<22} | {'Initial Loss':<12} | {'Final Loss':<10} | {'Conv. Epoch (<0.35)':<20} | {'Test Accuracy':<12}")
print("-" * 85)
for name, res in results.items():
    print(f"{name:<22} | {res['initial_loss']:<12.4f} | {res['final_loss']:<10.4f} | {res['conv_epoch']:<20} | {res['test_acc']:<10.2f}%")

print("\nAnalysis:")
print("- Adam converges rapidly due to adaptive per-parameter learning rates and momentum.")
print("- RMSprop quickly lowers loss by adjusting step sizes according to moving average of squared gradients.")
print("- SGD with momentum accelerates steadily along persistent gradient directions.")

# Output:
# === Training Defective Product Detection Model across Optimizers ===
# [SGD (with Momentum) ] Epoch 10: 0.6407 | Epoch 20: 0.4963 | Final Loss: 0.3255 | Test Acc: 85.50%
# [RMSprop             ] Epoch 10: 0.2264 | Epoch 20: 0.1419 | Final Loss: 0.1159 | Test Acc: 92.50%
# [Adam                ] Epoch 10: 0.5532 | Epoch 20: 0.3703 | Final Loss: 0.2731 | Test Acc: 87.00%
# 
# === Convergence and Accuracy Comparison Table ===
# Optimizer              | Initial Loss | Final Loss | Conv. Epoch (<0.35)  | Test Accuracy
# -------------------------------------------------------------------------------------
# SGD (with Momentum)    | 0.6801       | 0.3255     | 29                   | 85.50     %
# RMSprop                | 0.6801       | 0.1159     | 6                    | 92.50     %
# Adam                   | 0.6801       | 0.2731     | 22                   | 87.00     %
# 
# Analysis:
# - Adam converges rapidly due to adaptive per-parameter learning rates and momentum.
# - RMSprop quickly lowers loss by adjusting step sizes according to moving average of squared gradients.
# - SGD with momentum accelerates steadily along persistent gradient directions.

# ======================================================================
