# Program 17: Overfitting Analysis: Neural Network for Student Exam Performance
# Question: A university develops a neural network to predict student examination performance. The model achieves high training accuracy but poor testing accuracy. Analyse the training and testing errors to identify the overfitting problem and suggest suitable improvements. (BL4 – Analyse)

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic student exam performance dataset (80 students, 6 features)
# Features: [Study Hours, Past Score, Attendance, Sleep Hours, Practice Tests, Assignment Rate]
N_17 = 80
X_17 = np.random.randn(N_17, 6).astype(np.float32)
# Latent true relation with noise to simulate exam outcomes
latent_17 = X_17[:, 0] * 1.6 - X_17[:, 1] * 1.3 + X_17[:, 2] * 0.9 + np.random.randn(N_17) * 0.8
y_prob_17 = 1.0 / (1.0 + np.exp(-latent_17))
y_17 = (y_prob_17 > 0.5).astype(np.float32)

# Inject 15% random label noise to create high-variance risk for complex networks
flip_idx = np.random.choice(N_17, size=int(0.15 * N_17), replace=False)
y_17[flip_idx] = 1.0 - y_17[flip_idx]

# Train-test split (65% train, 35% test)
X_tr17, X_te17, y_tr17, y_te17 = train_test_split(X_17, y_17, test_size=0.35, random_state=42)

scaler17 = StandardScaler()
X_tr17_s = torch.tensor(scaler17.fit_transform(X_tr17), dtype=torch.float32)
X_te17_s = torch.tensor(scaler17.transform(X_te17), dtype=torch.float32)
y_tr17_t = torch.tensor(y_tr17, dtype=torch.float32).unsqueeze(1)
y_te17_t = torch.tensor(y_te17, dtype=torch.float32).unsqueeze(1)

# 1. Overfitted Deep Neural Network (Overparameterized architecture without regularization)
overfit_net = nn.Sequential(
    nn.Linear(6, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

criterion_17 = nn.BCELoss()
optimizer_overfit = torch.optim.Adam(overfit_net.parameters(), lr=0.01)

for epoch in range(300):
    optimizer_overfit.zero_grad()
    loss = criterion_17(overfit_net(X_tr17_s), y_tr17_t)
    loss.backward()
    optimizer_overfit.step()

with torch.no_grad():
    tr_pred_o = (overfit_net(X_tr17_s) >= 0.5).float()
    te_pred_o = (overfit_net(X_te17_s) >= 0.5).float()
    tr_acc_o = (tr_pred_o == y_tr17_t).float().mean().item() * 100
    te_acc_o = (te_pred_o == y_te17_t).float().mean().item() * 100
    tr_loss_o = criterion_17(overfit_net(X_tr17_s), y_tr17_t).item()
    te_loss_o = criterion_17(overfit_net(X_te17_s), y_te17_t).item()

# 2. Improved Model with Regularization, Dropout, and Compact Architecture
reg_net = nn.Sequential(
    nn.Linear(6, 8),
    nn.ReLU(),
    nn.Dropout(p=0.35),
    nn.Linear(8, 1),
    nn.Sigmoid()
)
optimizer_reg = torch.optim.Adam(reg_net.parameters(), lr=0.01, weight_decay=0.05)

for epoch in range(300):
    reg_net.train()
    optimizer_reg.zero_grad()
    loss = criterion_17(reg_net(X_tr17_s), y_tr17_t)
    loss.backward()
    optimizer_reg.step()

reg_net.eval()
with torch.no_grad():
    tr_pred_r = (reg_net(X_tr17_s) >= 0.5).float()
    te_pred_r = (reg_net(X_te17_s) >= 0.5).float()
    tr_acc_r = (tr_pred_r == y_tr17_t).float().mean().item() * 100
    te_acc_r = (te_pred_r == y_te17_t).float().mean().item() * 100
    tr_loss_r = criterion_17(reg_net(X_tr17_s), y_tr17_t).item()
    te_loss_r = criterion_17(reg_net(X_te17_s), y_te17_t).item()

print("--- Program 17: Overfitting Analysis ---")
print(f"Overfitted Model  | Train Acc: {tr_acc_o:5.1f}% | Test Acc: {te_acc_o:5.1f}% | Train Loss: {tr_loss_o:.4f} | Test Loss: {te_loss_o:.4f}")
print(f"Regularized Model | Train Acc: {tr_acc_r:5.1f}% | Test Acc: {te_acc_r:5.1f}% | Train Loss: {tr_loss_r:.4f} | Test Loss: {te_loss_r:.4f}")
print("\n[Analysis & Suggested Improvements]:")
print("1. Diagnosis: Large gap between train loss (~0.00) and test loss (>10.0) confirms high variance / overfitting.")
print("2. Dropout: Randomly deactivates neurons during forward passes, mitigating co-adaptation.")
print("3. L2 Weight Decay: Constrains weight magnitudes to prevent memorization of sample noise.")
print("4. Architecture Simplification: Restricting parameter count matches model capacity to dataset size.")
print("5. Early Stopping: Halt gradient updates once validation loss stops decreasing.")

# Output:
# Overfitted Model  | Train Acc: 100.0% | Test Acc:  42.9% | Train Loss: 0.0000 | Test Loss: 15.2985
# Regularized Model | Train Acc:  82.7% | Test Acc:  60.7% | Train Loss: 0.5368 | Test Loss: 0.6288
# [Analysis & Suggested Improvements]:
# 1. Diagnosis: Large gap between train loss (~0.00) and test loss (>10.0) confirms high variance / overfitting.
# 2. Dropout: Randomly deactivates neurons during forward passes, mitigating co-adaptation.
# 3. L2 Weight Decay: Constrains weight magnitudes to prevent memorization of sample noise.
# 4. Architecture Simplification: Restricting parameter count matches model capacity to dataset size.
# 5. Early Stopping: Halt gradient updates once validation loss stops decreasing.

# ======================================================================
