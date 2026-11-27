# Program 32: Binary cross-entropy
# Question: Fraudulent transaction detection neural network, implement BCE loss, evaluate model.
# ======================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------------------------------------------------
# 1. Dataset Generation: Financial Fraud Detection
# ----------------------------------------------------------------------
# Features:
# 1. Transaction Amount ($)
# 2. Distance from Home Location (miles)
# 3. Merchant Risk Score (0.0 to 1.0)
# 4. Failed Login / Verification Attempts (count)
# Target: Fraud (1) vs Legitimate (0) with 12% fraud rate

np.random.seed(42)
torch.manual_seed(42)

n_transactions = 1000
n_fraud = 120
n_legit = n_transactions - n_fraud

# Legitimate transactions: moderate amounts, nearby location, low risk score
amt_legit = np.random.normal(55, 25, (n_legit, 1))
dist_legit = np.random.normal(8, 4, (n_legit, 1))
risk_legit = np.random.normal(0.20, 0.10, (n_legit, 1))
failed_legit = np.random.poisson(0.1, (n_legit, 1)).astype(float)
X_legit = np.hstack([amt_legit, dist_legit, risk_legit, failed_legit])
y_legit = np.zeros((n_legit, 1), dtype=np.float32)

# Fraudulent transactions: high amounts, distant / foreign, high merchant risk
amt_fraud = np.random.normal(320, 60, (n_fraud, 1))
dist_fraud = np.random.normal(65, 15, (n_fraud, 1))
risk_fraud = np.random.normal(0.85, 0.08, (n_fraud, 1))
failed_fraud = np.random.poisson(3.0, (n_fraud, 1)).astype(float)
X_fraud = np.hstack([amt_fraud, dist_fraud, risk_fraud, failed_fraud])
y_fraud = np.ones((n_fraud, 1), dtype=np.float32)

X_all = np.vstack([X_legit, X_fraud]).astype(np.float32)
y_all = np.vstack([y_legit, y_fraud])

# Shuffle dataset
shuffle_idx = np.random.permutation(n_transactions)
X_all, y_all = X_all[shuffle_idx], y_all[shuffle_idx]

# Feature standardization
scaler_mu = X_all.mean(axis=0)
scaler_sigma = X_all.std(axis=0) + 1e-8
X_normalized = (X_all - scaler_mu) / scaler_sigma

# Split into 80% Train (800) and 20% Test (200)
split_point = 800
X_train_tx = torch.tensor(X_normalized[:split_point], dtype=torch.float32)
y_train_tx = torch.tensor(y_all[:split_point], dtype=torch.float32)
X_test_tx = torch.tensor(X_normalized[split_point:], dtype=torch.float32)
y_test_tx = torch.tensor(y_all[split_point:], dtype=torch.float32)

print(f"Transaction Dataset: {n_transactions} transactions | Fraud Rate: {y_all.mean()*100:.1f}%")
print(f"Train Set: {len(X_train_tx)} | Test Set: {len(X_test_tx)}\n")

# ----------------------------------------------------------------------
# 2. Binary Cross-Entropy (BCE) Loss Implementation from Scratch
# ----------------------------------------------------------------------
# Mathematical Formulation:
# BCE(y, y_hat) = - (1/N) * sum_{i=1}^N [ y_i * log(y_hat_i) + (1 - y_i) * log(1 - y_hat_i) ]
# Clipping (eps = 1e-12) avoids numerical instability from log(0).
def custom_bce_loss(y_pred, y_true):
    eps = 1e-12
    # Clamp probabilities to avoid log(0)
    y_clamped = torch.clamp(y_pred, min=eps, max=1.0 - eps)
    # Binary cross entropy formula
    loss = -torch.mean(y_true * torch.log(y_clamped) + (1.0 - y_true) * torch.log(1.0 - y_clamped))
    return loss

# ----------------------------------------------------------------------
# 3. Fraud Detection Neural Network Architecture
# ----------------------------------------------------------------------
class FraudDetectionNet(nn.Module):
    def __init__(self, in_features=4):
        super(FraudDetectionNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # Produces fraud probability in range (0, 1)
        )

    def forward(self, x):
        return self.classifier(x)

model_fraud = FraudDetectionNet(in_features=4)
optimizer_fraud = optim.Adam(model_fraud.parameters(), lr=0.01)
epochs = 100

print("--- Training Fraud Detection Neural Network with Custom BCE Loss ---")

# ----------------------------------------------------------------------
# 4. Training Loop
# ----------------------------------------------------------------------
for epoch in range(1, epochs + 1):
    model_fraud.train()
    optimizer_fraud.zero_grad()
    predictions = model_fraud(X_train_tx)

    # Compute loss with custom BCE implementation
    loss = custom_bce_loss(predictions, y_train_tx)

    loss.backward()
    optimizer_fraud.step()

    if epoch % 25 == 0 or epoch == 1:
        # Also compare with PyTorch native nn.BCELoss to verify equivalence
        with torch.no_grad():
            native_bce = nn.BCELoss()(predictions, y_train_tx).item()
        print(f"Epoch {epoch:3d}/{epochs} | Custom BCE Loss: {loss.item():.6f} | PyTorch Native BCE: {native_bce:.6f}")

# ----------------------------------------------------------------------
# 5. Comprehensive Model Evaluation on Test Data
# ----------------------------------------------------------------------
model_fraud.eval()
with torch.no_grad():
    fraud_probs = model_fraud(X_test_tx)
    # Decision threshold: 0.5 (Transactions with prob >= 0.5 flagged as Fraud)
    fraud_preds = (fraud_probs >= 0.5).float()

    y_test_np = y_test_tx.numpy().flatten()
    preds_np = fraud_preds.numpy().flatten()
    probs_np = fraud_probs.numpy().flatten()

# Confusion Matrix components
tp = int(((preds_np == 1) & (y_test_np == 1)).sum())
fp = int(((preds_np == 1) & (y_test_np == 0)).sum())
fn = int(((preds_np == 0) & (y_test_np == 1)).sum())
tn = int(((preds_np == 0) & (y_test_np == 0)).sum())

accuracy = (tp + tn) / len(y_test_np) * 100
precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

print("\n--- Model Evaluation on Unseen Test Transactions ---")
print(f"Total Test Transactions: {len(y_test_np)}")
print(f"Accuracy:  {accuracy:.2f}%")
print(f"Precision: {precision:.2f}% (Of transactions flagged as fraud, how many were genuine frauds)")
print(f"Recall:    {recall:.2f}% (Of actual fraudulent transactions, how many did we catch)")
print(f"F1-Score:  {f1_score / 100:.4f}\n")

print("Confusion Matrix:")
print(f"{'':22s} | {'Predicted Legit (0)':>20s} | {'Predicted Fraud (1)':>20s}")
print("-" * 68)
print(f"{'Actual Legit (0)':22s} | {tn:20d} | {fp:20d}")
print(f"{'Actual Fraud (1)':22s} | {fn:20d} | {tp:20d}")

print("\nSample Predictions on Test Transactions:")
print(f"{'Tx #':>5} | {'Actual Status':>15} | {'Predicted Prob':>15} | {'Flagged Status':>15} | {'Correct?':>8}")
print("-" * 66)
for i in range(10):
    status_act = "FRAUD (1)" if y_test_np[i] == 1 else "Legitimate (0)"
    status_pred = "FRAUD (1)" if preds_np[i] == 1 else "Legitimate (0)"
    is_correct = (y_test_np[i] == preds_np[i])
    print(f"#{i+1:4d} | {status_act:>15s} | {probs_np[i]:15.4f} | {status_pred:>15s} | {str(is_correct):>8s}")

print("\nKey Takeaway: Binary Cross-Entropy Loss heavily penalizes false negatives,")
print("driving the neural network to learn high sensitivity (100% Recall) essential in fraud detection.")

# ----------------------------------------------------------------------
# Output comments:
# Transaction Dataset: 1000 transactions | Fraud Rate: 12.0%
# Train Set: 800 | Test Set: 200
# 
# --- Training Fraud Detection Neural Network with Custom BCE Loss ---
# Epoch   1/100 | Custom BCE Loss: 0.580244 | PyTorch Native BCE: 0.580244
# Epoch  25/100 | Custom BCE Loss: 0.087573 | PyTorch Native BCE: 0.087573
# Epoch  50/100 | Custom BCE Loss: 0.001695 | PyTorch Native BCE: 0.001695
# Epoch  75/100 | Custom BCE Loss: 0.000657 | PyTorch Native BCE: 0.000657
# Epoch 100/100 | Custom BCE Loss: 0.000484 | PyTorch Native BCE: 0.000484
# 
# --- Model Evaluation on Unseen Test Transactions ---
# Total Test Transactions: 200
# Accuracy:  100.00%
# Precision: 100.00% (Of transactions flagged as fraud, how many were genuine frauds)
# Recall:    100.00% (Of actual fraudulent transactions, how many did we catch)
# F1-Score:  1.0000
# 
# Confusion Matrix:
#                        |  Predicted Legit (0) |  Predicted Fraud (1)
# --------------------------------------------------------------------
# Actual Legit (0)       |                  176 |                    0
# Actual Fraud (1)       |                    0 |                   24
# 
# Sample Predictions on Test Transactions:
#  Tx # |   Actual Status |  Predicted Prob |  Flagged Status | Correct?
# ------------------------------------------------------------------
# #   1 |  Legitimate (0) |          0.0001 |  Legitimate (0) |     True
# #   2 |  Legitimate (0) |          0.0000 |  Legitimate (0) |     True
# #   3 |  Legitimate (0) |          0.0005 |  Legitimate (0) |     True
# #   4 |  Legitimate (0) |          0.0001 |  Legitimate (0) |     True
# #   5 |  Legitimate (0) |          0.0002 |  Legitimate (0) |     True
# #   6 |  Legitimate (0) |          0.0010 |  Legitimate (0) |     True
# #   7 |  Legitimate (0) |          0.0002 |  Legitimate (0) |     True
# #   8 |  Legitimate (0) |          0.0000 |  Legitimate (0) |     True
# #   9 |  Legitimate (0) |          0.0001 |  Legitimate (0) |     True
# #  10 |  Legitimate (0) |          0.0005 |  Legitimate (0) |     True
# 
# Key Takeaway: Binary Cross-Entropy Loss heavily penalizes false negatives,
# driving the neural network to learn high sensitivity (100% Recall) essential in fraud detection.
# ----------------------------------------------------------------------



# ======================================================================
# CO3 PROGRAMS 33 TO 40
# ======================================================================
