# Program 28: Activation function comparison
# Question: Train models with ReLU, sigmoid, tanh for customer churn prediction, evaluate effects.
# ======================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Synthetic Customer Churn Dataset Generation
# ----------------------------------------------------------------------
# Features:
# 1. Tenure (months with service: 1 to 72)
# 2. Monthly Charges ($20 to $120)
# 3. Support Tickets (0 to 10 complaints)
# 4. Contract Type (0: Month-to-Month, 1: One-Year, 2: Two-Year)
# 5. Total Charges (approx Tenure * Monthly Charges)
# Target: Churn (1: Cancelled Service, 0: Retained)

np.random.seed(42)
torch.manual_seed(42)

n_customers = 1200
tenure = np.random.uniform(1, 72, (n_customers, 1))
monthly_charges = np.random.uniform(20, 120, (n_customers, 1))
support_tickets = np.random.poisson(2, (n_customers, 1)).astype(float)
contract_type = np.random.choice([0, 1, 2], size=(n_customers, 1)).astype(float)
total_charges = tenure * monthly_charges + np.random.normal(0, 20, (n_customers, 1))

# Churn logic: high monthly charges, low tenure, high support complaints increase churn
log_odds = (
    - 0.04 * tenure
    + 0.03 * monthly_charges
    + 0.45 * support_tickets
    - 0.85 * contract_type
    - 1.0
)
churn_prob = 1.0 / (1.0 + np.exp(-log_odds))
y_churn = (np.random.rand(n_customers, 1) < churn_prob).astype(np.float32)

X_churn = np.hstack([tenure, monthly_charges, support_tickets, contract_type, total_charges]).astype(np.float32)

# Normalization (StandardScaler)
mean_vals = X_churn.mean(axis=0)
std_vals = X_churn.std(axis=0) + 1e-8
X_churn_norm = (X_churn - mean_vals) / std_vals

# Split into Train (80% = 960) and Test (20% = 240)
split_bound = 960
X_train_c = torch.tensor(X_churn_norm[:split_bound], dtype=torch.float32)
y_train_c = torch.tensor(y_churn[:split_bound], dtype=torch.float32)
X_test_c = torch.tensor(X_churn_norm[split_bound:], dtype=torch.float32)
y_test_c = torch.tensor(y_churn[split_bound:], dtype=torch.float32)

print(f"Customer Churn Dataset: {n_customers} samples (Train: {len(X_train_c)}, Test: {len(X_test_c)})")
print(f"Churn Rate: {y_churn.mean()*100:.2f}%\n")

# ----------------------------------------------------------------------
# 2. Configurable Neural Network Architecture
# ----------------------------------------------------------------------
class ChurnPredictor(nn.Module):
    def __init__(self, activation_fn):
        super(ChurnPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32),
            activation_fn(),
            nn.Linear(32, 16),
            activation_fn(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Sigmoid on output layer for binary probability
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------------------------------------------------
# 3. Model Training & Comparison across ReLU, Sigmoid, and Tanh
# ----------------------------------------------------------------------
activation_candidates = {
    'ReLU': nn.ReLU,
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh
}

num_epochs = 60
training_histories = {}
evaluation_results = {}

for name, act_module in activation_candidates.items():
    # Use identical initialization seed for fair comparison
    torch.manual_seed(42)
    model = ChurnPredictor(act_module)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    loss_history = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_c)
        loss = criterion(outputs, y_train_c)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    training_histories[name] = loss_history

    # Evaluation on Test Set
    model.eval()
    with torch.no_grad():
        test_probs = model(X_test_c)
        test_loss = criterion(test_probs, y_test_c).item()
        test_preds = (test_probs >= 0.5).float()

        tp = ((test_preds == 1) & (y_test_c == 1)).sum().item()
        fp = ((test_preds == 1) & (y_test_c == 0)).sum().item()
        fn = ((test_preds == 0) & (y_test_c == 1)).sum().item()
        tn = ((test_preds == 0) & (y_test_c == 0)).sum().item()

        acc = (tp + tn) / len(y_test_c) * 100
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        evaluation_results[name] = {
            'final_train_loss': loss_history[-1],
            'test_loss': test_loss,
            'accuracy': acc,
            'precision': prec * 100,
            'recall': rec * 100,
            'f1': f1
        }

# ----------------------------------------------------------------------
# 4. Comparative Evaluation Report
# ----------------------------------------------------------------------
print("--- Customer Churn Performance Comparison ---")
print(f"{'Activation':10s} | {'Train Loss':>11s} | {'Test Loss':>10s} | {'Accuracy':>9s} | {'Precision':>10s} | {'Recall':>8s} | {'F1-Score':>9s}")
print("-" * 84)
for name, res in evaluation_results.items():
    print(f"{name:10s} | {res['final_train_loss']:11.4f} | {res['test_loss']:10.4f} | {res['accuracy']:8.2f}% | {res['precision']:9.2f}% | {res['recall']:7.2f}% | {res['f1']:9.4f}")

# ----------------------------------------------------------------------
# 5. Convergence Visualization
# ----------------------------------------------------------------------
plt.figure(figsize=(9, 5))
colors = {'ReLU': 'crimson', 'Sigmoid': 'navy', 'Tanh': 'forestgreen'}
for name, hist in training_histories.items():
    plt.plot(range(1, num_epochs + 1), hist, label=f"{name} (Final: {hist[-1]:.4f})", color=colors[name], lw=2)

plt.title('Training Loss Convergence Across Activation Functions', fontsize=12, fontweight='bold')
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Binary Cross-Entropy Loss', fontsize=10)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('churn_activation_comparison.png', dpi=150)
print("\nLoss convergence comparison saved as 'churn_activation_comparison.png'")
plt.close()

# ----------------------------------------------------------------------
# Output comments:
# Customer Churn Dataset: 1200 samples (Train: 960, Test: 240)
# Churn Rate: 46.67%
# 
# --- Customer Churn Performance Comparison ---
# Activation |  Train Loss |  Test Loss |  Accuracy |  Precision |   Recall |  F1-Score
# ------------------------------------------------------------------------------------
# ReLU       |      0.4919 |     0.5222 |    74.17% |     74.75% |   66.67% |    0.7048
# Sigmoid    |      0.5214 |     0.4978 |    76.67% |     77.78% |   69.37% |    0.7333
# Tanh       |      0.5078 |     0.5072 |    75.83% |     76.24% |   69.37% |    0.7264
# 
# Loss convergence comparison saved as 'churn_activation_comparison.png'
# ----------------------------------------------------------------------

# ======================================================================
