# Program 35: Dropout Regularization for Student Academic Performance Prediction
# Question: A university develops a deep neural network to predict student academic performance but observes overfitting. Apply dropout regularization and evaluate the model's performance before and after applying dropout. (BL5 – Evaluate)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Synthetic Student Academic Performance Dataset
# 10 features: attendance, internal marks, study hours/week, assignment scores,
# quiz performance, lab assessment, online engagement, extracurriculars, sleep hours, stress index.
# Limited sample size (150 training samples) with high feature complexity to induce overfitting.
num_samples = 350
num_features = 10

X = np.random.randn(num_samples, num_features).astype(np.float32)
# Academic performance level: 0 = Low, 1 = Medium, 2 = High
# Academic signal depends on primary features with added noise
academic_score = (
    1.6 * X[:, 0]
    + 1.4 * X[:, 1]
    + 1.2 * X[:, 2]
    - 0.8 * X[:, 3]
    + 0.5 * np.random.randn(num_samples)
)

y = np.zeros(num_samples, dtype=np.int64)
y[academic_score > 0.8] = 2
y[(academic_score >= -0.8) & (academic_score <= 0.8)] = 1

# Train (150) / Test (200) split
X_train = torch.tensor(X[:150])
y_train = torch.tensor(y[:150])
X_test = torch.tensor(X[150:])
y_test = torch.tensor(y[150:])

# 1. Deep Neural Network WITHOUT Dropout (susceptible to overfitting)
class DeepNNWithoutDropout(nn.Module):
    def __init__(self, in_features=10, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 2. Deep Neural Network WITH Dropout Regularization
class DeepNNWithDropout(nn.Module):
    def __init__(self, in_features=10, num_classes=3, p=0.45):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p * 0.8),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p * 0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

criterion = nn.CrossEntropyLoss()
epochs = 80

# Train model WITHOUT Dropout
torch.manual_seed(42)
model_no_drop = DeepNNWithoutDropout()
optimizer_no_drop = optim.Adam(model_no_drop.parameters(), lr=0.008)

print("=== Training Deep NN WITHOUT Dropout (Baseline) ===")
for epoch in range(1, epochs + 1):
    model_no_drop.train()
    optimizer_no_drop.zero_grad()
    loss = criterion(model_no_drop(X_train), y_train)
    loss.backward()
    optimizer_no_drop.step()

model_no_drop.eval()
with torch.no_grad():
    train_acc_nodrop = (model_no_drop(X_train).argmax(1) == y_train).float().mean().item() * 100
    test_acc_nodrop = (model_no_drop(X_test).argmax(1) == y_test).float().mean().item() * 100
    test_loss_nodrop = criterion(model_no_drop(X_test), y_test).item()

# Train model WITH Dropout Regularization
torch.manual_seed(42)
model_with_drop = DeepNNWithDropout(p=0.45)
optimizer_with_drop = optim.Adam(model_with_drop.parameters(), lr=0.008)

print("=== Training Deep NN WITH Dropout Regularization ===")
for epoch in range(1, epochs + 1):
    model_with_drop.train()
    optimizer_with_drop.zero_grad()
    loss = criterion(model_with_drop(X_train), y_train)
    loss.backward()
    optimizer_with_drop.step()

model_with_drop.eval()
with torch.no_grad():
    train_acc_drop = (model_with_drop(X_train).argmax(1) == y_train).float().mean().item() * 100
    test_acc_drop = (model_with_drop(X_test).argmax(1) == y_test).float().mean().item() * 100
    test_loss_drop = criterion(model_with_drop(X_test), y_test).item()

# Comparative Evaluation
gap_nodrop = train_acc_nodrop - test_acc_nodrop
gap_drop = train_acc_drop - test_acc_drop

print("\n=== Performance Comparison Before and After Dropout ===")
print(f"{'Metric':<25} | {'Without Dropout':<18} | {'With Dropout':<18}")
print("-" * 68)
print(f"{'Training Accuracy':<25} | {train_acc_nodrop:<17.2f}% | {train_acc_drop:<17.2f}%")
print(f"{'Testing Accuracy':<25} | {test_acc_nodrop:<17.2f}% | {test_acc_drop:<17.2f}%")
print(f"{'Generalization Gap':<25} | {gap_nodrop:<17.2f}% | {gap_drop:<17.2f}%")
print(f"{'Testing Loss':<25} | {test_loss_nodrop:<18.4f} | {test_loss_drop:<18.4f}")

print("\nEvaluation & Analysis:")
print(f"1. Without Dropout: High training accuracy ({train_acc_nodrop:.2f}%) but lower testing accuracy ({test_acc_nodrop:.2f}%)")
print(f"   leaves a large generalization gap ({gap_nodrop:.2f}%), indicating severe memorization/overfitting.")
print(f"2. With Dropout: Randomly deactivating neurons during training prevents co-adaptation of features,")
print(f"   reducing the generalization gap to {gap_drop:.2f}% and improving model generalization on unseen student data.")

# Output:
# === Training Deep NN WITHOUT Dropout (Baseline) ===
# === Training Deep NN WITH Dropout Regularization ===
# 
# === Performance Comparison Before and After Dropout ===
# Metric                    | Without Dropout    | With Dropout      
# --------------------------------------------------------------------
# Training Accuracy         | 100.00           % | 100.00           %
# Testing Accuracy          | 79.50            % | 80.50            %
# Generalization Gap        | 20.50            % | 19.50            %
# Testing Loss              | 2.1918             | 0.7125            
# 
# Evaluation & Analysis:
# 1. Without Dropout: High training accuracy (100.00%) but lower testing accuracy (79.50%)
#    leaves a large generalization gap (20.50%), indicating severe memorization/overfitting.
# 2. With Dropout: Randomly deactivating neurons during training prevents co-adaptation of features,
#    reducing the generalization gap to 19.50% and improving model generalization on unseen student data.

# ======================================================================
