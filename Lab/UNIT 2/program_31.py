# Program 31: MSE Loss
# Question: Neural network for house price prediction, train with MSE loss, analyse actual vs predicted.
# ======================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Dataset Generation: Real Estate Valuation (Continuous Regression)
# ----------------------------------------------------------------------
# Features:
# 1. Square Footage (900 to 3800 sq ft)
# 2. Number of Bedrooms (1 to 5)
# 3. Number of Bathrooms (1 to 3)
# 4. Property Age (0 to 45 years)
# 5. Distance to City Center (1 to 25 km)
# 6. Neighborhood Crime Index (0.1 to 4.0)
# Target: House Sale Price (in $1,000s)

np.random.seed(42)
torch.manual_seed(42)

n_properties = 600
sqft = np.random.uniform(900, 3800, (n_properties, 1))
bedrooms = np.random.randint(1, 6, (n_properties, 1)).astype(float)
bathrooms = np.random.randint(1, 4, (n_properties, 1)).astype(float)
prop_age = np.random.uniform(0, 45, (n_properties, 1))
dist_center = np.random.uniform(1, 25, (n_properties, 1))
crime_idx = np.random.uniform(0.1, 4.0, (n_properties, 1))

# Continuous price function with Gaussian market variation
actual_prices = (
    40.0
    + 0.16 * sqft
    + 22.0 * bedrooms
    + 28.0 * bathrooms
    - 1.2 * prop_age
    - 3.5 * dist_center
    - 7.0 * crime_idx
    + np.random.normal(0, 12, (n_properties, 1))
).astype(np.float32)

X_house_raw = np.hstack([sqft, bedrooms, bathrooms, prop_age, dist_center, crime_idx]).astype(np.float32)

# Normalization (Crucial for gradient descent with MSE loss)
X_mean, X_std = X_house_raw.mean(axis=0), X_house_raw.std(axis=0) + 1e-8
y_mean, y_std = actual_prices.mean(), actual_prices.std() + 1e-8

X_house_norm = (X_house_raw - X_mean) / X_std
y_house_norm = (actual_prices - y_mean) / y_std

# Split: 80% Train (480), 20% Test (120)
split_idx = 480
X_train_h = torch.tensor(X_house_norm[:split_idx], dtype=torch.float32)
y_train_h = torch.tensor(y_house_norm[:split_idx], dtype=torch.float32)
X_test_h = torch.tensor(X_house_norm[split_idx:], dtype=torch.float32)
y_test_h = torch.tensor(y_house_norm[split_idx:], dtype=torch.float32)

# ----------------------------------------------------------------------
# 2. Regression Neural Network with MSE Loss
# ----------------------------------------------------------------------
class HousePriceRegressor(nn.Module):
    def __init__(self, in_features=6):
        super(HousePriceRegressor, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Linear output for continuous regression
        )

    def forward(self, x):
        return self.regressor(x)

model_house = HousePriceRegressor(in_features=6)

# MSE Loss formulation: MSE = (1/N) * sum((y_true - y_pred)^2)
criterion_mse = nn.MSELoss()
optimizer_house = optim.Adam(model_house.parameters(), lr=0.01)
epochs = 100

print(f"Housing Dataset: {n_properties} properties (Train: {len(X_train_h)}, Test: {len(X_test_h)})")
print(f"Average Actual Price: ${y_mean:.2f}k | Std: ${y_std:.2f}k\n")
print("--- Training Neural Network with MSE Loss ---")

# ----------------------------------------------------------------------
# 3. Model Training Loop
# ----------------------------------------------------------------------
train_losses = []
for epoch in range(1, epochs + 1):
    model_house.train()
    optimizer_house.zero_grad()
    preds = model_house(X_train_h)
    loss = criterion_mse(preds, y_train_h)
    loss.backward()
    optimizer_house.step()
    train_losses.append(loss.item())

    if epoch % 25 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{epochs} | MSE Loss (normalized): {loss.item():.6f}")

# ----------------------------------------------------------------------
# 4. Actual vs. Predicted Analysis on Unseen Test Houses
# ----------------------------------------------------------------------
model_house.eval()
with torch.no_grad():
    y_pred_norm_np = model_house(X_test_h).numpy()
    y_true_norm_np = y_test_h.numpy()

# Denormalize predictions and targets back to dollar scale ($k)
y_pred_dollar = y_pred_norm_np * y_std + y_mean
y_true_dollar = y_true_norm_np * y_std + y_mean

# Statistical Performance Metrics
mse_val = np.mean((y_true_dollar - y_pred_dollar) ** 2)
rmse_val = np.sqrt(mse_val)
mae_val = np.mean(np.abs(y_true_dollar - y_pred_dollar))
ss_tot = np.sum((y_true_dollar - np.mean(y_true_dollar)) ** 2)
ss_res = np.sum((y_true_dollar - y_pred_dollar) ** 2)
r2_score = 1.0 - (ss_res / ss_tot)

print("\n--- Regression Performance Metrics on Test Data ---")
print(f"Mean Squared Error (MSE):       {mse_val:10.2f} ($k)^2")
print(f"Root Mean Squared Error (RMSE): ${rmse_val:9.2f}k")
print(f"Mean Absolute Error (MAE):      ${mae_val:9.2f}k")
print(f"R-squared Score (R^2):           {r2_score:10.4f} (Explains {r2_score*100:.2f}% of variance)")

# Comparative Sample Analysis
print("\n--- Actual vs Predicted Price Comparison (First 10 Test Houses) ---")
print(f"{'House #':>7} | {'Actual Price':>13} | {'Predicted Price':>16} | {'Abs Error':>11} | {'% Error':>8}")
print("-" * 65)
for i in range(10):
    act = y_true_dollar[i, 0]
    pred = y_pred_dollar[i, 0]
    err = np.abs(act - pred)
    pct = (err / act) * 100
    print(f"#{i+1:6d} | ${act:11.2f}k | ${pred:14.2f}k | ${err:9.2f}k | {pct:7.2f}%")

# ----------------------------------------------------------------------
# 5. Scatter Plot: Actual vs Predicted
# ----------------------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_true_dollar, y_pred_dollar, color='royalblue', alpha=0.7, edgecolors='k', label='Test Houses')
min_val = min(y_true_dollar.min(), y_pred_dollar.min())
max_val = max(y_true_dollar.max(), y_pred_dollar.max())
plt.plot([min_val, max_val], [min_val, max_val], color='crimson', linestyle='--', lw=2, label='Ideal Fit (y = x)')
plt.title(f'Actual vs. Predicted House Prices (R^2 = {r2_score:.4f})', fontsize=12, fontweight='bold')
plt.xlabel('Actual Price ($1,000s)', fontsize=10)
plt.ylabel('Predicted Price ($1,000s)', fontsize=10)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('house_price_actual_vs_predicted.png', dpi=150)
print("\nActual vs Predicted plot saved as 'house_price_actual_vs_predicted.png'")
plt.close()

# ----------------------------------------------------------------------
# Output comments:
# Housing Dataset: 600 properties (Train: 480, Test: 120)
# Average Actual Price: $450.73k | Std: $147.04k
# 
# --- Training Neural Network with MSE Loss ---
# Epoch   1/100 | MSE Loss (normalized): 1.025406
# Epoch  25/100 | MSE Loss (normalized): 0.038215
# Epoch  50/100 | MSE Loss (normalized): 0.011928
# Epoch  75/100 | MSE Loss (normalized): 0.007704
# Epoch 100/100 | MSE Loss (normalized): 0.006646
# 
# --- Regression Performance Metrics on Test Data ---
# Mean Squared Error (MSE):           221.69 ($k)^2
# Root Mean Squared Error (RMSE): $    14.89k
# Mean Absolute Error (MAE):      $    11.94k
# R-squared Score (R^2):               0.9893 (Explains 98.93% of variance)
# 
# --- Actual vs Predicted Price Comparison (First 10 Test Houses) ---
# House # |  Actual Price |  Predicted Price |   Abs Error |  % Error
# -----------------------------------------------------------------
# #     1 | $     463.55k | $        467.32k | $     3.77k |    0.81%
# #     2 | $     507.06k | $        514.70k | $     7.65k |    1.51%
# #     3 | $     444.31k | $        438.01k | $     6.29k |    1.42%
# #     4 | $     535.81k | $        555.77k | $    19.96k |    3.72%
# #     5 | $     513.21k | $        506.63k | $     6.58k |    1.28%
# #     6 | $     706.07k | $        696.57k | $     9.50k |    1.35%
# #     7 | $     343.29k | $        328.06k | $    15.23k |    4.44%
# #     8 | $     336.90k | $        306.11k | $    30.79k |    9.14%
# #     9 | $     669.71k | $        657.65k | $    12.06k |    1.80%
# #    10 | $     632.92k | $        643.22k | $    10.29k |    1.63%
# 
# Actual vs Predicted plot saved as 'house_price_actual_vs_predicted.png'
# ----------------------------------------------------------------------

# ======================================================================
