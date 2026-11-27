# Program 16: Real Estate: Multivariate Linear Regression for House Price Prediction
# Question: Real estate: Linear regression model for house prices based on area, bedrooms, location features.

import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Realistic Real Estate Dataset (N = 100 Properties)
np.random.seed(42)
num_properties = 100

area_sqft = np.random.uniform(800, 3500, num_properties)       # Square footage
bedrooms = np.random.randint(1, 6, num_properties)             # Bedrooms (1-5)
dist_city_km = np.random.uniform(1.5, 30.0, num_properties)    # Distance to center (km)
location_quality = np.random.uniform(1.0, 10.0, num_properties)# Location score (1-10)

# True underlying pricing formula ($ in thousands):
# Base $150k + $0.18k/sqft + $28k/bedroom - $4.2k/km_distance + $22k/quality_pt + noise
noise = np.random.normal(0, 15.0, num_properties)
house_prices = 150.0 + 0.18 * area_sqft + 28.0 * bedrooms - 4.2 * dist_city_km + 22.0 * location_quality + noise

X_raw = np.column_stack([area_sqft, bedrooms, dist_city_km, location_quality])
y_raw = house_prices

# 2. Train-Test Split (80% Train, 20% Test)
train_size = int(0.8 * num_properties)
X_train_raw = X_raw[:train_size]
y_train = y_raw[:train_size]
X_test_raw = X_raw[train_size:]
y_test = y_raw[train_size:]

# 3. Z-Score Feature Normalization
mean_train = np.mean(X_train_raw, axis=0)
std_train = np.std(X_train_raw, axis=0)

X_train = (X_train_raw - mean_train) / std_train
X_test = (X_test_raw - mean_train) / std_train

# 4. Train Model via Gradient Descent
m, d = X_train.shape
weights_gd = np.zeros(d)
bias_gd = 0.0
learning_rate = 0.05
iterations = 1000
cost_history = []

for it in range(iterations):
    y_pred = X_train @ weights_gd + bias_gd
    err = y_pred - y_train
    grad_w = (1.0 / m) * (X_train.T @ err)
    grad_b = (1.0 / m) * np.sum(err)
    weights_gd -= learning_rate * grad_w
    bias_gd -= learning_rate * grad_b
    cost_history.append((1.0 / (2 * m)) * np.sum(err**2))

# 5. Closed-Form Solution (Normal Equation) for Analytical Verification
# theta = (X_b^T X_b)^(-1) X_b^T y
X_b = np.column_stack([np.ones(m), X_train])
theta_normal = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y_train
bias_ne = theta_normal[0]
weights_ne = theta_normal[1:]

# 6. Evaluation Metrics on Test Set
y_pred_test_gd = X_test @ weights_gd + bias_gd
y_pred_test_ne = X_test @ weights_ne + bias_ne

mse_gd = np.mean((y_test - y_pred_test_gd)**2)
rmse_gd = np.sqrt(mse_gd)
mae_gd = np.mean(np.abs(y_test - y_pred_test_gd))
r2_gd = 1.0 - (np.sum((y_test - y_pred_test_gd)**2) / np.sum((y_test - np.mean(y_test))**2))

mse_ne = np.mean((y_test - y_pred_test_ne)**2)
rmse_ne = np.sqrt(mse_ne)
r2_ne = 1.0 - (np.sum((y_test - y_pred_test_ne)**2) / np.sum((y_test - np.mean(y_test))**2))

print("=" * 55)
print("REAL ESTATE: MULTIVARIATE LINEAR REGRESSION FOR HOUSE PRICES")
print("=" * 55)
print(f"Dataset Size: {num_properties} properties (80 Train, 20 Test)")
print(f"Features: Area (sq ft), Bedrooms, Distance to City (km), Location Quality (1-10)\n")
print("FEATURE WEIGHTS COMPARISON (Gradient Descent vs. Normal Equation):")
feature_names = ['Area (sq ft)', 'Bedrooms', 'Distance (km)', 'Location Quality']
print(f"{'Feature':<20} | {'GD Normalized Weight':<22} | {'NE Normalized Weight'}")
print("-" * 55)
for feat, wg, wn in zip(feature_names, weights_gd, weights_ne):
    print(f"{feat:<20} | {wg:22.4f} | {wn:22.4f}")
print(f"{'Bias / Intercept':<20} | {bias_gd:22.4f} | {bias_ne:22.4f}")
print("-" * 55)
print("TEST SET EVALUATION METRICS:")
print(f"  Mean Squared Error (MSE)   : GD = {mse_gd:.2f} | Normal Eq = {mse_ne:.2f}")
print(f"  Root Mean Sq Error (RMSE)  : GD = ${rmse_gd:.2f}k | Normal Eq = ${rmse_ne:.2f}k")
print(f"  Mean Absolute Error (MAE)  : GD = ${mae_gd:.2f}k")
print(f"  R-squared (R2) Score       : GD = {r2_gd:.4f} ({r2_gd*100:.2f}%) | Normal Eq = {r2_ne:.4f}")
print("-" * 55)
print("PRICE PREDICTIONS FOR NEW HOUSES:")
new_properties = np.array([
    [2200, 3, 5.0, 8.5],    # Property 1: Suburban premium family home
    [1100, 2, 18.0, 4.0],   # Property 2: Commuter starter home
    [3200, 5, 2.5, 9.5]     # Property 3: Downtown luxury estate
])
new_norm = (new_properties - mean_train) / std_train
predicted_prices = new_norm @ weights_gd + bias_gd

for idx, (features_prop, price_pred) in enumerate(zip(new_properties, predicted_prices)):
    print(f"Property {idx+1}: {features_prop[0]:.0f} sqft, {features_prop[1]:.0f} beds, {features_prop[2]:.1f} km, Quality: {features_prop[3]:.1f}/10")
    print(f"            => Predicted Market Price: ${price_pred:.2f}k (${price_pred*1000:,.0f})")

# Visualizing Actual vs Predicted Prices on Test Set
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_test_gd, color='royalblue', edgecolor='k', s=60, alpha=0.8, label='Test Properties')
min_val = min(np.min(y_test), np.min(y_pred_test_gd))
max_val = max(np.max(y_test), np.max(y_pred_test_gd))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y = x)')
plt.title(f"Real Estate House Prices: Actual vs. Predicted (R² = {r2_gd:.4f})", fontweight='bold')
plt.xlabel("Actual Price ($ in thousands)")
plt.ylabel("Predicted Price ($ in thousands)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("p16_real_estate_regression.png", dpi=150)
# plt.show()
plt.close()
print("\nVisualization saved as 'p16_real_estate_regression.png'.")

# Output:
# =======================================================
# REAL ESTATE: MULTIVARIATE LINEAR REGRESSION FOR HOUSE PRICES
# =======================================================
# Dataset Size: 100 properties (80 Train, 20 Test)
# Features: Area (sq ft), Bedrooms, Distance to City (km), Location Quality (1-10)
# 
# FEATURE WEIGHTS COMPARISON (Gradient Descent vs. Normal Equation):
# Feature              | GD Normalized Weight   | NE Normalized Weight
# -------------------------------------------------------
# Area (sq ft)         |               149.2779 |               149.2779
# Bedrooms             |                41.7873 |                41.7873
# Distance (km)        |               -32.8481 |               -32.8481
# Location Quality     |                60.8058 |                60.8058
# Bias / Intercept     |               646.0828 |               646.0828
# -------------------------------------------------------
# TEST SET EVALUATION METRICS:
#   Mean Squared Error (MSE)   : GD = 199.58 | Normal Eq = 199.58
#   Root Mean Sq Error (RMSE)  : GD = $14.13k | Normal Eq = $14.13k
#   Mean Absolute Error (MAE)  : GD = $11.85k
#   R-squared (R2) Score       : GD = 0.9885 (98.85%) | Normal Eq = 0.9885
# -------------------------------------------------------
# PRICE PREDICTIONS FOR NEW HOUSES:
# Property 1: 2200 sqft, 3 beds, 5.0 km, Quality: 8.5/10
#             => Predicted Market Price: $793.56k ($793,562)
# Property 2: 1100 sqft, 2 beds, 18.0 km, Quality: 4.0/10
#             => Predicted Market Price: $412.96k ($412,956)
# Property 3: 3200 sqft, 5 beds, 2.5 km, Quality: 9.5/10
#             => Predicted Market Price: $1065.22k ($1,065,220)
# 
# Visualization saved as 'p16_real_estate_regression.png'.



# ======================================================================
# CO2 PROGRAMS 17 TO 24
# ======================================================================

# -*- coding: utf-8 -*-
"""
Deep Learning Lab Experiments (Programs 17 to 24)
Self-contained, runnable implementations using NumPy, PyTorch, Scikit-Learn, and Matplotlib.
"""
import sys
import warnings
from unittest.mock import MagicMock

# Environment compatibility check
warnings.filterwarnings("ignore")
try:
    import sklearn.preprocessing._csr_polynomial_expansion
except Exception:
    sys.modules['sklearn.preprocessing._csr_polynomial_expansion'] = MagicMock()

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# ======================================================================
