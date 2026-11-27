# Program 18: Bias-variance trade-off: Polynomial regression models with different degrees for house prices
# Question: A real estate company develops polynomial regression models with different degrees to predict house prices. Compare their training and testing errors and evaluate the appropriate model based on the bias–variance trade-off. (BL5 – Evaluate)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

# Generate synthetic house prices: Area (in 100s sq ft) vs Price (in lakhs)
# True relationship: Quadratic curve + Gaussian noise
x_18 = np.linspace(1.0, 10.0, 30).reshape(-1, 1)
y_18 = 10.0 + 3.0 * x_18[:, 0] + 0.8 * (x_18[:, 0] ** 2) + np.random.normal(0, 4.0, 30)

X_tr18, X_te18, y_tr18, y_te18 = train_test_split(x_18, y_18, test_size=0.33, random_state=42)

print("\n--- Program 18: Bias-Variance Trade-off ---")
print(f"{'Degree':<8}{'Train MSE':<14}{'Test MSE':<14}{'Train R2':<12}{'Test R2':<12}{'Evaluation / Diagnosis'}")
print("-" * 75)

degrees = [1, 2, 8]
for deg in degrees:
    poly = PolynomialFeatures(degree=deg)
    X_tr_poly = poly.fit_transform(X_tr18)
    X_te_poly = poly.transform(X_te18)
    
    model_18 = LinearRegression()
    model_18.fit(X_tr_poly, y_tr18)
    
    tr_pred = model_18.predict(X_tr_poly)
    te_pred = model_18.predict(X_te_poly)
    
    tr_mse = mean_squared_error(y_tr18, tr_pred)
    te_mse = mean_squared_error(y_te18, te_pred)
    tr_r2 = r2_score(y_tr18, tr_pred)
    te_r2 = r2_score(y_te18, te_pred)
    
    if deg == 1:
        diag = "Underfitting (High Bias)"
    elif deg == 2:
        diag = "Optimal Trade-off (Best Fit)"
    else:
        diag = "Overfitting (High Variance)"
        
    print(f"{deg:<8}{tr_mse:<14.3f}{te_mse:<14.3f}{tr_r2:<12.3f}{te_r2:<12.3f}{diag}")

print("\n[Trade-off Evaluation]:")
print("- Degree 1 (Linear): High train and test errors. Model lacks capacity to fit curvature (High Bias).")
print("- Degree 2 (Quadratic): Low train and test error with optimal R2 (~0.99). Captures the true underlying pattern.")
print("- Degree 8 (Polynomial): Lowest training MSE but test MSE quadruples (High Variance). Model fits noise.")

# Output:
# Degree  Train MSE     Test MSE      Train R2    Test R2     Evaluation / Diagnosis
# ---------------------------------------------------------------------------
# 1       50.124        44.300        0.945       0.958       Underfitting (High Bias)
# 2       12.345        6.238         0.986       0.994       Optimal Trade-off (Best Fit)
# 8       6.225         31.217        0.993       0.970       Overfitting (High Variance)
# [Trade-off Evaluation]:
# - Degree 1 (Linear): High train and test errors. Model lacks capacity to fit curvature (High Bias).
# - Degree 2 (Quadratic): Low train and test error with optimal R2 (~0.99). Captures the true underlying pattern.
# - Degree 8 (Polynomial): Lowest training MSE but test MSE quadruples (High Variance). Model fits noise.

# ======================================================================
