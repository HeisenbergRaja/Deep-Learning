# Program 19: L1 and L2 regularization: Regression model for loan prediction
# Question: A financial institution develops a regression model to predict loan amounts using multiple financial features. Implement L1 and L2 regularization and analyse their effects on model coefficients and prediction performance. (BL4 – Analyse)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

# Generate loan application dataset (120 applicants, 8 financial features)
n_samples_19 = 120
income = np.random.uniform(30, 150, n_samples_19)
credit_score = np.random.uniform(500, 850, n_samples_19)
savings = np.random.uniform(5, 50, n_samples_19)
dti = np.random.uniform(10, 50, n_samples_19)
exp_years = np.random.uniform(1, 20, n_samples_19)
num_cards = np.random.randint(1, 8, n_samples_19)
noise_feature = np.random.normal(0, 5, n_samples_19)       # Irrelevant noise
collinear_income = income * 1.5 + np.random.normal(0, 2, n_samples_19)  # Highly collinear

X_19 = np.column_stack([income, credit_score, savings, dti, exp_years, num_cards, noise_feature, collinear_income])
# Target loan amount
y_19 = 5.0 + 1.2 * income + 0.3 * credit_score + 0.8 * savings - 1.5 * dti + 0.5 * exp_years + np.random.normal(0, 15, n_samples_19)

feature_labels = ["Income", "CreditScore", "Savings", "DTI_Ratio", "ExpYears", "NumCards", "NoiseFeat", "CollinearInc"]

X_tr19, X_te19, y_tr19, y_te19 = train_test_split(X_19, y_19, test_size=0.25, random_state=42)

scaler19 = StandardScaler()
X_tr19_s = scaler19.fit_transform(X_tr19)
X_te19_s = scaler19.transform(X_te19)

# 1. Ordinary Least Squares (OLS)
ols_19 = LinearRegression().fit(X_tr19_s, y_tr19)
# 2. L1 Regularization (Lasso)
lasso_19 = Lasso(alpha=2.5, random_state=42).fit(X_tr19_s, y_tr19)
# 3. L2 Regularization (Ridge)
ridge_19 = Ridge(alpha=20.0, random_state=42).fit(X_tr19_s, y_tr19)

print("\n--- Program 19: L1 and L2 Regularization ---")
print(f"{'Feature':<15}{'OLS (No Reg)':<16}{'L1 (Lasso)':<16}{'L2 (Ridge)':<16}")
print("-" * 63)
for fn, c_o, c_l, c_r in zip(feature_labels, ols_19.coef_, lasso_19.coef_, ridge_19.coef_):
    print(f"{fn:<15}{c_o:<16.2f}{c_l:<16.2f}{c_r:<16.2f}")

print("-" * 63)
for name, model in [("OLS", ols_19), ("L1 Lasso", lasso_19), ("L2 Ridge", ridge_19)]:
    tr_mse = mean_squared_error(y_tr19, model.predict(X_tr19_s))
    te_mse = mean_squared_error(y_te19, model.predict(X_te19_s))
    te_r2 = r2_score(y_te19, model.predict(X_te19_s))
    print(f"{name:<10} | Train MSE: {tr_mse:8.2f} | Test MSE: {te_mse:8.2f} | Test R2: {te_r2:.4f}")

print("\n[Coefficient Effects Analysis]:")
print("1. L1 Regularization (Lasso): Enforces sparsity by driving coefficients of uninformative features (NoiseFeat, NumCards) to exactly 0.0.")
print("2. L2 Regularization (Ridge): Shrinks large coefficients smoothly without zeroing them, balancing weights across collinear features.")

# Output:
# Feature        OLS (No Reg)    L1 (Lasso)      L2 (Ridge)      
# ---------------------------------------------------------------
# Income         36.90           27.53           18.39           
# CreditScore    29.42           26.62           23.81           
# Savings        12.98           10.24           10.47           
# DTI_Ratio      -18.70          -16.60          -15.49          
# ExpYears       4.11            1.47            3.68            
# NumCards       -1.77           -0.00           -2.25           
# NoiseFeat      -0.40           0.00            0.94            
# CollinearInc   5.48            12.40           18.46           
# ---------------------------------------------------------------
# OLS        | Train MSE:   228.06 | Test MSE:   175.99 | Test R2: 0.9369
# L1 Lasso   | Train MSE:   263.10 | Test MSE:   201.39 | Test R2: 0.9278
# L2 Ridge   | Train MSE:   296.90 | Test MSE:   191.71 | Test R2: 0.9312
# [Coefficient Effects Analysis]:
# 1. L1 Regularization (Lasso): Enforces sparsity by driving coefficients of uninformative features (NoiseFeat, NumCards) to exactly 0.0.
# 2. L2 Regularization (Ridge): Shrinks large coefficients smoothly without zeroing them, balancing weights across collinear features.

# ======================================================================
