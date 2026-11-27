# Program 15: Machine Learning Optimization: Batch, Stochastic, and Mini-Batch Gradient Descent
# Question: ML optimization: Implement batch, stochastic, and mini-batch gradient descent, compare convergence.

import numpy as np
import matplotlib.pyplot as plt
import time

# 1. Generate Synthetic Regression Dataset
np.random.seed(42)
N = 600   # 600 training samples
d = 2     # 2 input features
X = np.random.randn(N, d)
true_weights = np.array([3.5, -2.0])
true_bias = 5.0
noise = np.random.randn(N) * 0.5
y = X @ true_weights + true_bias + noise

def compute_mse(X_data, y_data, w, b):
    predictions = X_data @ w + b
    return np.mean((predictions - y_data)**2)

epochs = 40

# --- Method 1: Batch Gradient Descent (BGD) ---
# Uses all N=600 samples for every parameter update (1 update per epoch)
w_bgd = np.zeros(d)
b_bgd = 0.0
lr_bgd = 0.1
losses_bgd = []
trajectory_bgd = [(w_bgd[0], w_bgd[1])]
t0 = time.time()

for epoch in range(epochs):
    preds = X @ w_bgd + b_bgd
    error = preds - y
    grad_w = (1.0 / N) * (X.T @ error)
    grad_b = (1.0 / N) * np.sum(error)
    w_bgd -= lr_bgd * grad_w
    b_bgd -= lr_bgd * grad_b
    losses_bgd.append(compute_mse(X, y, w_bgd, b_bgd))
    trajectory_bgd.append((w_bgd[0], w_bgd[1]))
time_bgd = (time.time() - t0) * 1000

# --- Method 2: Stochastic Gradient Descent (SGD) ---
# Updates parameters on every single sample (600 updates per epoch)
w_sgd = np.zeros(d)
b_sgd = 0.0
lr_sgd = 0.01
losses_sgd = []
trajectory_sgd = [(w_sgd[0], w_sgd[1])]
t0 = time.time()

for epoch in range(epochs):
    indices = np.random.permutation(N)
    for idx in indices:
        xi = X[idx:idx+1]
        yi = y[idx:idx+1]
        err = (xi @ w_sgd + b_sgd) - yi
        w_sgd -= lr_sgd * (xi.T @ err).flatten()
        b_sgd -= lr_sgd * np.sum(err)
    losses_sgd.append(compute_mse(X, y, w_sgd, b_sgd))
    trajectory_sgd.append((w_sgd[0], w_sgd[1]))
time_sgd = (time.time() - t0) * 1000

# --- Method 3: Mini-Batch Gradient Descent (MBGD) ---
# Updates parameters in mini-batches of size B=32 (~19 updates per epoch)
w_mbgd = np.zeros(d)
b_mbgd = 0.0
lr_mbgd = 0.05
batch_size = 32
losses_mbgd = []
trajectory_mbgd = [(w_mbgd[0], w_mbgd[1])]
t0 = time.time()

for epoch in range(epochs):
    indices = np.random.permutation(N)
    for i in range(0, N, batch_size):
        batch_idx = indices[i:i+batch_size]
        xb = X[batch_idx]
        yb = y[batch_idx]
        m = len(yb)
        err = (xb @ w_mbgd + b_mbgd) - yb
        w_mbgd -= lr_mbgd * (1.0 / m) * (xb.T @ err)
        b_mbgd -= lr_mbgd * (1.0 / m) * np.sum(err)
    losses_mbgd.append(compute_mse(X, y, w_mbgd, b_mbgd))
    trajectory_mbgd.append((w_mbgd[0], w_mbgd[1]))
time_mbgd = (time.time() - t0) * 1000

print("=" * 55)
print("ML OPTIMIZATION: BATCH vs SGD vs MINI-BATCH GRADIENT DESCENT")
print("=" * 55)
print(f"Dataset Size: N = {N} samples, 2 features | Target w = {true_weights}, b = {true_bias}")
print(f"Training Epochs: {epochs} | Mini-Batch Size: {batch_size}")
print("-" * 55)
print(f"{'Epoch':<6} | {'Batch GD Loss':<16} | {'SGD Loss':<16} | {'Mini-Batch Loss'}")
print("-" * 55)
sample_epochs = [1, 2, 5, 10, 20, 40]
for ep in sample_epochs:
    print(f" {ep:4d}  | {losses_bgd[ep-1]:16.5f} | {losses_sgd[ep-1]:16.5f} | {losses_mbgd[ep-1]:16.5f}")
print("-" * 55)
print("ALGORITHM PERFORMANCE SUMMARY:")
print(f"Batch GD     : Final Loss = {losses_bgd[-1]:.5f} | Time = {time_bgd:6.2f} ms | Weights = [{w_bgd[0]:.3f}, {w_bgd[1]:.3f}], b = {b_bgd:.3f}")
print(f"SGD          : Final Loss = {losses_sgd[-1]:.5f} | Time = {time_sgd:6.2f} ms | Weights = [{w_sgd[0]:.3f}, {w_sgd[1]:.3f}], b = {b_sgd:.3f}")
print(f"Mini-Batch GD: Final Loss = {losses_mbgd[-1]:.5f} | Time = {time_mbgd:6.2f} ms | Weights = [{w_mbgd[0]:.3f}, {w_mbgd[1]:.3f}], b = {b_mbgd:.3f}")
print("-" * 55)
print("COMPARATIVE INSIGHTS:")
print("1. Batch GD computes exact gradients: smooth descent trajectory but slowest per-epoch convergence.")
print("2. SGD makes rapid initial leaps by updating every sample, but fluctuates around the optimum.")
print("3. Mini-Batch GD combines vectorization hardware speed with robust convergence, representing the industry standard.")

# Visualizing Convergence Curves
plt.figure(figsize=(8, 4.5))
plt.plot(range(1, epochs + 1), losses_bgd, 'b-o', markersize=3, label='Batch GD (BGD)')
plt.plot(range(1, epochs + 1), losses_sgd, 'r-s', markersize=3, label='Stochastic GD (SGD)')
plt.plot(range(1, epochs + 1), losses_mbgd, 'g-^', markersize=3, label=f'Mini-Batch GD (B={batch_size})')
plt.yscale('log')
plt.title("Convergence Comparison: BGD vs. SGD vs. Mini-Batch GD", fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error Loss (Log Scale)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("p15_gd_comparison.png", dpi=150)
# plt.show()
plt.close()
print("Visualization saved as 'p15_gd_comparison.png'.")

# Output:
# =======================================================
# ML OPTIMIZATION: BATCH vs SGD vs MINI-BATCH GRADIENT DESCENT
# =======================================================
# Dataset Size: N = 600 samples, 2 features | Target w = [ 3.5 -2. ], b = 5.0
# Training Epochs: 40 | Mini-Batch Size: 32
# -------------------------------------------------------
# Epoch  | Batch GD Loss    | SGD Loss         | Mini-Batch Loss
# -------------------------------------------------------
#     1  |         34.63494 |          0.24760 |          6.13960
#     2  |         27.96662 |          0.24578 |          1.04275
#     5  |         14.76542 |          0.24476 |          0.24662
#    10  |          5.18690 |          0.24635 |          0.24419
#    20  |          0.81684 |          0.24666 |          0.24411
#    40  |          0.25180 |          0.24475 |          0.24413
# -------------------------------------------------------
# ALGORITHM PERFORMANCE SUMMARY:
# Batch GD     : Final Loss = 0.25180 | Time =  16.08 ms | Weights = [3.484, -1.965], b = 4.965
# SGD          : Final Loss = 0.24475 | Time = 184.48 ms | Weights = [3.537, -2.002], b = 5.055
# Mini-Batch GD: Final Loss = 0.24413 | Time =   0.00 ms | Weights = [3.535, -1.992], b = 5.030
# -------------------------------------------------------
# COMPARATIVE INSIGHTS:
# 1. Batch GD computes exact gradients: smooth descent trajectory but slowest per-epoch convergence.
# 2. SGD makes rapid initial leaps by updating every sample, but fluctuates around the optimum.
# 3. Mini-Batch GD combines vectorization hardware speed with robust convergence, representing the industry standard.
# Visualization saved as 'p15_gd_comparison.png'.

# ======================================================================
