# Program 27: Activation functions
# Question: Implement sigmoid, tanh, ReLU, visualize outputs, analyse differences.
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Mathematical Implementation of Activation Functions & Derivatives
# ----------------------------------------------------------------------
# Sigmoid: sigma(x) = 1 / (1 + exp(-x))
# Derivative: sigma'(x) = sigma(x) * (1 - sigma(x))
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1.0 - s)

# Hyperbolic Tangent: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
# Derivative: tanh'(x) = 1 - tanh(x)^2
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2

# Rectified Linear Unit: ReLU(x) = max(0, x)
# Derivative: ReLU'(x) = 1 if x > 0 else 0
def relu(x):
    return np.maximum(0.0, x)

def relu_derivative(x):
    return np.where(x > 0.0, 1.0, 0.0)

# ----------------------------------------------------------------------
# 2. Numerical Analysis across Test Inputs
# ----------------------------------------------------------------------
sample_points = np.array([-4.0, -2.0, -0.5, 0.0, 0.5, 2.0, 4.0])

print("--- Activation Functions & Derivatives Evaluation ---")
header = f"{'x':>5} | {'Sigmoid':>8} {'Tanh':>8} {'ReLU':>8} | {'dSigmoid':>9} {'dTanh':>8} {'dReLU':>8}"
print(header)
print("-" * len(header))
for pt in sample_points:
    s_val = sigmoid(pt)
    t_val = tanh(pt)
    r_val = relu(pt)
    ds_val = sigmoid_derivative(pt)
    dt_val = tanh_derivative(pt)
    dr_val = relu_derivative(pt)
    print(f"{pt:5.1f} | {s_val:8.4f} {t_val:8.4f} {r_val:8.4f} | {ds_val:9.4f} {dt_val:8.4f} {dr_val:8.4f}")

# ----------------------------------------------------------------------
# 3. Comprehensive Comparative Analysis
# ----------------------------------------------------------------------
analysis_text = """
========================================================================================
                         THEORETICAL AND PRACTICAL ANALYSIS
========================================================================================
1. SIGMOID:
   - Output Range: (0, 1) -> Ideal for output layers of binary classifiers.
   - Zero-Centered: NO. All outputs are positive, causing 'zigzagging' gradient updates.
   - Vanishing Gradient: SEVERE. Maximum derivative is 0.25 (at x=0). For large |x|,
     the gradient approaches 0, stalling learning in deep networks.
   - Computation: Expensive due to exponential computation (exp(-x)).

2. TANH (Hyperbolic Tangent):
   - Output Range: (-1, 1).
   - Zero-Centered: YES. Helps mean-center hidden layer activations, speeding up convergence.
   - Vanishing Gradient: PRESENT. Maximum derivative is 1.00 (at x=0). Satures for |x| > 2,
     leading to vanishing gradients in deep architectures.
   - Computation: Involves exponentials.

3. ReLU (Rectified Linear Unit):
   - Output Range: [0, +infinity).
   - Zero-Centered: NO.
   - Vanishing Gradient: ELIMINATED for positive inputs (gradient is strictly 1.0).
     Accelerates convergence of Stochastic Gradient Descent (often 6x faster than Tanh).
   - Vulnerability: 'Dying ReLU' problem - neurons that output negative values receive 0
     gradient and can permanently cease updating if learning rate is too high.
   - Computation: Extremely cheap (simple thresholding at zero). Default for hidden layers.
========================================================================================
"""
print(analysis_text)

# ----------------------------------------------------------------------
# 4. Visualization of Functions and Derivatives
# ----------------------------------------------------------------------
x_domain = np.linspace(-6, 6, 500)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Activation Function Outputs
axes[0].plot(x_domain, sigmoid(x_domain), label=r'Sigmoid: $\sigma(x) = \frac{1}{1 + e^{-x}}$', color='navy', lw=2)
axes[0].plot(x_domain, tanh(x_domain), label=r'Tanh: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$', color='forestgreen', lw=2)
axes[0].plot(x_domain, relu(x_domain), label=r'ReLU: $\max(0, x)$', color='crimson', lw=2)
axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axes[0].axvline(0, color='gray', linestyle='--', linewidth=0.8)
axes[0].set_ylim(-1.5, 4.0)
axes[0].set_title('Activation Functions Output Comparison', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Input (x)', fontsize=10)
axes[0].set_ylabel('Activation f(x)', fontsize=10)
axes[0].legend(loc='upper left', fontsize=10)
axes[0].grid(True, linestyle=':', alpha=0.6)

# Subplot 2: First Derivatives (Gradient Flow)
axes[1].plot(x_domain, sigmoid_derivative(x_domain), label=r"Sigmoid' $(\max = 0.25)$", color='navy', lw=2)
axes[1].plot(x_domain, tanh_derivative(x_domain), label=r"Tanh' $(\max = 1.00)$", color='forestgreen', lw=2)
axes[1].plot(x_domain, relu_derivative(x_domain), label=r"ReLU' $(\text{constant } 1.0 \text{ for } x > 0)$", color='crimson', lw=2)
axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axes[1].axvline(0, color='gray', linestyle='--', linewidth=0.8)
axes[1].set_ylim(-0.2, 1.2)
axes[1].set_title('Activation Function Derivatives (Gradient Flow)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Input (x)', fontsize=10)
axes[1].set_ylabel("Derivative f'(x)", fontsize=10)
axes[1].legend(loc='upper right', fontsize=10)
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('activation_functions_comparison.png', dpi=150)
print("Plot successfully saved as 'activation_functions_comparison.png'")
plt.close()

# ----------------------------------------------------------------------
# Output comments:
# --- Activation Functions & Derivatives Evaluation ---
#     x |  Sigmoid     Tanh     ReLU |  dSigmoid    dTanh    dReLU
# -----------------------------------------------------------------
#  -4.0 |   0.0180  -0.9993   0.0000 |    0.0177   0.0013   0.0000
#  -2.0 |   0.1192  -0.9640   0.0000 |    0.1050   0.0707   0.0000
#  -0.5 |   0.3775  -0.4621   0.0000 |    0.2350   0.7864   0.0000
#   0.0 |   0.5000   0.0000   0.0000 |    0.2500   1.0000   0.0000
#   0.5 |   0.6225   0.4621   0.5000 |    0.2350   0.7864   1.0000
#   2.0 |   0.8808   0.9640   2.0000 |    0.1050   0.0707   1.0000
#   4.0 |   0.9820   0.9993   4.0000 |    0.0177   0.0013   1.0000
# 
# Plot successfully saved as 'activation_functions_comparison.png'
# ----------------------------------------------------------------------

# ======================================================================
