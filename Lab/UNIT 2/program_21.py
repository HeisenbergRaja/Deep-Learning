# Program 21: Floating-point precision: Demonstrate floating-point precision errors with different data types
# Question: A scientific research laboratory performs calculations involving extremely large and small numerical values. Demonstrate floating-point precision errors using different numerical data types and analyse their impact on computational accuracy. (BL4 – Analyse)

import numpy as np

print("\n--- Program 21: Floating-Point Precision Errors ---")

# 1. Machine Epsilon (smallest positive step epsilon such that 1.0 + epsilon > 1.0)
print("[1. Machine Epsilon (Precision Limits)]:")
for dtype in [np.float16, np.float32, np.float64]:
    eps = dtype(1.0)
    while dtype(1.0) + (eps / dtype(2.0)) > dtype(1.0):
        eps /= dtype(2.0)
    mantissa_bits = 10 if dtype == np.float16 else (23 if dtype == np.float32 else 52)
    print(f"  {dtype.__name__:<10}: Machine Epsilon = {float(eps):.10e} (Mantissa bits: {mantissa_bits})")

# 2. Large and Small Numerical Value Addition (Catastrophic Absorption)
print("\n[2. Swallowing / Absorption of Small Value in Large Number (1000.0 + 1000 * 0.001)]:")
for dtype in [np.float16, np.float32, np.float64]:
    total = dtype(1000.0)
    increment = dtype(0.001)
    for _ in range(1000):
        total += increment
    expected = 1001.0
    abs_err = abs(float(total) - expected)
    print(f"  {dtype.__name__:<10}: Result = {float(total):<15.8f} | Absolute Error = {abs_err:.8f}")

# 3. Numerical Overflow and Underflow Limits
print("\n[3. Numerical Overflow Limits (400.0 * 200.0 = 80,000)]:")
for dtype in [np.float16, np.float32]:
    max_val = np.finfo(dtype).max
    prod = dtype(400.0) * dtype(200.0)
    print(f"  {dtype.__name__:<10}: Max Representable = {max_val} | Result = {prod}")

print("\n[4. Computational Accuracy Analysis]:")
print("1. FP16 (Half Precision): 10-bit mantissa leads to severe precision loss; additions < 0.061 are swallowed at 1000.")
print("2. FP32 (Single Precision): Balances memory footprint and dynamic range; standard workhorse for DL activations.")
print("3. FP64 (Double Precision): Minimal rounding drift; essential for scientific computing and exact convergence.")
print("4. Deep Learning Implication: Mixed precision (FP16/BF16) requires loss scaling to prevent gradient underflow.")

# Output:
# [1. Machine Epsilon (Precision Limits)]:
#   float16   : Machine Epsilon = 9.7656250000e-04 (Mantissa bits: 10)
#   float32   : Machine Epsilon = 1.1920928955e-07 (Mantissa bits: 23)
#   float64   : Machine Epsilon = 2.2204460493e-16 (Mantissa bits: 52)
# [2. Swallowing / Absorption of Small Value in Large Number (1000.0 + 1000 * 0.001)]:
#   float16   : Result = 1000.00000000   | Absolute Error = 1.00000000
#   float32   : Result = 1000.97656250   | Absolute Error = 0.02343750
#   float64   : Result = 1001.00000000   | Absolute Error = 0.00000000
# [3. Numerical Overflow Limits (400.0 * 200.0 = 80,000)]:
#   float16   : Max Representable = 65504.0 | Result = inf
#   float32   : Max Representable = 3.4028234663852886e+38 | Result = 80000.0
# [4. Computational Accuracy Analysis]:
# 1. FP16 (Half Precision): 10-bit mantissa leads to severe precision loss; additions < 0.061 are swallowed at 1000.
# 2. FP32 (Single Precision): Balances memory footprint and dynamic range; standard workhorse for DL activations.
# 3. FP64 (Double Precision): Minimal rounding drift; essential for scientific computing and exact convergence.
# 4. Deep Learning Implication: Mixed precision (FP16/BF16) requires loss scaling to prevent gradient underflow.

# ======================================================================
