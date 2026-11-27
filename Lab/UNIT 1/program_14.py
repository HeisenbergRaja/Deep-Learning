# Program 14: Manufacturing: Gradient Descent Optimization for Production Cost Minimization
# Question: Manufacturing: Implement gradient descent to minimize production costs (quadratic loss function).

import numpy as np
import matplotlib.pyplot as plt

def production_cost(x1, x2):
    """
    Quadratic production cost function:
    C(x1, x2) = 2*x1^2 + 3*x2^2 - 2*x1*x2 - 20*x1 - 30*x2 + 250
    where x1: Raw material feed rate (kg/hr), x2: Operating speed (RPM/100)
    """
    return 2.0 * x1**2 + 3.0 * x2**2 - 2.0 * x1 * x2 - 20.0 * x1 - 30.0 * x2 + 250.0

def cost_gradient(x1, x2):
    """
    Computes analytical partial derivatives:
    dC/dx1 = 4*x1 - 2*x2 - 20
    dC/dx2 = -2*x1 + 6*x2 - 30
    """
    d_x1 = 4.0 * x1 - 2.0 * x2 - 20.0
    d_x2 = -2.0 * x1 + 6.0 * x2 - 30.0
    return np.array([d_x1, d_x2])

# Analytical Solution:
# 4*x1 - 2*x2 = 20  => 2*x1 - x2 = 10 => x2 = 2*x1 - 10
# -2*x1 + 6*(2*x1 - 10) = 30 => 10*x1 = 90 => x1* = 9.0, x2* = 8.0
x1_analytical = 9.0
x2_analytical = 8.0
cost_analytical = production_cost(x1_analytical, x2_analytical)

# Gradient Descent Implementation
x = np.array([0.0, 0.0])   # Initial operating condition (x1=0, x2=0)
learning_rate = 0.1        # Step size
tolerance = 1e-6           # Convergence stopping threshold
max_iterations = 100

trajectory = [(x[0], x[1], production_cost(x[0], x[1]))]

print("=" * 55)
print("MANUFACTURING: PRODUCTION COST MINIMIZATION VIA GRADIENT DESCENT")
print("=" * 55)
print("Cost Function: C(x1, x2) = 2*x1^2 + 3*x2^2 - 2*x1*x2 - 20*x1 - 30*x2 + 250")
print(f"Initial State       : x1 = {x[0]:.2f}, x2 = {x[1]:.2f}, Cost = ${production_cost(x[0], x[1]):.2f}")
print(f"Analytical Optimum  : x1* = {x1_analytical:.2f}, x2* = {x2_analytical:.2f}, Min Cost = ${cost_analytical:.2f}")
print(f"Hyperparameters     : Learning Rate = {learning_rate}, Tolerance = {tolerance}\n")
print(f"{'Iter':<5} | {'x1 (Feed)':<10} | {'x2 (Speed)':<10} | {'Cost ($)':<12} | {'||grad||':<12} | {'Step Reduction'}")
print("-" * 55)

converged_iter = max_iterations
for it in range(1, max_iterations + 1):
    grad = cost_gradient(x[0], x[1])
    grad_norm = np.linalg.norm(grad)
    prev_cost = production_cost(x[0], x[1])
    
    if grad_norm < tolerance:
        converged_iter = it - 1
        print(f"--> Converged at iteration {converged_iter} with ||grad|| = {grad_norm:.8f} < {tolerance}")
        break
        
    x = x - learning_rate * grad
    curr_cost = production_cost(x[0], x[1])
    trajectory.append((x[0], x[1], curr_cost))
    
    if it <= 5 or it % 5 == 0 or it == max_iterations:
        reduction = prev_cost - curr_cost
        print(f"{it:4d}  | {x[0]:10.4f} | {x[1]:10.4f} | {curr_cost:12.4f} | {grad_norm:12.6f} | ${reduction:10.4f}")

final_cost = production_cost(x[0], x[1])
print("-" * 55)
print(f"Final Optimized Parameters : x1 = {x[0]:.4f}, x2 = {x[1]:.4f}")
print(f"Optimized Minimum Cost     : ${final_cost:.4f}")
print(f"Analytical Optimum Cost    : ${cost_analytical:.4f}")
print(f"Parameter Estimation Error : ||x_opt - x*|| = {np.linalg.norm(x - np.array([x1_analytical, x2_analytical])):.8f}")
print(f"Total Cost Reduction       : ${trajectory[0][2] - final_cost:.2f} ({((trajectory[0][2] - final_cost)/trajectory[0][2])*100:.1f}%)")

# Contour Plot with Gradient Descent Path
x1_grid = np.linspace(-1, 12, 100)
x2_grid = np.linspace(-1, 11, 100)
X1, X2 = np.meshgrid(x1_grid, x2_grid)
Z = production_cost(X1, X2)

traj_x1 = [p[0] for p in trajectory]
traj_x2 = [p[1] for p in trajectory]

plt.figure(figsize=(7.5, 5))
contours = plt.contour(X1, X2, Z, levels=25, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)
plt.plot(traj_x1, traj_x2, 'r.-', lw=1.8, markersize=5, label='Gradient Descent Path')
plt.plot(x1_analytical, x2_analytical, 'k*', markersize=12, label=f'Global Min ({x1_analytical}, {x2_analytical})')
plt.title("Manufacturing Cost Surface Contour & Gradient Descent Trajectory", fontweight='bold')
plt.xlabel("Raw Material Feed Rate x1 (kg/hr)")
plt.ylabel("Machine Operating Speed x2 (RPM/100)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("p14_manufacturing_gd.png", dpi=150)
# plt.show()
plt.close()
print("Visualization saved as 'p14_manufacturing_gd.png'.")

# Output:
# =======================================================
# MANUFACTURING: PRODUCTION COST MINIMIZATION VIA GRADIENT DESCENT
# =======================================================
# Cost Function: C(x1, x2) = 2*x1^2 + 3*x2^2 - 2*x1*x2 - 20*x1 - 30*x2 + 250
# Initial State       : x1 = 0.00, x2 = 0.00, Cost = $250.00
# Analytical Optimum  : x1* = 9.00, x2* = 8.00, Min Cost = $40.00
# Hyperparameters     : Learning Rate = 0.1, Tolerance = 1e-06
# 
# Iter  | x1 (Feed)  | x2 (Speed) | Cost ($)     | ||grad||     | Step Reduction
# -------------------------------------------------------
#    1  |     2.0000 |     3.0000 |     143.0000 |    36.055513 | $  107.0000
#    2  |     3.8000 |     4.6000 |      93.4000 |    24.083189 | $   49.6000
#    3  |     5.2000 |     5.6000 |      67.9200 |    17.204651 | $   25.4800
#    4  |     6.2400 |     6.2800 |      54.6160 |    12.425780 | $   13.3040
#    5  |     7.0000 |     6.7600 |      47.6528 |     8.988882 | $    6.9632
#   10  |     8.6029 |     7.7546 |      40.3012 |     1.783193 | $    0.2740
#   15  |     8.9212 |     7.9513 |      40.0119 |     0.353762 | $    0.0108
#   20  |     8.9844 |     7.9903 |      40.0005 |     0.070182 | $    0.0004
#   25  |     8.9969 |     7.9981 |      40.0000 |     0.013923 | $    0.0000
#   30  |     8.9994 |     7.9996 |      40.0000 |     0.002762 | $    0.0000
#   35  |     8.9999 |     7.9999 |      40.0000 |     0.000548 | $    0.0000
#   40  |     9.0000 |     8.0000 |      40.0000 |     0.000109 | $    0.0000
#   45  |     9.0000 |     8.0000 |      40.0000 |     0.000022 | $    0.0000
#   50  |     9.0000 |     8.0000 |      40.0000 |     0.000004 | $    0.0000
# --> Converged at iteration 54 with ||grad|| = 0.00000085 < 1e-06
# -------------------------------------------------------
# Final Optimized Parameters : x1 = 9.0000, x2 = 8.0000
# Optimized Minimum Cost     : $40.0000
# Analytical Optimum Cost    : $40.0000
# Parameter Estimation Error : ||x_opt - x*|| = 0.00000031
# Total Cost Reduction       : $210.00 (84.0%)
# Visualization saved as 'p14_manufacturing_gd.png'.

# ======================================================================
