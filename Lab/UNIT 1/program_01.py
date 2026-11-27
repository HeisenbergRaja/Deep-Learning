# Program 1: Robotics navigation
# Question: Implement vector addition, subtraction, and scalar multiplication to calculate robot's new position using 3D coordinates.

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Setup Initial 3D Coordinates & Parameters
# ----------------------------------------------------------------------
# Initial 3D position vector P0 = [x, y, z] of the robot in meters
pos_initial = np.array([12.0, 8.0, 2.0], dtype=float)

# Velocity vector V = [vx, vy, vz] in meters per second (m/s)
velocity = np.array([2.5, -1.0, 0.5], dtype=float)

# Time elapsed (t) in seconds (scalar)
time_step = 4.0

# Position of a target obstacle in 3D workspace [x, y, z] in meters
pos_obstacle = np.array([25.0, 10.0, 6.0], dtype=float)

# ----------------------------------------------------------------------
# 2. Vector Operations
# ----------------------------------------------------------------------
# A. SCALAR MULTIPLICATION: Displacement = velocity * time
#    Formula: d = t * V = [t*vx, t*vy, t*vz]
displacement = time_step * velocity

# B. VECTOR ADDITION: New Position = Initial Position + Displacement
#    Formula: P_new = P0 + d = [x0 + dx, y0 + dy, z0 + dz]
pos_new = pos_initial + displacement

# C. VECTOR SUBTRACTION: Vector from new robot position to target obstacle
#    Formula: vec_obs = P_obstacle - P_new
vec_to_obstacle = pos_obstacle - pos_new

# D. EUCLIDEAN DISTANCE & DIRECTION (L2-norm)
#    Formula: ||vec_obs|| = sqrt(x^2 + y^2 + z^2)
distance_to_obstacle = np.linalg.norm(vec_to_obstacle)
unit_direction_vector = vec_to_obstacle / distance_to_obstacle

# ----------------------------------------------------------------------
# 3. Display Results
# ----------------------------------------------------------------------
print("=" * 60)
print("ROBOTICS 3D NAVIGATION: VECTOR OPERATIONS")
print("=" * 60)
print(f"Initial Robot Position (P0)                  : {pos_initial} m")
print(f"Robot Velocity Vector (V)                    : {velocity} m/s")
print(f"Time Step Duration (scalar t)                : {time_step} s")
print("-" * 60)
print(f"[Scalar Multiplication] Displacement (V * t) : {displacement} m")
print(f"[Vector Addition] New Position (P0 + V*t)    : {pos_new} m")
print(f"Target Obstacle Position (P_obs)             : {pos_obstacle} m")
print(f"[Vector Subtraction] Vector to Obstacle      : {vec_to_obstacle} m")
print(f"Euclidean Distance to Obstacle               : {distance_to_obstacle:.4f} m")
print(f"Unit Direction Vector to Obstacle            : {np.round(unit_direction_vector, 4)}")
print("=" * 60)

# ----------------------------------------------------------------------
# 4. 3D Trajectory Visualization using Matplotlib
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot positions
ax.scatter(*pos_initial, color='blue', s=80, label='Initial Position (P0)')
ax.scatter(*pos_new, color='green', s=100, label='New Position (P_new)')
ax.scatter(*pos_obstacle, color='red', s=100, marker='^', label='Obstacle (P_obs)')

# Plot trajectory and obstacle vectors
ax.plot([pos_initial[0], pos_new[0]], [pos_initial[1], pos_new[1]], [pos_initial[2], pos_new[2]],
        'g--', linewidth=2, label='Trajectory Vector')
ax.plot([pos_new[0], pos_obstacle[0]], [pos_new[1], pos_obstacle[1]], [pos_new[2], pos_obstacle[2]],
        'r:', linewidth=2, label='Vector to Obstacle')

ax.set_title("Robot 3D Navigation & Obstacle Vector")
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.set_zlabel("Z (meters)")
ax.legend(loc='best')
plt.tight_layout()
# plt.show() # Display plot window (uncomment when running interactively)
plt.close()

# Output:
# ============================================================
# ROBOTICS 3D NAVIGATION: VECTOR OPERATIONS
# ============================================================
# Initial Robot Position (P0)                  : [12.  8.  2.] m
# Robot Velocity Vector (V)                    : [ 2.5 -1.   0.5] m/s
# Time Step Duration (scalar t)                : 4.0 s
# ------------------------------------------------------------
# [Scalar Multiplication] Displacement (V * t) : [10. -4.  2.] m
# [Vector Addition] New Position (P0 + V*t)    : [22.  4.  4.] m
# Target Obstacle Position (P_obs)             : [25. 10.  6.] m
# [Vector Subtraction] Vector to Obstacle      : [3. 6. 2.] m
# Euclidean Distance to Obstacle               : 7.0000 m
# Unit Direction Vector to Obstacle            : [0.4286 0.8571 0.2857]
# ============================================================

# ======================================================================
