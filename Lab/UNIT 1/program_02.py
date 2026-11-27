# Program 2: Image processing
# Question: Perform matrix multiplication and transpose operations, analyse how they affect resulting matrix.

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Image Representation as a Matrix
# ----------------------------------------------------------------------
# Sample 4x3 grayscale image patch with pixel intensity values [0, 255]
image_matrix = np.array([
    [150,  20, 210],
    [ 80, 200,  45],
    [190,  30, 240],
    [ 60, 175,  90]
], dtype=float)

print("=" * 60)
print("IMAGE PROCESSING: MATRIX MULTIPLICATION & TRANSPOSE")
print("=" * 60)
print("Original Image Matrix (Shape: {}):".format(image_matrix.shape))
print(image_matrix)

# ----------------------------------------------------------------------
# 2. Transpose Operation
# ----------------------------------------------------------------------
# Transposing flips rows and columns: A_T[i, j] = A[j, i]
transposed_matrix = image_matrix.T

print("-" * 60)
print("Transposed Matrix (Shape: {}):".format(transposed_matrix.shape))
print(transposed_matrix)

# ----------------------------------------------------------------------
# 3. Matrix Multiplication
# ----------------------------------------------------------------------
# Define a 3x3 transformation matrix T (color channel weighting / spatial projection)
T = np.array([
    [0.6, 0.2, 0.2],
    [0.1, 0.8, 0.1],
    [0.3, 0.3, 0.4]
], dtype=float)

print("-" * 60)
print("Linear Transformation Matrix T (Shape: {}):".format(T.shape))
print(T)

# Matrix multiplication: Result = image_matrix @ T
# Shape: (4, 3) @ (3, 3) -> (4, 3)
transformed_matrix = np.matmul(image_matrix, T)

print("-" * 60)
print("Transformed Image Matrix (image_matrix @ T, Shape: {}):".format(transformed_matrix.shape))
print(np.round(transformed_matrix, 2))

# ----------------------------------------------------------------------
# 4. In-depth Analysis of Matrix Operations
# ----------------------------------------------------------------------
print("-" * 60)
print("ANALYSIS OF HOW OPERATIONS AFFECT THE MATRIX:")
print("-" * 60)
print("1. Transpose Operation:")
print(f"   - Dimensional Change: {image_matrix.shape} -> {transposed_matrix.shape} (Rows <-> Columns swapped).")
print(f"   - Spatial Effect    : 90-degree rotation + horizontal reflection across main diagonal.")
print(f"   - Double Transpose  : ((A^T)^T == A) is {np.array_equal(transposed_matrix.T, image_matrix)}.")
print(f"   - Diagonal Element  : Element [0, 2] ({image_matrix[0, 2]}) moves to [2, 0] ({transposed_matrix[2, 0]}).")

print("\n2. Matrix Multiplication:")
print(f"   - Dimensional Rule  : Inner dimensions must match ({image_matrix.shape[1]} == {T.shape[0]}).")
print(f"   - Resulting Shape   : ({image_matrix.shape[0]}, {T.shape[1]}).")
print("   - Linear Combination: Each new column is a linear combination of all original columns.")
print("   - Non-Commutative   : A @ B != B @ A in general (order of transformation matters).")
# Algebraic property: (A @ B).T == B.T @ A.T
transpose_product_check = np.allclose((image_matrix @ T).T, T.T @ image_matrix.T)
print(f"   - Property Check    : (A @ T)^T == T^T @ A^T -> {transpose_product_check}")
print("=" * 60)

# ----------------------------------------------------------------------
# 5. Visualizing Matrices as Heatmaps using Matplotlib
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
im0 = axes[0].imshow(image_matrix, cmap='viridis', aspect='auto')
axes[0].set_title(f"Original Matrix {image_matrix.shape}")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(transposed_matrix, cmap='viridis', aspect='auto')
axes[1].set_title(f"Transposed Matrix {transposed_matrix.shape}")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(transformed_matrix, cmap='viridis', aspect='auto')
axes[2].set_title(f"Transformed Matrix {transformed_matrix.shape}")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
# plt.show() # Display plot window (uncomment when running interactively)
plt.close()

# Output:
# ============================================================
# IMAGE PROCESSING: MATRIX MULTIPLICATION & TRANSPOSE
# ============================================================
# Original Image Matrix (Shape: (4, 3)):
# [[150.  20. 210.]
#  [ 80. 200.  45.]
#  [190.  30. 240.]
#  [ 60. 175.  90.]]
# ------------------------------------------------------------
# Transposed Matrix (Shape: (3, 4)):
# [[150.  80. 190.  60.]
#  [ 20. 200.  30. 175.]
#  [210.  45. 240.  90.]]
# ------------------------------------------------------------
# Linear Transformation Matrix T (Shape: (3, 3)):
# [[0.6 0.2 0.2]
#  [0.1 0.8 0.1]
#  [0.3 0.3 0.4]]
# ------------------------------------------------------------
# Transformed Image Matrix (image_matrix @ T, Shape: (4, 3)):
# [[155.  109.  116. ]
#  [ 81.5 189.5  54. ]
#  [189.  134.  137. ]
#  [ 80.5 179.   65.5]]
# ------------------------------------------------------------
# ANALYSIS OF HOW OPERATIONS AFFECT THE MATRIX:
# ------------------------------------------------------------
# 1. Transpose Operation:
#    - Dimensional Change: (4, 3) -> (3, 4) (Rows <-> Columns swapped).
#    - Spatial Effect    : 90-degree rotation + horizontal reflection across main diagonal.
#    - Double Transpose  : ((A^T)^T == A) is True.
#    - Diagonal Element  : Element [0, 2] (210.0) moves to [2, 0] (210.0).
#
# 2. Matrix Multiplication:
#    - Dimensional Rule  : Inner dimensions must match (3 == 3).
#    - Resulting Shape   : (4, 3).
#    - Linear Combination: Each new column is a linear combination of all original columns.
#    - Non-Commutative   : A @ B != B @ A in general (order of transformation matters).
#    - Property Check    : (A @ T)^T == T^T @ A^T -> True
# ============================================================

# ======================================================================
