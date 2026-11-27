# Program 69: PCA visualization
# Question: A research organization collects high-dimensional data containing multiple features. Apply PCA and visualize the dataset using the first two principal components to identify patterns and clusters. (BL3 – Apply)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

np.random.seed(42)

# 1. Generate High-Dimensional Synthetic Dataset (300 samples, 20 features, 3 clusters)
X_blobs, y_blobs = make_blobs(
    n_samples=300, n_features=20, centers=3, cluster_std=2.5, random_state=42
)

# 2. Standardize Features
X_blobs_std = StandardScaler().fit_transform(X_blobs)

# 3. Apply PCA to Project 20 Dimensions into 2 Principal Components
pca_vis = PCA(n_components=2)
X_blobs_pca = pca_vis.fit_transform(X_blobs_std)

var1 = pca_vis.explained_variance_ratio_[0] * 100
var2 = pca_vis.explained_variance_ratio_[1] * 100

# 4. Pattern and Dimensionality Analysis
print(f"Original High-Dimensional Space: {X_blobs.shape[1]} features")
print(f"Transformed Projection Space:     {X_blobs_pca.shape[1]} principal components")
print(f"PC1 Explained Variance:          {var1:.2f}%")
print(f"PC2 Explained Variance:          {var2:.2f}%")
print(f"Total 2D Explained Variance:     {var1 + var2:.2f}%")
print("Pattern Analysis: High-dimensional clusters project into 3 distinct, well-separated clusters in the 2D plane.")

# 5. Visualize Clusters along PC1 and PC2
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_blobs_pca[:, 0], X_blobs_pca[:, 1],
    c=y_blobs, cmap='viridis', edgecolors='k', s=60, alpha=0.85
)
plt.colorbar(scatter, label="Underlying Cluster ID")
plt.xlabel(f"Principal Component 1 ({var1:.2f}% Variance)")
plt.ylabel(f"Principal Component 2 ({var2:.2f}% Variance)")
plt.title("PCA 2D Projection of 20-Dimensional Dataset: Cluster Pattern Identification")
plt.grid(True)
plt.tight_layout()
plt.show()

# Output:
# Original High-Dimensional Space: 20 features
# Transformed Projection Space:     2 principal components
# PC1 Explained Variance:          47.28%
# PC2 Explained Variance:          24.90%
# Total 2D Explained Variance:     72.18%
# Pattern Analysis: High-dimensional clusters project into 3 distinct, well-separated clusters in the 2D plane.

# ======================================================================
