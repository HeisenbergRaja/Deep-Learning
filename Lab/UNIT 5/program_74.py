# Program 74: PCA vs ICA Comparison on Multivariate Dataset
# Question: Implement both PCA and ICA on multivariate dataset, compare results based on extracted component characteristics.
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA

np.random.seed(42)
n_samples = 2000

# 1. Generate 3 independent non-Gaussian sources with distinct kurtosis profiles
# Source 1: Uniform distribution (sub-Gaussian / platykurtic, negative excess kurtosis)
s1 = np.random.uniform(-np.sqrt(3), np.sqrt(3), n_samples)
# Source 2: Deterministic sawtooth wave (non-Gaussian, negative excess kurtosis)
s2 = 2 * (np.mod(np.linspace(0, 50, n_samples), 1.0)) - 1.0
# Source 3: Laplace distribution (super-Gaussian / leptokurtic, positive excess kurtosis)
s3 = np.random.laplace(0, 1.0 / np.sqrt(2), n_samples)

S = np.c_[s1, s2, s3]
S = (S - S.mean(axis=0)) / S.std(axis=0)

# 2. Linear mixing into observed multivariate dataset
A = np.array([[0.5, 0.3, 0.2],
              [0.1, 0.7, 0.2],
              [0.3, 0.2, 0.5]])
X = np.dot(S, A.T)

# 3. Fit PCA: finds orthogonal directions of maximal variance
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X)

# 4. Fit ICA: finds statistically independent, non-Gaussian components
ica = FastICA(n_components=3, random_state=42, max_iter=1000)
X_ica = ica.fit_transform(X)

# 5. Calculate Excess Kurtosis (Non-Gaussianity: Gaussian excess kurtosis = 0)
def excess_kurtosis(data):
    mean = np.mean(data, axis=0)
    m4 = np.mean((data - mean)**4, axis=0)
    m2 = np.var(data, axis=0)
    return (m4 / (m2**2)) - 3.0

kurt_orig = excess_kurtosis(S)
kurt_pca = excess_kurtosis(X_pca)
kurt_ica = excess_kurtosis(X_ica)

# 6. Calculate Component Covariance (Orthogonality check)
cov_pca = np.cov(X_pca, rowvar=False)
cov_ica = np.cov(X_ica, rowvar=False)

print("--- Program 74: PCA vs ICA Comparison Results ---")
print("1. Excess Kurtosis Comparison (Gaussian baseline = 0.0):")
print(f"   Original Sources:   {np.round(kurt_orig, 3)}")
print(f"   PCA Components:     {np.round(kurt_pca, 3)}")
print(f"   ICA Components:     {np.round(kurt_ica, 3)}")

print("\n2. Component Covariance / Orthogonality:")
print("   PCA Component Covariance (strictly diagonal, orthogonal):")
print(np.round(cov_pca, 4))
print("   ICA Component Covariance (statistically independent):")
print(np.round(cov_ica, 4))

print("\n3. Variance Distribution:")
print(f"   PCA Explained Variance Ratios: {np.round(pca.explained_variance_ratio_, 4)}")
print(f"   PCA Cumulative Variance:       {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
print(f"   ICA Component Variances:       {np.round(np.var(X_ica, axis=0), 4)}")

# Plot distributions comparing PCA and ICA component shapes
fig, axes = plt.subplots(2, 3, figsize=(11, 5))
for i in range(3):
    axes[0, i].hist(X_pca[:, i], bins=40, color='tab:blue', alpha=0.7, density=True)
    axes[0, i].set_title(f"PCA Comp {i+1} (Kurt: {kurt_pca[i]:.2f})")
    axes[1, i].hist(X_ica[:, i], bins=40, color='tab:green', alpha=0.7, density=True)
    axes[1, i].set_title(f"ICA Comp {i+1} (Kurt: {kurt_ica[i]:.2f})")
plt.tight_layout()
plt.savefig("program_74_pca_vs_ica_distributions.png", dpi=120)
plt.close()
print("\nComponent distribution histograms saved to 'program_74_pca_vs_ica_distributions.png'.")

# Output:
# --- Program 74: PCA vs ICA Comparison Results ---
# 1. Excess Kurtosis Comparison (Gaussian baseline = 0.0):
#    Original Sources:   [-1.227 -1.2    3.93 ]
#    PCA Components:     [-0.013 -0.206  0.635]
#    ICA Components:     [-1.22  -1.199  3.947]
#
# 2. Component Covariance / Orthogonality:
#    PCA Component Covariance (strictly diagonal, orthogonal):
# [[ 1.0561  0.     -0.    ]
#  [ 0.      0.2043  0.    ]
#  [-0.      0.      0.0667]]
#    ICA Component Covariance (statistically independent):
# [[ 1.0005 -0.      0.    ]
#  [-0.      1.0005 -0.    ]
#  [ 0.     -0.      1.0005]]
#
# 3. Variance Distribution:
#    PCA Explained Variance Ratios: [0.7958 0.154  0.0503]
#    PCA Cumulative Variance:       100.00%
#    ICA Component Variances:       [1. 1. 1.]
#
# Component distribution histograms saved to 'program_74_pca_vs_ica_distributions.png'.
#
# Output comments:
# - PCA finds orthogonal coordinate axes that maximize variance (ranked: 79.58%, 15.40%, 5.03%).
#   However, its components are linear mixtures with kurtosis close to 0 due to the Central Limit Theorem.
# - ICA reconstructs the underlying non-Gaussian independent sources, accurately recovering their
#   true excess kurtoses (-1.22, -1.20, +3.95). ICA focuses on higher-order independence rather than variance.

# ======================================================================
