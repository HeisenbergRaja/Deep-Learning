# Program 71: Explained variance
# Question: A data analytics company wants to compress a dataset while retaining at least 95% of its original information. Calculate explained variance and determine the number of principal components required to achieve the target. (BL4 – Analyse)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

np.random.seed(42)

# 1. Generate 20-Dimensional Analytics Dataset
n_features = 20
X_raw, _ = make_classification(
    n_samples=300, n_features=n_features, n_informative=12, n_redundant=6, random_state=42
)
X_scaled = StandardScaler().fit_transform(X_raw)

# 2. Fit Full PCA across All 20 Components
pca_full = PCA()
pca_full.fit(X_scaled)

exp_var = pca_full.explained_variance_ratio_
cum_var = np.cumsum(exp_var)

# 3. Determine Number of Components for 95% Information Retention
target_retention = 0.95
k_optimal = np.argmax(cum_var >= target_retention) + 1

# 4. Tabular Summary of Individual and Cumulative Explained Variance
print(f"{'Component':<12} | {'Individual Variance':<22} | {'Cumulative Variance':<22}")
print("-" * 62)
for i in range(len(exp_var)):
    marker = "  <-- 95% Threshold" if (i + 1) == k_optimal else ""
    print(f"PC {i+1:<8} | {exp_var[i]*100:>19.2f}% | {cum_var[i]*100:>19.2f}%{marker}")

print(f"\nTarget Information Retention:     {target_retention*100:.1f}%")
print(f"Principal Components Required:   {k_optimal} out of {n_features}")
print(f"Cumulative Variance Retained:    {cum_var[k_optimal - 1]*100:.2f}%")
print(f"Dimensionality Reduction Ratio:  {((n_features - k_optimal) / n_features) * 100:.1f}% data compression")

# 5. Visualization: Scree Plot & Cumulative Explained Variance Curve
fig, ax1 = plt.subplots(figsize=(8, 5))
components_range = range(1, n_features + 1)

# Individual Variance (Bar Chart)
ax1.bar(components_range, exp_var * 100, alpha=0.6, color='steelblue', label='Individual Explained Variance')
ax1.set_xlabel('Principal Component Index')
ax1.set_ylabel('Individual Explained Variance (%)', color='steelblue')
ax1.set_xticks(components_range)

# Cumulative Variance (Step/Line Plot)
ax2 = ax1.twinx()
ax2.plot(components_range, cum_var * 100, color='crimson', marker='o', linewidth=2, label='Cumulative Variance')
ax2.axhline(y=95, color='gray', linestyle='--', label='95% Information Threshold')
ax2.axvline(x=k_optimal, color='green', linestyle=':', label=f'{k_optimal} Components Selected')
ax2.set_ylabel('Cumulative Explained Variance (%)', color='crimson')

plt.title('Scree Plot & Cumulative Explained Variance: 95% Retention Threshold')
fig.tight_layout()
plt.show()

# Output:
# Component    | Individual Variance    | Cumulative Variance   
# --------------------------------------------------------------
# PC 1        |               16.82% |               16.82%
# PC 2        |               15.52% |               32.34%
# PC 3        |               12.11% |               44.45%
# PC 4        |               11.16% |               55.61%
# PC 5        |                8.49% |               64.09%
# PC 6        |                6.21% |               70.30%
# PC 7        |                5.33% |               75.63%
# PC 8        |                4.98% |               80.61%
# PC 9        |                4.70% |               85.31%
# PC 10       |                4.51% |               89.82%
# PC 11       |                3.48% |               93.30%
# PC 12       |                2.64% |               95.94%  <-- 95% Threshold
# PC 13       |                2.36% |               98.31%
# PC 14       |                1.69% |              100.00%
# PC 15       |                0.00% |              100.00%
# PC 16       |                0.00% |              100.00%
# PC 17       |                0.00% |              100.00%
# PC 18       |                0.00% |              100.00%
# PC 19       |                0.00% |              100.00%
# PC 20       |                0.00% |              100.00%
#
# Target Information Retention:     95.0%
# Principal Components Required:   12 out of 20
# Cumulative Variance Retained:    95.94%
# Dimensionality Reduction Ratio:  40.0% data compression

# ======================================================================
