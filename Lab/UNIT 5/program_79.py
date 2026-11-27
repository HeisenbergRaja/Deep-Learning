# Program 79: Gaussian Mixture Model for Customer Segmentation
# Question: Develop GMM to cluster customers, estimate probability of each customer belonging to different clusters.
# ======================================================================

import numpy as np
from sklearn.mixture import GaussianMixture

np.random.seed(42)

# 1. Synthesize 500 customers across 4 behavioral dimensions:
# [Annual_Income_k$, Spending_Score (1-100), Purchase_Freq (orders/mo), Avg_Order_Value_$]
# Cluster 0: Impulsive Spenders (Modest income, high spending, high frequency)
c0 = np.random.multivariate_normal([35, 75, 6.0, 80], np.diag([20, 25, 1.0, 70]), size=100)
# Cluster 1: VIP Spenders (High income, high spending, frequent large orders)
c1 = np.random.multivariate_normal([95, 85, 8.0, 250], np.diag([30, 20, 1.5, 120]), size=130)
# Cluster 2: Budget Savers (Low income, low spending, low frequency)
c2 = np.random.multivariate_normal([30, 25, 2.0, 40], np.diag([15, 20, 0.5, 50]), size=150)
# Cluster 3: Affluent Savers (High income, conservative spending, moderate frequency)
c3 = np.random.multivariate_normal([90, 30, 3.0, 120], np.diag([25, 15, 0.8, 80]), size=120)

X_customers = np.vstack([c0, c1, c2, c3])

# 2. Fit Gaussian Mixture Model with 4 components
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm.fit(X_customers)

# Predict cluster labels and posterior probabilities (responsibilities)
cluster_labels = gmm.predict(X_customers)
posterior_probs = gmm.predict_proba(X_customers)

print("--- Program 79: Gaussian Mixture Model for Customer Segmentation ---")
print(f"Model Fit Selection Metrics: AIC = {gmm.aic(X_customers):.1f}, BIC = {gmm.bic(X_customers):.1f}")

persona_names = {
    0: "Impulsive Spenders",
    1: "VIP High-Spenders",
    2: "Budget Savers",
    3: "Affluent Savers"
}

print("\nEstimated Cluster Profiles (Centroids & Persona Mapping):")
print(f"{'Cluster':<9} {'Weight':<8} {'Income($k)':<12} {'SpendScore':<12} {'PurchFreq':<12} {'AvgOrderVal($)':<15} {'Persona'}")
print("-" * 85)
for k in range(4):
    mean = gmm.means_[k]
    weight = gmm.weights_[k]
    print(f"Cluster {k:<2} {weight:<8.3f} {mean[0]:<12.1f} {mean[1]:<12.1f} {mean[2]:<12.1f} {mean[3]:<15.1f} {persona_names.get(k, 'Unknown')}")

print("\nCustomer Soft Clustering: Posterior Membership Probabilities P(Cluster | Data):")
sample_indices = [0, 105, 235, 385] # Representative customers from each true group
for idx in sample_indices:
    feats = np.round(X_customers[idx], 1)
    probs = np.round(posterior_probs[idx], 4)
    assigned = cluster_labels[idx]
    print(f"Customer #{idx+1} [Income: ${feats[0]:.0f}k, Spend: {feats[1]:.0f}, Freq: {feats[2]:.1f}/mo, AOV: ${feats[3]:.0f}]:")
    print(f"  Posterior Probabilities: {probs}")
    print(f"  -> Assigned to Cluster {assigned} ({persona_names[assigned]}) with {probs[assigned]*100:.1f}% confidence\n")

# Output:
# --- Program 79: Gaussian Mixture Model for Customer Segmentation ---
# Model Fit Selection Metrics: AIC = 12181.8, BIC = 12430.5
#
# Estimated Cluster Profiles (Centroids & Persona Mapping):
# Cluster   Weight   Income($k)   SpendScore   PurchFreq    AvgOrderVal($)  Persona
# -------------------------------------------------------------------------------------
# Cluster 0  0.200    35.3         75.4         6.1          79.9            Impulsive Spenders
# Cluster 1  0.260    96.0         84.9         8.0          250.2           VIP High-Spenders
# Cluster 2  0.300    30.0         24.9         2.0          39.9            Budget Savers
# Cluster 3  0.240    90.2         30.6         3.0          121.1           Affluent Savers
#
# Customer Soft Clustering: Posterior Membership Probabilities P(Cluster | Data):
# Customer #1 [Income: $37k, Spend: 74, Freq: 5.6/mo, AOV: $75]:
#   Posterior Probabilities: [1. 0. 0. 0.]
#   -> Assigned to Cluster 0 (Impulsive Spenders) with 100.0% confidence
#
# Customer #106 [Income: $97k, Spend: 83, Freq: 8.6/mo, AOV: $249]:
#   Posterior Probabilities: [0. 1. 0. 0.]
#   -> Assigned to Cluster 1 (VIP High-Spenders) with 100.0% confidence
#
# Customer #236 [Income: $31k, Spend: 28, Freq: 1.8/mo, AOV: $42]:
#   Posterior Probabilities: [0. 0. 1. 0.]
#   -> Assigned to Cluster 2 (Budget Savers) with 100.0% confidence
#
# Customer #386 [Income: $88k, Spend: 28, Freq: 3.4/mo, AOV: $123]:
#   Posterior Probabilities: [0. 0. 0. 1.]
#   -> Assigned to Cluster 3 (Affluent Savers) with 100.0% confidence
#
# Output comments:
# The GMM accurately recovers all 4 customer personas with appropriate mixing weights (0.20, 0.26, 0.30, 0.24).
# Unlike hard K-Means, GMM provides full posterior membership probability distributions, enabling
# probabilistic customer segmentation and uncertainty quantification for boundary cases.

# ======================================================================
