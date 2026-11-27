# Program 12: Spam Detection: Cross-Entropy Loss Calculation
# Question: Spam detection: Calculate cross-entropy loss using actual and predicted probabilities.

import numpy as np
import matplotlib.pyplot as plt

def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Computes per-sample Binary Cross-Entropy (Log Loss):
    L(y, p) = - [y * log(p) + (1 - y) * log(1 - p)]
    """
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return - (y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))

def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Computes Categorical Cross-Entropy for multi-class classification:
    L(y, p) = - sum(y_c * log(p_c))
    """
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return - np.sum(y_true * np.log(y_pred), axis=-1)

# --- 1. Binary Spam Detection Scenario (10 Emails) ---
# True labels: 1 = Spam, 0 = Ham (Legitimate)
y_actual = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])

# Model 1: Well-calibrated, high-confidence classifier
y_pred_m1 = np.array([0.92, 0.08, 0.88, 0.95, 0.05, 0.12, 0.85, 0.09, 0.91, 0.04])
# Model 2: Under-confident / vague classifier
y_pred_m2 = np.array([0.60, 0.40, 0.58, 0.65, 0.45, 0.42, 0.62, 0.48, 0.55, 0.46])
# Model 3: Confident model with a critical misclassification on Email #2 (Ham predicted as 0.99 Spam)
y_pred_m3 = np.array([0.99, 0.99, 0.98, 0.95, 0.02, 0.01, 0.99, 0.02, 0.97, 0.03])

losses_m1 = binary_cross_entropy(y_actual, y_pred_m1)
losses_m2 = binary_cross_entropy(y_actual, y_pred_m2)
losses_m3 = binary_cross_entropy(y_actual, y_pred_m3)

print("=" * 55)
print("SPAM DETECTION: BINARY CROSS-ENTROPY LOSS")
print("=" * 55)
print(f"{'Email':<6} | {'Actual':<10} | {'Model 1 (Good)':<18} | {'Model 2 (Vague)':<18} | {'Model 3 (Erroneous)'}")
print("-" * 55)
for i in range(len(y_actual)):
    label_str = "Spam (1)" if y_actual[i] == 1 else "Ham  (0)"
    print(f" #{i+1:<4} | {label_str:<10} | p={y_pred_m1[i]:.2f}, L={losses_m1[i]:.4f} | p={y_pred_m2[i]:.2f}, L={losses_m2[i]:.4f} | p={y_pred_m3[i]:.2f}, L={losses_m3[i]:.4f}")
print("-" * 55)
print(f"Average BCE Loss across 10 Emails:")
print(f"  Model 1 (Well-Calibrated) : {np.mean(losses_m1):.4f}")
print(f"  Model 2 (Under-Confident)  : {np.mean(losses_m2):.4f}")
print(f"  Model 3 (Confident Error)  : {np.mean(losses_m3):.4f}")
print("\nLoss Analysis:")
print(f"On Email #2 (actual Ham), Model 3 confidently predicted Spam with p=0.99.")
print(f"Cross-entropy penalizes this with a massive loss of {losses_m3[1]:.4f}, dragging")
print(f"down its overall performance. Cross-entropy heavily punishes confident wrong answers.")

# --- 2. Multi-Class Email Routing Scenario (Primary, Social, Spam) ---
categories = ['Primary', 'Social', 'Spam']
y_mc_true = np.array([
    [1, 0, 0],  # Email 1: Primary
    [0, 1, 0],  # Email 2: Social
    [0, 0, 1],  # Email 3: Spam
    [1, 0, 0]   # Email 4: Primary
])
y_mc_pred = np.array([
    [0.85, 0.10, 0.05],
    [0.10, 0.80, 0.10],
    [0.05, 0.05, 0.90],
    [0.70, 0.20, 0.10]
])
cce_losses = categorical_cross_entropy(y_mc_true, y_mc_pred)

print("\n" + "=" * 55)
print("MULTI-CLASS EMAIL CATEGORIZATION (CATEGORICAL CROSS-ENTROPY)")
print("=" * 55)
for idx, (t, p, l) in enumerate(zip(y_mc_true, y_mc_pred, cce_losses)):
    true_cat = categories[np.argmax(t)]
    pred_str = ", ".join([f"{cat}:{prob:.2f}" for cat, prob in zip(categories, p)])
    print(f"Email {idx+1} [True: {true_cat:<7}] -> Probabilities: [{pred_str}] => Loss: {l:.4f}")
print(f"Mean Categorical Cross-Entropy Loss: {np.mean(cce_losses):.4f}")

# Visualize Cross-Entropy Loss Curve
p_curve = np.linspace(0.01, 0.99, 200)
loss_y1 = -np.log(p_curve)
loss_y0 = -np.log(1.0 - p_curve)

plt.figure(figsize=(7.5, 4.5))
plt.plot(p_curve, loss_y1, 'r-', lw=2, label="Actual = 1 (Spam) : L = -log(p)")
plt.plot(p_curve, loss_y0, 'b-', lw=2, label="Actual = 0 (Ham)  : L = -log(1 - p)")
plt.title("Cross-Entropy Loss vs. Predicted Probability", fontweight='bold')
plt.xlabel("Predicted Probability of Spam (p)")
plt.ylabel("Cross-Entropy Loss")
plt.ylim(0, 5)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("p12_cross_entropy.png", dpi=150)
# plt.show()
plt.close()
print("Visualization saved as 'p12_cross_entropy.png'.")

# Output:
# =======================================================
# SPAM DETECTION: BINARY CROSS-ENTROPY LOSS
# =======================================================
# Email  | Actual     | Model 1 (Good)     | Model 2 (Vague)    | Model 3 (Erroneous)
# -------------------------------------------------------
#  #1    | Spam (1)   | p=0.92, L=0.0834 | p=0.60, L=0.5108 | p=0.99, L=0.0101
#  #2    | Ham  (0)   | p=0.08, L=0.0834 | p=0.40, L=0.5108 | p=0.99, L=4.6052
#  #3    | Spam (1)   | p=0.88, L=0.1278 | p=0.58, L=0.5447 | p=0.98, L=0.0202
#  #4    | Spam (1)   | p=0.95, L=0.0513 | p=0.65, L=0.4308 | p=0.95, L=0.0513
#  #5    | Ham  (0)   | p=0.05, L=0.0513 | p=0.45, L=0.5978 | p=0.02, L=0.0202
#  #6    | Ham  (0)   | p=0.12, L=0.1278 | p=0.42, L=0.5447 | p=0.01, L=0.0101
#  #7    | Spam (1)   | p=0.85, L=0.1625 | p=0.62, L=0.4780 | p=0.99, L=0.0101
#  #8    | Ham  (0)   | p=0.09, L=0.0943 | p=0.48, L=0.6539 | p=0.02, L=0.0202
#  #9    | Spam (1)   | p=0.91, L=0.0943 | p=0.55, L=0.5978 | p=0.97, L=0.0305
#  #10   | Ham  (0)   | p=0.04, L=0.0408 | p=0.46, L=0.6162 | p=0.03, L=0.0305
# -------------------------------------------------------
# Average BCE Loss across 10 Emails:
#   Model 1 (Well-Calibrated) : 0.0917
#   Model 2 (Under-Confident)  : 0.5486
#   Model 3 (Confident Error)  : 0.4808
# 
# Loss Analysis:
# On Email #2 (actual Ham), Model 3 confidently predicted Spam with p=0.99.
# Cross-entropy penalizes this with a massive loss of 4.6052, dragging
# down its overall performance. Cross-entropy heavily punishes confident wrong answers.
# 
# =======================================================
# MULTI-CLASS EMAIL CATEGORIZATION (CATEGORICAL CROSS-ENTROPY)
# =======================================================
# Email 1 [True: Primary] -> Probabilities: [Primary:0.85, Social:0.10, Spam:0.05] => Loss: 0.1625
# Email 2 [True: Social ] -> Probabilities: [Primary:0.10, Social:0.80, Spam:0.10] => Loss: 0.2231
# Email 3 [True: Spam   ] -> Probabilities: [Primary:0.05, Social:0.05, Spam:0.90] => Loss: 0.1054
# Email 4 [True: Primary] -> Probabilities: [Primary:0.70, Social:0.20, Spam:0.10] => Loss: 0.3567
# Mean Categorical Cross-Entropy Loss: 0.2119
# Visualization saved as 'p12_cross_entropy.png'.

# ======================================================================
