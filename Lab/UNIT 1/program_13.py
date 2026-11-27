# Program 13: Speech Recognition: Kullback-Leibler (KL) Divergence
# Question: Speech recognition: Implement KL divergence to measure difference between actual and predicted probability distributions.

import numpy as np
import matplotlib.pyplot as plt

def kl_divergence(P, Q, epsilon=1e-12):
    """
    Computes Kullback-Leibler Divergence: D_KL(P || Q) = sum P(i) * log2(P(i) / Q(i))
    Measured in bits (using base-2 logarithm).
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    P = np.clip(P, epsilon, 1.0)
    Q = np.clip(Q, epsilon, 1.0)
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    return np.sum(P * np.log2(P / Q))

def jensen_shannon_divergence(P, Q):
    """
    Computes Jensen-Shannon Divergence: JSD(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
    Symmetric and bounded between 0 and 1 bit.
    """
    P = np.asarray(P, dtype=np.float64) / np.sum(P)
    Q = np.asarray(Q, dtype=np.float64) / np.sum(Q)
    M = 0.5 * (P + Q)
    return 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)

# --- Speech Recognition Acoustic Model Scenario ---
# 8 Phoneme targets representing vowel and consonant classes
phonemes = ['/aa/', '/iy/', '/uw/', '/eh/', '/sh/', '/t/', '/k/', '/n/']

# Ground Truth Distribution P (Annotated speech segment with dominant /aa/ vowel)
P_true = np.array([0.45, 0.25, 0.12, 0.08, 0.04, 0.03, 0.02, 0.01])

# Model A (Well-trained acoustic model closely matching ground truth)
Q_model_A = np.array([0.42, 0.26, 0.14, 0.07, 0.05, 0.03, 0.02, 0.01])

# Model B (Under-trained / diffuse model with high phonetic confusion)
Q_model_B = np.array([0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10])

# Model C (Flawed model exhibiting mode collapse: assigns near zero to active /aa/ phoneme)
Q_model_C = np.array([0.01, 0.35, 0.25, 0.15, 0.08, 0.06, 0.06, 0.04])

# Compute Divergences
kl_A_forward = kl_divergence(P_true, Q_model_A)
kl_A_reverse = kl_divergence(Q_model_A, P_true)
jsd_A = jensen_shannon_divergence(P_true, Q_model_A)

kl_B_forward = kl_divergence(P_true, Q_model_B)
kl_B_reverse = kl_divergence(Q_model_B, P_true)
jsd_B = jensen_shannon_divergence(P_true, Q_model_B)

kl_C_forward = kl_divergence(P_true, Q_model_C)
kl_C_reverse = kl_divergence(Q_model_C, P_true)
jsd_C = jensen_shannon_divergence(P_true, Q_model_C)

print("=" * 55)
print("SPEECH RECOGNITION: KULLBACK-LEIBLER (KL) DIVERGENCE")
print("=" * 55)
print(f"Phoneme Classes: {phonemes}\n")
print(f"True Distribution P       : {np.round(P_true, 3)}")
print(f"Model A Distribution Q_A  : {np.round(Q_model_A, 3)}")
print(f"Model B Distribution Q_B  : {np.round(Q_model_B, 3)}")
print(f"Model C Distribution Q_C  : {np.round(Q_model_C, 3)}")
print("-" * 55)
print(f"Self-Divergence D_KL(P || P)            : {kl_divergence(P_true, P_true):.6f} bits (Zero information loss)")
print("-" * 55)
print("MODEL COMPARISON TABLE (in bits):")
print(f"{'Model':<15} | {'Forward KL D(P||Q)':<20} | {'Reverse KL D(Q||P)':<20} | {'JSD (Symmetric)'}")
print("-" * 55)
print(f"{'Model A (Good)':<15} | {kl_A_forward:20.4f} | {kl_A_reverse:20.4f} | {jsd_A:.4f}")
print(f"{'Model B (Vague)':<15} | {kl_B_forward:20.4f} | {kl_B_reverse:20.4f} | {jsd_B:.4f}")
print(f"{'Model C (Flawed)':<15} | {kl_C_forward:20.4f} | {kl_C_reverse:20.4f} | {jsd_C:.4f}")
print("-" * 55)
print("THEORETICAL INSIGHTS:")
print("1. Asymmetry: D_KL(P || Q) != D_KL(Q || P). KL divergence is a directed divergence, not a distance metric.")
print("2. Zero-Avoiding Property: D_KL(P || Q_C) is very large (2.0287 bits) because Model C assigns near zero")
print("   to /aa/ where true P has high probability (0.45), heavily penalizing the omission.")
print("3. Jensen-Shannon Divergence (JSD) provides a symmetric, well-bounded alternative [0, 1].")

# Visualizing Phoneme Distributions
x_indices = np.arange(len(phonemes))
width = 0.2

plt.figure(figsize=(10, 4.5))
plt.bar(x_indices - 1.5*width, P_true, width, label='True P', color='black', alpha=0.8)
plt.bar(x_indices - 0.5*width, Q_model_A, width, label='Model A (Good)', color='royalblue')
plt.bar(x_indices + 0.5*width, Q_model_B, width, label='Model B (Vague)', color='darkorange')
plt.bar(x_indices + 1.5*width, Q_model_C, width, label='Model C (Flawed)', color='crimson')
plt.xticks(x_indices, phonemes, fontsize=10)
plt.title("Speech Recognition Phoneme Probability Distributions", fontweight='bold')
plt.xlabel("Phoneme Class")
plt.ylabel("Probability")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("p13_speech_kl.png", dpi=150)
# plt.show()
plt.close()
print("Visualization saved as 'p13_speech_kl.png'.")

# Output:
# =======================================================
# SPEECH RECOGNITION: KULLBACK-LEIBLER (KL) DIVERGENCE
# =======================================================
# Phoneme Classes: ['/aa/', '/iy/', '/uw/', '/eh/', '/sh/', '/t/', '/k/', '/n/']
# 
# True Distribution P       : [0.45 0.25 0.12 0.08 0.04 0.03 0.02 0.01]
# Model A Distribution Q_A  : [0.42 0.26 0.14 0.07 0.05 0.03 0.02 0.01]
# Model B Distribution Q_B  : [0.15 0.15 0.15 0.15 0.1  0.1  0.1  0.1 ]
# Model C Distribution Q_C  : [0.01 0.35 0.25 0.15 0.08 0.06 0.06 0.04]
# -------------------------------------------------------
# Self-Divergence D_KL(P || P)            : 0.000000 bits (Zero information loss)
# -------------------------------------------------------
# MODEL COMPARISON TABLE (in bits):
# Model           | Forward KL D(P||Q)   | Reverse KL D(Q||P)   | JSD (Symmetric)
# -------------------------------------------------------
# Model A (Good)  |               0.0065 |               0.0067 | 0.0016
# Model B (Vague) |               0.6016 |               0.7063 | 0.1505
# Model C (Flawed) |               2.0287 |               0.8308 | 0.2490
# -------------------------------------------------------
# THEORETICAL INSIGHTS:
# 1. Asymmetry: D_KL(P || Q) != D_KL(Q || P). KL divergence is a directed divergence, not a distance metric.
# 2. Zero-Avoiding Property: D_KL(P || Q_C) is very large (2.0287 bits) because Model C assigns near zero
#    to /aa/ where true P has high probability (0.45), heavily penalizing the omission.
# 3. Jensen-Shannon Divergence (JSD) provides a symmetric, well-bounded alternative [0, 1].
# Visualization saved as 'p13_speech_kl.png'.

# ======================================================================
