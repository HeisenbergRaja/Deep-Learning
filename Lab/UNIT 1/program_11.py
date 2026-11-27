# Program 11: Bank Loan Classification: Entropy and Uncertainty Analysis
# Question: Bank loan classification: Calculate entropy of applicant dataset, analyse uncertainty.

import numpy as np
import matplotlib.pyplot as plt

def shannon_entropy(labels):
    """
    Computes Shannon Entropy: H(S) = - sum(p_i * log2(p_i)).
    Quantifies average uncertainty in bits.
    """
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    # Small epsilon to avoid log2(0)
    return -np.sum(probabilities * np.log2(probabilities + 1e-15))

def conditional_entropy(feature_values, target_labels):
    """
    Computes Conditional Entropy: H(Y|X) = sum P(X=v) * H(Y|X=v).
    Quantifies remaining uncertainty in Y after knowing X.
    """
    total = len(feature_values)
    unique_vals, val_counts = np.unique(feature_values, return_counts=True)
    h_cond = 0.0
    for val, count in zip(unique_vals, val_counts):
        subset_target = target_labels[feature_values == val]
        weight = count / total
        h_cond += weight * shannon_entropy(subset_target)
    return h_cond

def information_gain(feature_values, target_labels):
    """
    Computes Information Gain: IG(Y, X) = H(Y) - H(Y|X).
    Measures the reduction in uncertainty about Y provided by X.
    """
    return shannon_entropy(target_labels) - conditional_entropy(feature_values, target_labels)

# --- Bank Loan Applicant Dataset (20 sample applicants) ---
credit_score = np.array(['Good', 'Fair', 'Poor', 'Good', 'Good', 'Fair', 'Poor', 'Fair',
                         'Good', 'Poor', 'Good', 'Fair', 'Fair', 'Poor', 'Good', 'Fair',
                         'Poor', 'Good', 'Fair', 'Poor'])
income_level = np.array(['High', 'Medium', 'Low', 'Medium', 'High', 'Low', 'Low', 'Medium',
                        'High', 'Low', 'Medium', 'Medium', 'High', 'Low', 'Low', 'High',
                        'Medium', 'High', 'Low', 'Medium'])
employment = np.array(['Employed', 'Employed', 'Unemployed', 'Self-Emp', 'Employed', 'Unemployed',
                      'Unemployed', 'Self-Emp', 'Employed', 'Unemployed', 'Self-Emp', 'Employed',
                      'Employed', 'Unemployed', 'Employed', 'Self-Emp', 'Unemployed', 'Employed',
                      'Self-Emp', 'Unemployed'])
loan_status = np.array(['Approved', 'Approved', 'Rejected', 'Approved', 'Approved', 'Rejected',
                       'Rejected', 'Approved', 'Approved', 'Rejected', 'Approved', 'Approved',
                       'Approved', 'Rejected', 'Rejected', 'Approved', 'Rejected', 'Approved',
                       'Rejected', 'Rejected'])

# 1. Base Target Entropy
h_target = shannon_entropy(loan_status)
n_approved = np.sum(loan_status == 'Approved')
n_rejected = np.sum(loan_status == 'Rejected')

features = {
    'Credit Score': credit_score,
    'Income Level': income_level,
    'Employment Status': employment
}

print("=" * 55)
print("BANK LOAN CLASSIFICATION: ENTROPY & UNCERTAINTY ANALYSIS")
print("=" * 55)
print(f"Total Applicants Evaluated : {len(loan_status)} (Approved: {n_approved}, Rejected: {n_rejected})")
print(f"Base Target Entropy H(Y)   : {h_target:.4f} bits")
print(f"Theoretical Maximum Entropy: {1.0000:.4f} bits (Equally likely 50/50 binary split)")
print(f"Relative Uncertainty Level : {(h_target / 1.0) * 100:.1f}%")
print("-" * 55)
print(f"{'Feature Name':<20} | {'H(Y|X) [bits]':<14} | {'Info Gain [bits]':<16} | {'Uncertainty Reduction'}")
print("-" * 55)

best_feature = None
max_ig = -1.0
ig_results = {}

for name, feat_data in features.items():
    h_cond = conditional_entropy(feat_data, loan_status)
    ig = information_gain(feat_data, loan_status)
    ig_results[name] = ig
    reduction_pct = (ig / h_target) * 100
    print(f"{name:<20} | {h_cond:14.4f} | {ig:16.4f} | {reduction_pct:.1f}%")
    if ig > max_ig:
        max_ig = ig
        best_feature = name

print("-" * 55)
print("UNCERTAINTY ANALYSIS CONCLUSION:")
print(f"Feature '{best_feature}' achieves the highest Information Gain ({max_ig:.4f} bits),")
print(f"reducing applicant loan decision uncertainty by {(max_ig / h_target)*100:.1f}%.")
print("In a Decision Tree classifier, this feature is selected as the optimal root split.")

# Bar plot of Information Gain
plt.figure(figsize=(7.5, 4.2))
bars = plt.bar(list(ig_results.keys()), list(ig_results.values()), color=['royalblue', 'darkorange', 'seagreen'], width=0.5)
plt.axhline(y=h_target, color='r', linestyle='--', label=f'Base Entropy H(Y) = {h_target:.4f} bits')
plt.title("Information Gain (Uncertainty Reduction) Across Loan Features", fontweight='bold')
plt.ylabel("Information Gain (bits)")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f} bits", ha='center', va='bottom', fontsize=9)
plt.ylim(0, 1.15)
plt.legend()
plt.tight_layout()
plt.savefig("p11_loan_entropy.png", dpi=150)
# plt.show()
plt.close()
print("Visualization saved as 'p11_loan_entropy.png'.")

# Output:
# =======================================================
# BANK LOAN CLASSIFICATION: ENTROPY & UNCERTAINTY ANALYSIS
# =======================================================
# Total Applicants Evaluated : 20 (Approved: 11, Rejected: 9)
# Base Target Entropy H(Y)   : 0.9928 bits
# Theoretical Maximum Entropy: 1.0000 bits (Equally likely 50/50 binary split)
# Relative Uncertainty Level : 99.3%
# -------------------------------------------------------
# Feature Name         | H(Y|X) [bits]  | Info Gain [bits] | Uncertainty Reduction
# -------------------------------------------------------
# Credit Score         |         0.5092 |           0.4836 | 48.7%
# Income Level         |         0.3021 |           0.6907 | 69.6%
# Employment Status    |         0.3979 |           0.5949 | 59.9%
# -------------------------------------------------------
# UNCERTAINTY ANALYSIS CONCLUSION:
# Feature 'Income Level' achieves the highest Information Gain (0.6907 bits),
# reducing applicant loan decision uncertainty by 69.6%.
# In a Decision Tree classifier, this feature is selected as the optimal root split.
# Visualization saved as 'p11_loan_entropy.png'.

# ======================================================================
