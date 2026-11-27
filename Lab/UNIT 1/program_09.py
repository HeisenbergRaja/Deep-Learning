# Program 9: Hospital Diagnostic: Bayes' Theorem Application
# Question: Hospital diagnostic: Apply Bayes' theorem - given disease prevalence, sensitivity, specificity, calculate P(disease|positive test).

import numpy as np
import matplotlib.pyplot as plt

def bayes_disease_diagnostic(prevalence, sensitivity, specificity):
    """
    Applies Bayes' Theorem to compute posterior probabilities for medical diagnosis.
    
    Parameters:
        prevalence (float): Prior probability of having the disease P(D).
        sensitivity (float): True Positive Rate P(+|D).
        specificity (float): True Negative Rate P(-|~D).
        
    Returns:
        dict: Intermediate and posterior diagnostic probabilities.
    """
    # Prior probabilities
    p_disease = prevalence
    p_no_disease = 1.0 - prevalence
    
    # Conditional test probabilities (Likelihoods)
    p_pos_given_disease = sensitivity                   # Sensitivity (True Positive Rate)
    p_neg_given_disease = 1.0 - sensitivity             # False Negative Rate
    p_neg_given_no_disease = specificity               # Specificity (True Negative Rate)
    p_pos_given_no_disease = 1.0 - specificity         # False Positive Rate
    
    # Law of Total Probability for Evidence P(+) and P(-)
    # P(+) = P(+|D) * P(D) + P(+|~D) * P(~D)
    p_pos = (p_pos_given_disease * p_disease) + (p_pos_given_no_disease * p_no_disease)
    # P(-) = P(-|D) * P(D) + P(-|~D) * P(~D)
    p_neg = (p_neg_given_disease * p_disease) + (p_neg_given_no_disease * p_no_disease)
    
    # Bayes' Theorem for Posterior Probabilities:
    # P(D|+) = [P(+|D) * P(D)] / P(+)   (Positive Predictive Value - PPV)
    p_disease_given_pos = (p_pos_given_disease * p_disease) / p_pos
    
    # P(~D|-) = [P(-|~D) * P(~D)] / P(-) (Negative Predictive Value - NPV)
    p_no_disease_given_neg = (p_neg_given_no_disease * p_no_disease) / p_neg
    
    return {
        'P(D)': p_disease,
        'P(~D)': p_no_disease,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'FPR': p_pos_given_no_disease,
        'P(+)': p_pos,
        'P(-)': p_neg,
        'P(D|+)': p_disease_given_pos,
        'P(~D|-)': p_no_disease_given_neg
    }

# --- Clinical Sample Scenario ---
# Rare disease affecting 1% of the population
# Diagnostic test with 95% sensitivity and 90% specificity
prevalence_val = 0.01     # 1% prevalence
sensitivity_val = 0.95    # 95% sensitivity
specificity_val = 0.90    # 90% specificity

results = bayes_disease_diagnostic(prevalence_val, sensitivity_val, specificity_val)

print("=" * 55)
print("EXPERIMENT 9: HOSPITAL DIAGNOSTIC - BAYES' THEOREM")
print("=" * 55)
print(f"Prior Probability / Disease Prevalence P(D)      : {results['P(D)']:.4f} ({results['P(D)']*100:.2f}%)")
print(f"Test Sensitivity P(+|D) [True Positive Rate]     : {results['Sensitivity']:.4f} ({results['Sensitivity']*100:.2f}%)")
print(f"Test Specificity P(-|~D) [True Negative Rate]    : {results['Specificity']:.4f} ({results['Specificity']*100:.2f}%)")
print(f"False Positive Rate P(+|~D) [1 - Specificity]    : {results['FPR']:.4f} ({results['FPR']*100:.2f}%)")
print(f"Total Probability of Testing Positive P(+)       : {results['P(+)']:.4f} ({results['P(+)']*100:.2f}%)")
print("-" * 55)
print(f"POSTERIOR P(Disease | Positive Test) [PPV]       : {results['P(D|+)']:.4f} ({results['P(D|+)']*100:.2f}%)")
print(f"POSTERIOR P(No Disease | Negative Test) [NPV]    : {results['P(~D|-)']:.4f} ({results['P(~D|-)']*100:.2f}%)")
print("-" * 55)
print("CLINICAL INSIGHT (Base Rate Fallacy):")
print(f"Even with a 95% accurate test, because the base disease rate is rare (1%),")
print(f"a positive result corresponds to only a {results['P(D|+)']*100:.2f}% chance of actual infection.")
print(f"Out of every 10,000 people tested:")
print(f"  - True Positives  : {10000 * prevalence_val * sensitivity_val:.0f}")
print(f"  - False Positives : {10000 * (1 - prevalence_val) * results['FPR']:.0f}")

# --- Sensitivity Analysis: Effect of Disease Prevalence on P(D|+) ---
prev_range = np.linspace(0.001, 0.25, 200)
ppv_values = [bayes_disease_diagnostic(p, sensitivity_val, specificity_val)['P(D|+)'] * 100 for p in prev_range]

plt.figure(figsize=(8, 4.5))
plt.plot(prev_range * 100, ppv_values, color='darkblue', lw=2.2, label="P(Disease | Positive Test)")
plt.axvline(x=prevalence_val * 100, color='red', linestyle='--', label=f"Current Baseline (1% -> {results['P(D|+)']*100:.2f}%)")
plt.title("Bayes' Theorem: P(Disease | Positive) vs. Disease Prevalence", fontsize=12, fontweight='bold')
plt.xlabel("Disease Prevalence in Population (%)", fontsize=10)
plt.ylabel("P(Disease | Positive) (%)", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("p9_hospital_bayes.png", dpi=150)
# plt.show()
plt.close()
print("Visualization saved as 'p9_hospital_bayes.png'.")

# Output:
# =======================================================
# EXPERIMENT 9: HOSPITAL DIAGNOSTIC - BAYES' THEOREM
# =======================================================
# Prior Probability / Disease Prevalence P(D)      : 0.0100 (1.00%)
# Test Sensitivity P(+|D) [True Positive Rate]     : 0.9500 (95.00%)
# Test Specificity P(-|~D) [True Negative Rate]    : 0.9000 (90.00%)
# False Positive Rate P(+|~D) [1 - Specificity]    : 0.1000 (10.00%)
# Total Probability of Testing Positive P(+)       : 0.1085 (10.85%)
# -------------------------------------------------------
# POSTERIOR P(Disease | Positive Test) [PPV]       : 0.0876 (8.76%)
# POSTERIOR P(No Disease | Negative Test) [NPV]    : 0.9994 (99.94%)
# -------------------------------------------------------
# CLINICAL INSIGHT (Base Rate Fallacy):
# Even with a 95% accurate test, because the base disease rate is rare (1%),
# a positive result corresponds to only a 8.76% chance of actual infection.
# Out of every 10,000 people tested:
#   - True Positives  : 95
#   - False Positives : 990
# Visualization saved as 'p9_hospital_bayes.png'.

# ======================================================================
