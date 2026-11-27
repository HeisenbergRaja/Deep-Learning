# Program 8: Manufacturing
# Question: Two machines with different defect rates, calculate conditional probability of defective product source.

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Problem Formulation & Given Probabilities
# ----------------------------------------------------------------------
# Production proportions (Prior Probabilities):
# Machine 1 produces 65% of total parts -> P(M1) = 0.65
# Machine 2 produces 35% of total parts -> P(M2) = 0.35
p_m1 = 0.65
p_m2 = 0.35

# Conditional Defect Rates:
# Machine 1 defect rate: P(Defect | M1) = 0.02 (2.0%)
# Machine 2 defect rate: P(Defect | M2) = 0.08 (8.0%)
p_defect_given_m1 = 0.02
p_defect_given_m2 = 0.08

# ----------------------------------------------------------------------
# 2. Analytical Solution via Bayes' Theorem
# ----------------------------------------------------------------------
# Total Probability of selecting a defective product: P(D)
# Formula: P(D) = P(D | M1)*P(M1) + P(D | M2)*P(M2)
p_defect_and_m1 = p_defect_given_m1 * p_m1
p_defect_and_m2 = p_defect_given_m2 * p_m2
p_defect_total = p_defect_and_m1 + p_defect_and_m2

# Bayes' Theorem: Conditional probability of originating machine given item is defective
# Formula: P(Mi | D) = [P(D | Mi) * P(Mi)] / P(D)
p_m1_given_defect = p_defect_and_m1 / p_defect_total
p_m2_given_defect = p_defect_and_m2 / p_defect_total

print("=" * 70)
print("MANUFACTURING: BAYESIAN DEFECT SOURCE PROBABILITY")
print("=" * 70)
print("1. PRIOR PRODUCTION PROBABILITIES & DEFECT RATES:")
print(f"   P(Machine 1)             : {p_m1:.4f} (65.00% of factory output)")
print(f"   P(Machine 2)             : {p_m2:.4f} (35.00% of factory output)")
print(f"   P(Defect | Machine 1)    : {p_defect_given_m1:.4f} (2.00% defect rate)")
print(f"   P(Defect | Machine 2)    : {p_defect_given_m2:.4f} (8.00% defect rate)")
print("-" * 70)
print("2. TOTAL PROBABILITY OF DEFECTIVE ITEM:")
print(f"   P(Defect) = P(D|M1)*P(M1) + P(D|M2)*P(M2) = {p_defect_total:.4f} ({p_defect_total*100:.2f}%)")
print("-" * 70)
print("3. POSTERIOR CONDITIONAL PROBABILITIES (BAYES' THEOREM):")
print(f"   P(Machine 1 | Defect)    : {p_m1_given_defect:.4f} ({p_m1_given_defect*100:.2f}%)")
print(f"   P(Machine 2 | Defect)    : {p_m2_given_defect:.4f} ({p_m2_given_defect*100:.2f}%)")

# ----------------------------------------------------------------------
# 3. Empirical Verification via Monte Carlo Simulation (Inline Data)
# ----------------------------------------------------------------------
np.random.seed(42)
n_samples = 1_000_000

# Step 1: Assign machine source based on prior distribution
machine_source = np.random.choice([1, 2], size=n_samples, p=[p_m1, p_m2])

# Step 2: Assign defect status based on machine-specific defect probabilities
random_draws = np.random.rand(n_samples)
is_defective = np.where(
    machine_source == 1,
    random_draws < p_defect_given_m1,
    random_draws < p_defect_given_m2
)

# Step 3: Count defective items and compute empirical probabilities
total_defects = np.sum(is_defective)
defects_from_m1 = np.sum(is_defective & (machine_source == 1))
defects_from_m2 = np.sum(is_defective & (machine_source == 2))

emp_p_defect = total_defects / n_samples
emp_p_m1_given_defect = defects_from_m1 / total_defects
emp_p_m2_given_defect = defects_from_m2 / total_defects

print("-" * 70)
print(f"4. MONTE CARLO SIMULATION VERIFICATION ({n_samples:,} UNITS):")
print(f"   Simulated P(Defect)           : {emp_p_defect:.4f} ({emp_p_defect*100:.2f}%)")
print(f"   Simulated P(Machine 1 | Defect): {emp_p_m1_given_defect:.4f} ({emp_p_m1_given_defect*100:.2f}%)")
print(f"   Simulated P(Machine 2 | Defect): {emp_p_m2_given_defect:.4f} ({emp_p_m2_given_defect*100:.2f}%)")
print("-" * 70)
print("5. QUALITY CONTROL DIAGNOSIS:")
print(f"   Even though Machine 1 produces nearly double the parts of Machine 2 (65% vs 35%),")
print(f"   a randomly discovered defective part is MORE LIKELY ({p_m2_given_defect*100:.2f}%) to have come")
print(f"   from Machine 2 due to its 4x higher defect rate (8% vs 2%).")
print("=" * 70)

# ----------------------------------------------------------------------
# 6. Prior vs Posterior Comparison Visualization using Matplotlib
# ----------------------------------------------------------------------
machines = ['Machine 1', 'Machine 2']
x = np.arange(len(machines))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
rects1 = ax.bar(x - width/2, [p_m1, p_m2], width, label='Prior P(M)', color='cornflowerblue', edgecolor='black')
rects2 = ax.bar(x + width/2, [p_m1_given_defect, p_m2_given_defect], width, label='Posterior P(M | Defect)', color='salmon', edgecolor='black')

ax.set_ylabel('Probability')
ax.set_title('Bayesian Shift: Prior Production vs Posterior Defect Probability')
ax.set_xticks(x)
ax.set_xticklabels(machines)
ax.legend()

for rects in [rects1, rects2]:
    for bar in rects:
        height = bar.get_height()
        ax.annotate(f'{height:.2%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
# plt.show() # Display plot window (uncomment when running interactively)
plt.close()

# Output:
# ======================================================================
# MANUFACTURING: BAYESIAN DEFECT SOURCE PROBABILITY
# ======================================================================
# 1. PRIOR PRODUCTION PROBABILITIES & DEFECT RATES:
#    P(Machine 1)             : 0.6500 (65.00% of factory output)
#    P(Machine 2)             : 0.3500 (35.00% of factory output)
#    P(Defect | Machine 1)    : 0.0200 (2.00% defect rate)
#    P(Defect | Machine 2)    : 0.0800 (8.00% defect rate)
# ----------------------------------------------------------------------
# 2. TOTAL PROBABILITY OF DEFECTIVE ITEM:
#    P(Defect) = P(D|M1)*P(M1) + P(D|M2)*P(M2) = 0.0410 (4.10%)
# ----------------------------------------------------------------------
# 3. POSTERIOR CONDITIONAL PROBABILITIES (BAYES' THEOREM):
#    P(Machine 1 | Defect)    : 0.3171 (31.71%)
#    P(Machine 2 | Defect)    : 0.6829 (68.29%)
# ----------------------------------------------------------------------
# 4. MONTE CARLO SIMULATION VERIFICATION (1,000,000 UNITS):
#    Simulated P(Defect)           : 0.0413 (4.13%)
#    Simulated P(Machine 1 | Defect): 0.3144 (31.44%)
#    Simulated P(Machine 2 | Defect): 0.6856 (68.56%)
# ----------------------------------------------------------------------
# 5. QUALITY CONTROL DIAGNOSIS:
#    Even though Machine 1 produces nearly double the parts of Machine 2 (65% vs 35%),
#    a randomly discovered defective part is MORE LIKELY (68.29%) to have come
#    from Machine 2 due to its 4x higher defect rate (8% vs 2%).
# ======================================================================



# ======================================================================
# CO1 PROGRAMS 9 TO 16
# ======================================================================
