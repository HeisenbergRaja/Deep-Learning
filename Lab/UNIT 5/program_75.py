# Program 75: Factor Analysis on Student Academic Performance
# Question: Student marks, attendance, assignments data. Apply factor analysis to identify latent factors influencing performance.
# ======================================================================

import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n_students = 250

# 1. Simulate 3 underlying latent factors:
# F1 = STEM Aptitude, F2 = Humanities Aptitude, F3 = Conscientiousness / Diligence
f_stem = np.random.normal(0, 1, n_students)
f_hum = np.random.normal(0, 1, n_students)
f_dil = np.random.normal(0, 1, n_students)

# 2. Generate observed academic metrics influenced by the latent factors + noise
math = 65 + 15 * f_stem + 5 * f_dil + np.random.normal(0, 4, n_students)
physics = 62 + 14 * f_stem + 4 * f_dil + np.random.normal(0, 4, n_students)
compsci = 68 + 13 * f_stem + 6 * f_dil + np.random.normal(0, 4, n_students)
literature = 70 + 16 * f_hum + 5 * f_dil + np.random.normal(0, 4, n_students)
history = 67 + 14 * f_hum + 6 * f_dil + np.random.normal(0, 4, n_students)
attendance = 80 + 2 * f_stem + 2 * f_hum + 12 * f_dil + np.random.normal(0, 3, n_students)
assignments = 78 + 3 * f_stem + 3 * f_hum + 13 * f_dil + np.random.normal(0, 3, n_students)

X_students = np.c_[math, physics, compsci, literature, history, attendance, assignments]
var_names = ["Math", "Physics", "ComputerSc", "Literature", "History", "Attendance", "Assignments"]

# 3. Standardize variables (zero mean, unit variance)
scaler = StandardScaler()
X_std = scaler.fit_transform(X_students)

# 4. Apply Factor Analysis with 3 latent factors
fa = FactorAnalysis(n_components=3, random_state=42)
factor_scores = fa.fit_transform(X_std)
loadings = fa.components_.T # Shape: (n_features, n_factors)

# Communality (variance accounted for by common factors) and Uniqueness (specific variance)
communalities = np.sum(loadings**2, axis=1)
uniquenesses = fa.noise_variance_

print("--- Program 75: Factor Analysis on Student Performance ---")
print("Factor Loadings Matrix (Variables x Latent Factors):")
print(f"{'Variable':<14} {'Factor 1':<10} {'Factor 2':<10} {'Factor 3':<10} {'Communality':<12} {'Uniqueness'}")
print("-" * 68)
for i, name in enumerate(var_names):
    print(f"{name:<14} {loadings[i,0]:10.3f} {loadings[i,1]:10.3f} {loadings[i,2]:10.3f} {communalities[i]:12.3f} {uniquenesses[i]:10.3f}")

# Factor Variance Contributions
factor_variances = np.sum(loadings**2, axis=0)
total_var = len(var_names)
print("\nFactor Variance Contributions:")
for j in range(3):
    pct = (factor_variances[j] / total_var) * 100
    print(f"Factor {j+1}: Variance = {factor_variances[j]:.3f} ({pct:.1f}% of total variance)")

print("\nLatent Factor Interpretation:")
print("Factor 1: Academic Diligence & Overall Discipline (strongly loaded on Attendance, Assignments, Coursework)")
print("Factor 2: STEM vs Humanities Contrast (positive on Math/Physics/CS, negative on Literature/History)")
print("Factor 3: Humanities Verbal Fluency (positive loading on Literature, History)")

print("\nSample Student Factor Scores (Latent Profiles):")
for s_idx in range(3):
    scores_str = ", ".join([f"F{k+1}: {factor_scores[s_idx, k]:.2f}" for k in range(3)])
    print(f"Student {s_idx+1} Raw Marks [Math: {math[s_idx]:.1f}, Lit: {literature[s_idx]:.1f}, Att: {attendance[s_idx]:.1f}%] -> Factor Scores: ({scores_str})")

# Output:
# --- Program 75: Factor Analysis on Student Performance ---
# Factor Loadings Matrix (Variables x Latent Factors):
# Variable       Factor 1   Factor 2   Factor 3   Communality  Uniqueness
# --------------------------------------------------------------------
# Math               -0.708      0.620      0.238        0.941      0.059
# Physics            -0.712      0.610      0.229        0.932      0.068
# ComputerSc         -0.783      0.539      0.155        0.927      0.073
# Literature         -0.592     -0.684      0.355        0.944      0.056
# History            -0.641     -0.694      0.235        0.947      0.052
# Attendance         -0.872     -0.124     -0.422        0.954      0.047
# Assignments        -0.907     -0.098     -0.337        0.945      0.054
#
# Factor Variance Contributions:
# Factor 1: Variance = 3.964 (56.6% of total variance)
# Factor 2: Variance = 2.020 (28.9% of total variance)
# Factor 3: Variance = 0.606 (8.7% of total variance)
#
# Latent Factor Interpretation:
# Factor 1: Academic Diligence & Overall Discipline (strongly loaded on Attendance, Assignments, Coursework)
# Factor 2: STEM vs Humanities Contrast (positive on Math/Physics/CS, negative on Literature/History)
# Factor 3: Humanities Verbal Fluency (positive loading on Literature, History)
#
# Sample Student Factor Scores (Latent Profiles):
# Student 1 Raw Marks [Math: 72.3, Lit: 65.7, Att: 78.4%] -> Factor Scores: (F1: -0.28, F2: 0.22, F3: -0.26)
# Student 2 Raw Marks [Math: 64.9, Lit: 67.5, Att: 77.2%] -> Factor Scores: (F1: 0.11, F2: 0.08, F3: -0.19)
# Student 3 Raw Marks [Math: 79.7, Lit: 81.3, Att: 89.2%] -> Factor Scores: (F1: -1.33, F2: -0.21, F3: 0.44)
#
# Output comments:
# High communalities (> 0.92) across all observed variables indicate that the 3 latent factors
# capture over 94% of the shared variance in academic performance, successfully decomposing observed
# marks and attendance into underlying diligence, quantitative aptitude, and verbal ability.

# ======================================================================
