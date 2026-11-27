# Program 22: Hyperparameter optimization: Disease risk classification
# Question: A healthcare organization develops a classification model to predict disease risk. Apply grid search, random search, and Bayesian optimization to identify the best hyperparameters and evaluate the effectiveness of each optimization technique. (BL5 – Evaluate)

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

np.random.seed(42)

# Generate synthetic disease risk dataset (120 patients, 5 clinical attributes)
# Attributes: [Age, Blood Pressure, Cholesterol, Fasting Glucose, BMI]
N_22 = 120
X_22 = np.random.randn(N_22, 5)
risk_score = X_22[:, 0] * 1.5 + X_22[:, 1] - X_22[:, 2] * 0.8 + np.random.normal(0, 0.5, N_22)
y_22 = (risk_score > 0).astype(int)

param_grid_22 = {
    'n_estimators': [20, 50, 100],
    'max_depth': [3, 5, 8],
    'min_samples_split': [2, 5, 10]
}
rf_base = RandomForestClassifier(random_state=42)

print("\n--- Program 22: Hyperparameter Optimization ---")

# 1. Grid Search (Exhaustive Search)
grid_search = GridSearchCV(rf_base, param_grid_22, cv=3, scoring='accuracy')
grid_search.fit(X_22, y_22)
print(f"1. Grid Search   | Best Score: {grid_search.best_score_*100:.2f}% | Best Params: {grid_search.best_params_} (Evals: 27)")

# 2. Random Search (Random Sampling)
random_search = RandomizedSearchCV(rf_base, param_grid_22, n_iter=8, cv=3, scoring='accuracy', random_state=42)
random_search.fit(X_22, y_22)
print(f"2. Random Search | Best Score: {random_search.best_score_*100:.2f}% | Best Params: {random_search.best_params_} (Evals: 8)")

# 3. Bayesian Optimization (Gaussian Process with Upper Confidence Bound)
param_combinations = []
for n_est in [20, 50, 100]:
    for md in [3, 5, 8]:
        for mss in [2, 5, 10]:
            param_combinations.append({'n_estimators': n_est, 'max_depth': md, 'min_samples_split': mss})

def evaluate_config(cfg):
    clf = RandomForestClassifier(**cfg, random_state=42)
    return cross_val_score(clf, X_22, y_22, cv=3, scoring='accuracy').mean()

# Initial design: evaluate 3 spread configurations
evaluated_pts = []
evaluated_scores = []
init_indices = [0, 13, 26]
for idx in init_indices:
    cfg = param_combinations[idx]
    score = evaluate_config(cfg)
    evaluated_pts.append([cfg['n_estimators'], cfg['max_depth'], cfg['min_samples_split']])
    evaluated_scores.append(score)

# Sequential model-based optimization using GP surrogate
gp_surrogate = GaussianProcessRegressor(kernel=RBF(length_scale=10.0), alpha=1e-3, random_state=42)

for step in range(5):
    gp_surrogate.fit(np.array(evaluated_pts), np.array(evaluated_scores))
    all_candidates = np.array([[c['n_estimators'], c['max_depth'], c['min_samples_split']] for c in param_combinations])
    mu, sigma = gp_surrogate.predict(all_candidates, return_std=True)
    # Upper Confidence Bound acquisition function (Exploration-Exploitation)
    ucb = mu + 1.96 * sigma
    best_candidate_idx = int(np.argmax(ucb))
    chosen_cfg = param_combinations[best_candidate_idx]
    chosen_pt = [chosen_cfg['n_estimators'], chosen_cfg['max_depth'], chosen_cfg['min_samples_split']]
    
    if chosen_pt not in evaluated_pts:
        score = evaluate_config(chosen_cfg)
        evaluated_pts.append(chosen_pt)
        evaluated_scores.append(score)

best_bayes_idx = int(np.argmax(evaluated_scores))
best_bayes_score = evaluated_scores[best_bayes_idx]
best_bayes_params = {
    'n_estimators': int(evaluated_pts[best_bayes_idx][0]),
    'max_depth': int(evaluated_pts[best_bayes_idx][1]),
    'min_samples_split': int(evaluated_pts[best_bayes_idx][2])
}
print(f"3. Bayesian Opt  | Best Score: {best_bayes_score*100:.2f}% | Best Params: {best_bayes_params} (Evals: {len(evaluated_pts)})")

print("\n[Optimization Method Evaluation]:")
print("- Grid Search: Guarantees global optimum across predefined grid, but cost scales exponentially (O(k^d)).")
print("- Random Search: Highly efficient for broad sweeps; finds competitive configurations with 70% fewer evaluations.")
print("- Bayesian Optimization: Uses probabilistic surrogate to direct evaluations toward promising regions with minimal budget.")

# Output:
# 1. Grid Search   | Best Score: 82.50% | Best Params: {'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 20} (Evals: 27)
# 2. Random Search | Best Score: 82.50% | Best Params: {'n_estimators': 20, 'min_samples_split': 2, 'max_depth': 5} (Evals: 8)
# 3. Bayesian Opt  | Best Score: 80.00% | Best Params: {'n_estimators': 20, 'max_depth': 3, 'min_samples_split': 2} (Evals: 5)
# [Optimization Method Evaluation]:
# - Grid Search: Guarantees global optimum across predefined grid, but cost scales exponentially (O(k^d)).
# - Random Search: Highly efficient for broad sweeps; finds competitive configurations with 70% fewer evaluations.
# - Bayesian Optimization: Uses probabilistic surrogate to direct evaluations toward promising regions with minimal budget.

# ======================================================================
