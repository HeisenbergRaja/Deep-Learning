# Program 24: Single-layer perceptron: Bank loan classification (income, credit score)
# Question: A bank wants to classify loan applicants as eligible or ineligible based on income and credit score. Implement a single-layer perceptron and demonstrate its ability to perform binary classification. (BL3 – Apply)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(42)

# Generate synthetic bank applicant records (100 applicants)
# Features: [Annual Income in $k (25-110), Credit Score (350-850)]
N_24 = 100
income_k = np.random.uniform(25.0, 110.0, N_24)
credit_val = np.random.uniform(350.0, 850.0, N_24)

# Eligibility boundary: 0.03 * income + 0.015 * credit - 12.0 >= 0
linear_comb = 0.03 * income_k + 0.015 * credit_val - 12.0
y_loan_eligible = (linear_comb >= 0).astype(int)

X_loan = np.column_stack([income_k, credit_val])
X_tr24, X_te24, y_tr24, y_te24 = train_test_split(X_loan, y_loan_eligible, test_size=0.30, random_state=42)

# Standardize inputs for fast, stable perceptron convergence
mean_24 = np.mean(X_tr24, axis=0)
std_24 = np.std(X_tr24, axis=0)
X_tr24_s = (X_tr24 - mean_24) / std_24
X_te24_s = (X_te24 - mean_24) / std_24

# Single-Layer Perceptron Implementation (Rosenblatt Perceptron)
class SingleLayerPerceptron:
    def __init__(self, learning_rate=0.1, max_epochs=100):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.w = None
        self.b = 0.0
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        for epoch in range(1, self.max_epochs + 1):
            misclassified = 0
            for i in range(n_samples):
                # Linear activation: z = w^T * x + b
                z = np.dot(X[i], self.w) + self.b
                # Heaviside unit step function
                y_hat = 1 if z >= 0 else 0
                error = y[i] - y_hat
                
                if error != 0:
                    self.w += self.lr * error * X[i]
                    self.b += self.lr * error
                    misclassified += 1
            if misclassified == 0:
                print(f"Perceptron converged perfectly at epoch {epoch} with 0 errors!")
                break
                
    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return np.where(z >= 0, 1, 0)

print("\n--- Program 24: Single-Layer Perceptron ---")
slp = SingleLayerPerceptron(learning_rate=0.1, max_epochs=100)
slp.fit(X_tr24_s, y_tr24)

# Evaluate on unseen testing set
test_predictions = slp.predict(X_te24_s)
test_accuracy = accuracy_score(y_te24, test_predictions) * 100
conf_mat = confusion_matrix(y_te24, test_predictions)

print(f"\nModel Parameters (Standardized): w1 = {slp.w[0]:.4f}, w2 = {slp.w[1]:.4f}, bias = {slp.b:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print("Confusion Matrix:")
print(f"  [[TN={conf_mat[0,0]}  FP={conf_mat[0,1]}]\n   [FN={conf_mat[1,0]}  TP={conf_mat[1,1]}]]")

# Classify specific new loan applicants
sample_applicants = np.array([
    [95.0, 780.0],  # High income, excellent credit -> Eligible
    [30.0, 420.0],  # Low income, poor credit       -> Ineligible
    [45.0, 750.0],  # Moderate income, high credit  -> Eligible
    [85.0, 460.0]   # High income, low credit       -> Ineligible
])
sample_std = (sample_applicants - mean_24) / std_24
predictions = slp.predict(sample_std)

print("\nSample Applicant Predictions:")
for app, pred in zip(sample_applicants, predictions):
    decision = "ELIGIBLE (Approved)" if pred == 1 else "INELIGIBLE (Rejected)"
    print(f"  Income: ${app[0]:5.1f}k, Credit Score: {app[1]:3.0f} -> {decision}")

# Output:
# Perceptron converged perfectly at epoch 4 with 0 errors!
# Model Parameters (Standardized): w1 = 0.1013, w2 = 0.4120, bias = -0.2000
# Test Accuracy: 100.00%
# Confusion Matrix:
#   [[TN=17  FP=0]
#    [FN=0  TP=13]]
# Sample Applicant Predictions:
#   Income: $ 95.0k, Credit Score: 780 -> ELIGIBLE (Approved)
#   Income: $ 30.0k, Credit Score: 420 -> INELIGIBLE (Rejected)
#   Income: $ 45.0k, Credit Score: 750 -> ELIGIBLE (Approved)
#   Income: $ 85.0k, Credit Score: 460 -> INELIGIBLE (Rejected)




# ======================================================================
# CO2 PROGRAMS 25 TO 32
# ======================================================================

# ======================================================================
