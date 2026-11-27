# Program 80: Medical Deep Learning Application for Health Risk Prediction
# Question: Design deep learning application for medical data analysis (health prediction using neural network), evaluate with metrics.
# ======================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

torch.manual_seed(42)
np.random.seed(42)

n_patients = 1000

# 1. Synthesize patient medical records with 8 clinical biomarkers:
# Age (30-75), Resting BP (90-180), Cholesterol (140-320), Fasting Blood Sugar (70-200),
# Max Heart Rate (80-190), BMI (18-40), Exercise Induced Angina (0 or 1), ST Depression (0-4.5)
age = np.random.uniform(30, 75, n_patients)
bp = np.random.uniform(90, 180, n_patients)
chol = np.random.uniform(140, 320, n_patients)
fbs = np.random.uniform(70, 200, n_patients)
max_hr = np.random.uniform(80, 190, n_patients)
bmi = np.random.uniform(18, 40, n_patients)
angina = np.random.binomial(1, 0.35, n_patients)
st_dep = np.random.uniform(0.0, 4.5, n_patients)

# Realistic non-linear clinical cardiovascular risk formula
risk_score = (0.04 * (age - 50) + 
              0.02 * (bp - 120) + 
              0.015 * (chol - 200) + 
              0.01 * (fbs - 100) - 
              0.025 * (max_hr - 140) + 
              0.05 * (bmi - 25) + 
              0.8 * angina + 
              0.5 * st_dep)

# Map to probability and generate diagnostic ground truth (0 = Low Risk, 1 = High Cardiovascular Risk)
true_prob = 1.0 / (1.0 + np.exp(-risk_score))
y = (true_prob >= 0.5).astype(np.float32)

X = np.c_[age, bp, chol, fbs, max_hr, bmi, angina, st_dep]
feature_names = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "BMI", "Angina", "STDepression"]

# 2. Train/Test split (80% train, 20% test) and feature standardization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch Tensors
train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), 
                              torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# 3. Define Deep Neural Network architecture for health risk classification
class MedicalHealthRiskNet(nn.Module):
    def __init__(self, input_features):
        super(MedicalHealthRiskNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1) # Logit output
        )
        
    def forward(self, x):
        return self.network(x)

model = MedicalHealthRiskNet(input_features=8)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

# 4. Train the neural network
epochs = 35
model.train()
for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

# 5. Evaluate the model on test dataset
model.eval()
with torch.no_grad():
    test_logits = model(X_test_tensor)
    pred_probabilities = torch.sigmoid(test_logits).numpy().ravel()
    binary_predictions = (pred_probabilities >= 0.5).astype(int)

# 6. Compute Comprehensive Evaluation Metrics
accuracy = accuracy_score(y_test, binary_predictions)
precision = precision_score(y_test, binary_predictions)
recall = recall_score(y_test, binary_predictions)
f1 = f1_score(y_test, binary_predictions)
roc_auc = roc_auc_score(y_test, pred_probabilities)
cm = confusion_matrix(y_test, binary_predictions)

print("--- Program 80: Medical Deep Learning Health Risk Prediction ---")
print("Evaluation Metrics on Held-Out Test Set (N = 200 patients):")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print("\nConfusion Matrix:")
print("                 Predicted Negative  Predicted Positive")
print(f"Actual Negative        {cm[0, 0]:<19} {cm[0, 1]}")
print(f"Actual Positive        {cm[1, 0]:<19} {cm[1, 1]}")

print("\nPersonalized Clinical Risk Assessment on Sample Patients:")
for p_idx in [0, 15, 45]:
    raw = X_test[p_idx]
    prob = pred_probabilities[p_idx]
    status = "HIGH RISK (Positive)" if prob >= 0.5 else "LOW RISK (Negative)"
    print(f"Patient [Age: {raw[0]:.0f}, BP: {raw[1]:.0f}, Chol: {raw[2]:.0f}, MaxHR: {raw[4]:.0f}, Angina: {int(raw[6])}]")
    print(f"  -> Predicted Disease Probability: {prob*100:.1f}% | Diagnosis: {status}")

# Output:
# --- Program 80: Medical Deep Learning Health Risk Prediction ---
# Evaluation Metrics on Held-Out Test Set (N = 200 patients):
# Accuracy:  0.9900
# Precision: 0.9948
# Recall:    0.9948
# F1-Score:  0.9948
# ROC-AUC:   0.9963
#
# Confusion Matrix:
#                  Predicted Negative  Predicted Positive
# Actual Negative        6                   1
# Actual Positive        1                   192
#
# Personalized Clinical Risk Assessment on Sample Patients:
# Patient [Age: 70, BP: 147, Chol: 263, MaxHR: 139, Angina: 1]
#   -> Predicted Disease Probability: 100.0% | Diagnosis: HIGH RISK (Positive)
# Patient [Age: 62, BP: 151, Chol: 219, MaxHR: 99, Angina: 1]
#   -> Predicted Disease Probability: 100.0% | Diagnosis: HIGH RISK (Positive)
# Patient [Age: 53, BP: 142, Chol: 251, MaxHR: 172, Angina: 0]
#   -> Predicted Disease Probability: 99.8% | Diagnosis: HIGH RISK (Positive)
#
# Output comments:
# The PyTorch neural network achieves 99.00% accuracy and 0.9963 ROC-AUC on the medical dataset.
# The combination of Batch Normalization, Dropout, and non-linear ReLU activations allows
# the multi-layer perceptron to model complex clinical interactions and output calibrated risk scores.
