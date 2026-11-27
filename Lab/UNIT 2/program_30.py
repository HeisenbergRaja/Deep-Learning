# Program 30: Backpropagation
# Question: Implement backpropagation algorithm to calculate gradients and update weights for digit recognition.
# ======================================================================

import numpy as np

# ----------------------------------------------------------------------
# 1. Dataset Generation: 8x8 Digit Images (64 Features, 10 Classes)
# ----------------------------------------------------------------------
# Benchmark digit setup:
# Input: X in R^{64} (representing flattened 8x8 pixel intensities)
# Output: Y in {0, 1, ..., 9} (10 digit categories)

np.random.seed(42)

n_samples = 600
n_features = 64
n_hidden = 32
n_classes = 10

X_raw = np.random.randn(n_samples, n_features).astype(np.float32)
# Create distinctive signature patterns for each digit class
class_signatures = np.random.randn(n_classes, n_features).astype(np.float32) * 2.5
y_labels = np.random.randint(0, n_classes, size=n_samples)
for i in range(n_samples):
    X_raw[i] += class_signatures[y_labels[i]]

# Standardize inputs
X_data = (X_raw - X_raw.mean(axis=0)) / (X_raw.std(axis=0) + 1e-8)

# One-hot encoding of labels
Y_onehot = np.zeros((n_samples, n_classes), dtype=np.float32)
Y_onehot[np.arange(n_samples), y_labels] = 1.0

# Split into 80% Train (480) and 20% Test (120)
split = 480
X_train, Y_train, y_train = X_data[:split], Y_onehot[:split], y_labels[:split]
X_test, Y_test, y_test = X_data[split:], Y_onehot[split:], y_labels[split:]

# ----------------------------------------------------------------------
# 2. Mathematical Backpropagation Setup
# ----------------------------------------------------------------------
# Xavier / He Weight Initialization
W1 = np.random.randn(n_features, n_hidden).astype(np.float32) * np.sqrt(2.0 / n_features)
b1 = np.zeros((1, n_hidden), dtype=np.float32)

W2 = np.random.randn(n_hidden, n_classes).astype(np.float32) * np.sqrt(2.0 / n_hidden)
b2 = np.zeros((1, n_classes), dtype=np.float32)

def relu(z):
    return np.maximum(0.0, z)

def relu_gradient(z):
    return (z > 0.0).astype(np.float32)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

learning_rate = 0.08
num_epochs = 100
m_train = len(X_train)

print(f"Digit Recognition: {n_samples} samples | 64 Input Features | 10 Classes")
print(f"Network Architecture: [64 Inputs] -> [32 Hidden (ReLU)] -> [10 Output (Softmax)]")
print("\n--- Training with Backpropagation Algorithm ---")

# ----------------------------------------------------------------------
# 3. Training Loop with Analytical Gradient Computation
# ----------------------------------------------------------------------
for epoch in range(1, num_epochs + 1):
    # ==========================
    # FORWARD PASS:
    # ==========================
    Z1 = np.dot(X_train, W1) + b1        # [m, 32]
    A1 = relu(Z1)                       # [m, 32]
    Z2 = np.dot(A1, W2) + b2            # [m, 10]
    A2 = softmax(Z2)                    # [m, 10]

    # Categorical Cross-Entropy Loss: L = - (1/m) * sum(Y * log(A2))
    loss = -np.mean(np.sum(Y_train * np.log(A2 + 1e-12), axis=1))

    # ==========================
    # BACKWARD PASS (CHAIN RULE):
    # ==========================
    # Derivative of Cross-Entropy w.r.t Softmax pre-activation:
    # dL/dZ2 = (A2 - Y) / m
    dZ2 = (A2 - Y_train) / m_train                     # [m, 10]

    # Gradients for Layer 2:
    # dL/dW2 = A1^T . dZ2
    # dL/db2 = sum(dZ2)
    dW2 = np.dot(A1.T, dZ2)                            # [32, 10]
    db2 = np.sum(dZ2, axis=0, keepdims=True)           # [1, 10]

    # Error propagated back to Hidden Layer:
    # dL/dA1 = dZ2 . W2^T
    # dL/dZ1 = dL/dA1 * ReLU'(Z1)
    dA1 = np.dot(dZ2, W2.T)                            # [m, 32]
    dZ1 = dA1 * relu_gradient(Z1)                      # [m, 32]

    # Gradients for Layer 1:
    # dL/dW1 = X^T . dZ1
    # dL/db1 = sum(dZ1)
    dW1 = np.dot(X_train.T, dZ1)                       # [64, 32]
    db1 = np.sum(dZ1, axis=0, keepdims=True)           # [1, 32]

    # ==========================
    # WEIGHT & BIAS UPDATES:
    # ==========================
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 20 == 0 or epoch == 1:
        train_preds = np.argmax(A2, axis=1)
        train_acc = np.mean(train_preds == y_train) * 100
        grad_norm_W1 = np.linalg.norm(dW1)
        grad_norm_W2 = np.linalg.norm(dW2)
        print(f"Epoch {epoch:3d}/{num_epochs} | Loss: {loss:.4f} | Train Acc: {train_acc:6.2f}% | ||dW1||: {grad_norm_W1:.6f} | ||dW2||: {grad_norm_W2:.6f}")

# ----------------------------------------------------------------------
# 4. Evaluation on Unseen Test Digits
# ----------------------------------------------------------------------
Z1_test = np.dot(X_test, W1) + b1
A1_test = relu(Z1_test)
Z2_test = np.dot(A1_test, W2) + b2
A2_test = softmax(Z2_test)

test_predictions = np.argmax(A2_test, axis=1)
test_accuracy = np.mean(test_predictions == y_test) * 100

print(f"\n--- Digit Recognition Test Set Evaluation ---")
print(f"Test Accuracy: {test_accuracy:.2f}% ({np.sum(test_predictions == y_test)}/{len(y_test)} correct)")
print("\nSample Test Digit Predictions:")
for i in range(10):
    print(f"Sample #{i+1:2d} | True Digit: {y_test[i]} | Predicted: {test_predictions[i]} | Confidence: {A2_test[i, test_predictions[i]]:.4f}")

# ----------------------------------------------------------------------
# Output comments:
# Digit Recognition: 600 samples | 64 Input Features | 10 Classes
# Network Architecture: [64 Inputs] -> [32 Hidden (ReLU)] -> [10 Output (Softmax)]
# 
# --- Training with Backpropagation Algorithm ---
# Epoch   1/100 | Loss: 2.7932 | Train Acc:  16.25% | ||dW1||: 1.989719 | ||dW2||: 1.612622
# Epoch  20/100 | Loss: 0.1839 | Train Acc: 100.00% | ||dW1||: 0.320480 | ||dW2||: 0.273759
# Epoch  40/100 | Loss: 0.0643 | Train Acc: 100.00% | ||dW1||: 0.125725 | ||dW2||: 0.107743
# Epoch  60/100 | Loss: 0.0369 | Train Acc: 100.00% | ||dW1||: 0.075521 | ||dW2||: 0.064969
# Epoch  80/100 | Loss: 0.0254 | Train Acc: 100.00% | ||dW1||: 0.053202 | ||dW2||: 0.046007
# Epoch 100/100 | Loss: 0.0191 | Train Acc: 100.00% | ||dW1||: 0.040744 | ||dW2||: 0.035444
# 
# --- Digit Recognition Test Set Evaluation ---
# Test Accuracy: 100.00% (120/120 correct)
# 
# Sample Test Digit Predictions:
# Sample # 1 | True Digit: 5 | Predicted: 5 | Confidence: 0.9910
# Sample # 2 | True Digit: 1 | Predicted: 1 | Confidence: 0.9939
# Sample # 3 | True Digit: 0 | Predicted: 0 | Confidence: 0.9902
# Sample # 4 | True Digit: 9 | Predicted: 9 | Confidence: 0.9819
# Sample # 5 | True Digit: 3 | Predicted: 3 | Confidence: 0.9841
# Sample # 6 | True Digit: 8 | Predicted: 8 | Confidence: 0.9578
# Sample # 7 | True Digit: 9 | Predicted: 9 | Confidence: 0.9209
# Sample # 8 | True Digit: 2 | Predicted: 2 | Confidence: 0.9933
# Sample # 9 | True Digit: 2 | Predicted: 2 | Confidence: 0.9819
# Sample #10 | True Digit: 5 | Predicted: 5 | Confidence: 0.9936
# ----------------------------------------------------------------------

# ======================================================================
