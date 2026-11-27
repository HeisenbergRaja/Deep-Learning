# Program 29: Forward propagation
# Question: Multilayer neural network - implement forward propagation, calculate hidden and output layer outputs.
# ======================================================================

import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# 1. Forward Propagation Theory & Formulation
# ----------------------------------------------------------------------
# For each layer l = 1, 2, ..., L:
#   Pre-activation:   Z^[l] = A^[l-1] . W^[l] + b^[l]
#   Post-activation:  A^[l] = g^[l]( Z^[l] )
# Where:
#   A^[0] = X (Input feature matrix of shape [batch_size, n_features])
#   W^[l] is the weight matrix of shape [n_{l-1}, n_l]
#   b^[l] is the bias vector of shape [1, n_l]
#   g^[l] is the layer activation function

np.random.seed(42)

# Input data: 3 instances, 4 features each
X_input = np.array([
    [1.5, -0.8,  2.2, -1.0],
    [0.4,  1.6, -0.5,  0.9],
    [-1.2, 0.3,  0.7, -1.5]
], dtype=np.float32)

batch_size, input_dim = X_input.shape
h1_dim = 4  # Hidden Layer 1 neurons
h2_dim = 3  # Hidden Layer 2 neurons
out_dim = 2 # Output neurons (2 classes)

# ----------------------------------------------------------------------
# 2. Layer Parameters Initialization
# ----------------------------------------------------------------------
# Layer 1: (4 -> 4)
W1 = np.random.randn(input_dim, h1_dim).astype(np.float32) * 0.5
b1 = np.array([0.1, -0.2, 0.15, -0.05], dtype=np.float32)

# Layer 2: (4 -> 3)
W2 = np.random.randn(h1_dim, h2_dim).astype(np.float32) * 0.5
b2 = np.array([-0.1, 0.25, 0.0], dtype=np.float32)

# Output Layer: (3 -> 2)
W3 = np.random.randn(h2_dim, out_dim).astype(np.float32) * 0.5
b3 = np.array([0.05, -0.05], dtype=np.float32)

# Activation functions
def relu_act(z):
    return np.maximum(0.0, z)

def tanh_act(z):
    return np.tanh(z)

def softmax_act(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# ----------------------------------------------------------------------
# 3. Step-by-Step Forward Propagation Execution
# ----------------------------------------------------------------------
print("======================================================================")
print("              STEP-BY-STEP FORWARD PROPAGATION CALCULATION            ")
print("======================================================================")
print(f"Input Matrix X (Shape: {X_input.shape}):")
print(X_input)

# Step 1: Input to Hidden Layer 1 (ReLU Activation)
Z1 = np.dot(X_input, W1) + b1
A1 = relu_act(Z1)

print("\n--- [Hidden Layer 1] ---")
print(f"Weight W1 Shape: {W1.shape} | Bias b1 Shape: {b1.shape}")
print(f"Linear Pre-Activation Z1 = X . W1 + b1 (Shape: {Z1.shape}):")
print(np.round(Z1, 4))
print(f"Activated Output A1 = ReLU(Z1) (Shape: {A1.shape}):")
print(np.round(A1, 4))

# Step 2: Hidden Layer 1 to Hidden Layer 2 (Tanh Activation)
Z2 = np.dot(A1, W2) + b2
A2 = tanh_act(Z2)

print("\n--- [Hidden Layer 2] ---")
print(f"Weight W2 Shape: {W2.shape} | Bias b2 Shape: {b2.shape}")
print(f"Linear Pre-Activation Z2 = A1 . W2 + b2 (Shape: {Z2.shape}):")
print(np.round(Z2, 4))
print(f"Activated Output A2 = Tanh(Z2) (Shape: {A2.shape}):")
print(np.round(A2, 4))

# Step 3: Hidden Layer 2 to Output Layer (Softmax Activation)
Z3 = np.dot(A2, W3) + b3
A3 = softmax_act(Z3)

print("\n--- [Output Layer] ---")
print(f"Weight W3 Shape: {W3.shape} | Bias b3 Shape: {b3.shape}")
print(f"Linear Pre-Activation Z3 = A2 . W3 + b3 (Shape: {Z3.shape}):")
print(np.round(Z3, 4))
print(f"Final Output Probabilities A3 = Softmax(Z3) (Shape: {A3.shape}):")
print(np.round(A3, 4))
print(f"Row-wise Probability Sum Check: {np.sum(A3, axis=1)}")

# ----------------------------------------------------------------------
# 4. Verification with PyTorch Module
# ----------------------------------------------------------------------
class PyTorchForwardNet(nn.Module):
    def __init__(self):
        super(PyTorchForwardNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, out_dim)

        # Assign exact same weights to verify equivalence
        self.fc1.weight.data = torch.tensor(W1.T)
        self.fc1.bias.data = torch.tensor(b1)
        self.fc2.weight.data = torch.tensor(W2.T)
        self.fc2.bias.data = torch.tensor(b2)
        self.fc3.weight.data = torch.tensor(W3.T)
        self.fc3.bias.data = torch.tensor(b3)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        out = torch.softmax(self.fc3(h2), dim=1)
        return out

torch_net = PyTorchForwardNet()
torch_out = torch_net(torch.tensor(X_input)).detach().numpy()
max_diff = np.max(np.abs(A3 - torch_out))
print(f"\nPyTorch Forward Equivalence Check (Max Absolute Diff): {max_diff:.8e}")

# ----------------------------------------------------------------------
# Output comments:
# ======================================================================
#               STEP-BY-STEP FORWARD PROPAGATION CALCULATION            
# ======================================================================
# Input Matrix X (Shape: (3, 4)):
# [[ 1.5 -0.8  2.2 -1. ]
#  [ 0.4  1.6 -0.5  0.9]
#  [-1.2  0.3  0.7 -1.5]]
# 
# --- [Hidden Layer 1] ---
# Weight W1 Shape: (4, 4) | Bias b1 Shape: (4,)
# Linear Pre-Activation Z1 = X . W1 + b1 (Shape: (3, 4)):
# [[-0.0712  1.3434  0.3568  0.5541]
#  [ 0.2383 -1.4116  0.8825  0.732 ]
#  [-0.5789  1.4727  1.1298 -0.59  ]]
# Activated Output A1 = ReLU(Z1) (Shape: (3, 4)):
# [[0.     1.3434 0.3568 0.5541]
#  [0.2383 0.     0.8825 0.732 ]
#  [0.     1.4727 1.1298 0.    ]]
# 
# --- [Hidden Layer 2] ---
# Weight W2 Shape: (4, 3) | Bias b2 Shape: (3,)
# Linear Pre-Activation Z2 = A1 . W2 + b2 (Shape: (3, 3)):
# [[-1.0059  0.6614 -0.1447]
#  [-0.1503 -0.7625 -0.2109]
#  [-1.1018  0.5244 -0.4738]]
# Activated Output A2 = Tanh(Z2) (Shape: (3, 3)):
# [[-0.764   0.5793 -0.1437]
#  [-0.1491 -0.6426 -0.2078]
#  [-0.8011  0.4811 -0.4412]]
# 
# --- [Output Layer] ---
# Weight W3 Shape: (3, 2) | Bias b3 Shape: (2,)
# Linear Pre-Activation Z3 = A2 . W3 + b3 (Shape: (3, 2)):
# [[ 0.1061  0.6739]
#  [ 0.2895 -0.5134]
#  [ 0.1488  0.7458]]
# Final Output Probabilities A3 = Softmax(Z3) (Shape: (3, 2)):
# [[0.3617 0.6383]
#  [0.6906 0.3094]
#  [0.355  0.645 ]]
# Row-wise Probability Sum Check: [1.         1.         0.99999994]
# 
# PyTorch Forward Equivalence Check (Max Absolute Diff): 2.98023224e-08
# ----------------------------------------------------------------------

# ======================================================================
