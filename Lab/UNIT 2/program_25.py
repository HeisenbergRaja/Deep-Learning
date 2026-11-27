# Program 25: XOR neural network
# Question: Design feedforward neural network with hidden layer to solve XOR classification.
# ======================================================================

import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------------------------------------------------
# 1. Dataset Preparation
# ----------------------------------------------------------------------
# XOR Truth Table:
# Input x1 | Input x2 | Target y (x1 XOR x2)
#    0     |    0     |       0
#    0     |    1     |       1
#    1     |    0     |       1
#    1     |    1     |       0
# Linearly non-separable problem requiring at least one hidden layer with
# non-linear activation to create non-linear decision boundaries.

torch.manual_seed(42)

X_xor = torch.tensor([[0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0]], dtype=torch.float32)

y_xor = torch.tensor([[0.0],
                      [1.0],
                      [1.0],
                      [0.0]], dtype=torch.float32)

# ----------------------------------------------------------------------
# 2. Feedforward Neural Network Architecture
# ----------------------------------------------------------------------
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        # Input Layer: 2 neurons -> Hidden Layer: 4 neurons
        self.fc1 = nn.Linear(in_features=2, out_features=4)
        # Non-linear activation function (Tanh maps to range [-1, 1])
        self.tanh = nn.Tanh()
        # Hidden Layer: 4 neurons -> Output Layer: 1 neuron
        self.fc2 = nn.Linear(in_features=4, out_features=1)
        # Sigmoid activation maps output to probability [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.tanh(self.fc1(x))
        out = self.sigmoid(self.fc2(h))
        return out

model_xor = XORNet()

# ----------------------------------------------------------------------
# 3. Model Training
# ----------------------------------------------------------------------
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model_xor.parameters(), lr=0.05)
epochs = 1000

print("--- Training XOR Feedforward Neural Network ---")
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    predictions = model_xor(X_xor)
    loss = criterion(predictions, y_xor)
    loss.backward()
    optimizer.step()

    if epoch % 250 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss.item():.6f}")

# ----------------------------------------------------------------------
# 4. Evaluation and Prediction Verification
# ----------------------------------------------------------------------
model_xor.eval()
print("\n--- XOR Truth Table Verification ---")
with torch.no_grad():
    raw_preds = model_xor(X_xor)
    binary_preds = (raw_preds >= 0.5).int()

    all_correct = True
    for i in range(len(X_xor)):
        inp = X_xor[i].numpy()
        actual = int(y_xor[i].item())
        prob = raw_preds[i].item()
        pred = int(binary_preds[i].item())
        is_match = (actual == pred)
        if not is_match:
            all_correct = False
        print(f"Input: {inp} | Actual: {actual} | Raw Prob: {prob:.4f} | Predicted Class: {pred} | Match: {is_match}")

print(f"\nFinal Result: {'XOR Solved Successfully (100% Accuracy)' if all_correct else 'Failed'}")

# ----------------------------------------------------------------------
# Output comments:
# --- Training XOR Feedforward Neural Network ---
# Epoch    1/1000 | Loss: 0.759778
# Epoch  250/1000 | Loss: 0.006477
# Epoch  500/1000 | Loss: 0.002285
# Epoch  750/1000 | Loss: 0.001213
# Epoch 1000/1000 | Loss: 0.000762
# 
# --- XOR Truth Table Verification ---
# Input: [0. 0.] | Actual: 0 | Raw Prob: 0.0000 | Predicted Class: 0 | Match: True
# Input: [0. 1.] | Actual: 1 | Raw Prob: 0.9992 | Predicted Class: 1 | Match: True
# Input: [1. 0.] | Actual: 1 | Raw Prob: 0.9992 | Predicted Class: 1 | Match: True
# Input: [1. 1.] | Actual: 0 | Raw Prob: 0.0014 | Predicted Class: 0 | Match: True
# 
# Final Result: XOR Solved Successfully (100% Accuracy)
# ----------------------------------------------------------------------

# ======================================================================
