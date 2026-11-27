# Program 49: Bidirectional RNN
# Question: Develop bidirectional RNN for sequence classification, compare with unidirectional.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 1. Set reproducibility seed
torch.manual_seed(42)
np.random.seed(42)

# 2. Generate Synthetic Time-Series Data for Sequence Classification
# 1000 sequences of length 20. Classification depends on signals at both ends.
# Class 1: sum of first 5 steps + sum of last 5 steps > 0; Class 0: <= 0
num_samples = 1000
seq_length = 20
X_raw = np.random.randn(num_samples, seq_length, 1).astype(np.float32)
y_raw = ((np.sum(X_raw[:, :5, 0], axis=1) + np.sum(X_raw[:, -5:, 0], axis=1)) > 0).astype(np.int64)

# 80/20 Train/Test split
X_train_49, X_test_49 = torch.tensor(X_raw[:800]), torch.tensor(X_raw[800:])
y_train_49, y_test_49 = torch.tensor(y_raw[:800]), torch.tensor(y_raw[800:])

train_loader_49 = DataLoader(TensorDataset(X_train_49, y_train_49), batch_size=32, shuffle=True)
test_loader_49 = DataLoader(TensorDataset(X_test_49, y_test_49), batch_size=32, shuffle=False)

# 3. Model Architecture: Unidirectional RNN
class UnidirectionalRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_classes=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)  # out shape: (batch, seq_len, hidden_size)
        return self.fc(out[:, -1, :])  # uses only final forward hidden state

# 4. Model Architecture: Bidirectional RNN
class BidirectionalRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_classes=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)  # out shape: (batch, seq_len, 2 * hidden_size)
        # Concatenate forward RNN state at last step and backward RNN state at first step
        forward_last = out[:, -1, :self.hidden_size]
        backward_first = out[:, 0, self.hidden_size:]
        feat = torch.cat([forward_last, backward_first], dim=1)
        return self.fc(feat)

# 5. Training and Evaluation Routine
def train_and_eval(model, name="Model", epochs=15, lr=0.005):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    print(f"\n--- Training {name} ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader_49:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader_49)
        losses.append(avg_loss)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:2d}/{epochs}], Loss: {avg_loss:.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in test_loader_49:
            preds = torch.argmax(model(bx), dim=1)
            correct += (preds == by).sum().item()
            total += by.size(0)
    acc = (correct / total) * 100.0
    return losses, acc

uni_model = UnidirectionalRNN()
bi_model = BidirectionalRNN()

uni_loss, uni_acc = train_and_eval(uni_model, "Unidirectional RNN", epochs=15)
bi_loss, bi_acc = train_and_eval(bi_model, "Bidirectional RNN", epochs=15)

print("\n" + "=" * 46)
print("             PERFORMANCE COMPARISON            ")
print("=" * 46)
print(f"Unidirectional RNN Test Accuracy : {uni_acc:.2f}%")
print(f"Bidirectional RNN  Test Accuracy : {bi_acc:.2f}%")
print(f"Absolute Gain with Bi-RNN        : +{bi_acc - uni_acc:.2f}%")
print("=" * 46)

# 6. Visualization
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(uni_loss, label='Unidirectional RNN', color='chocolate', marker='o')
plt.plot(bi_loss, label='Bidirectional RNN', color='teal', marker='s')
plt.title('Program 49: Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
bars = plt.bar(['Uni-RNN', 'Bi-RNN'], [uni_acc, bi_acc], color=['chocolate', 'teal'], width=0.5)
plt.title('Program 49: Test Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('program49_bidirectional_rnn.png')
plt.show()

# Output comments:
# --- Training Unidirectional RNN ---
# Epoch [ 5/15], Loss: 0.3474
# Epoch [10/15], Loss: 0.3120
# Epoch [15/15], Loss: 0.2651
# 
# --- Training Bidirectional RNN ---
# Epoch [ 5/15], Loss: 0.1665
# Epoch [10/15], Loss: 0.1217
# Epoch [15/15], Loss: 0.0892
# 
# ==============================================
#              PERFORMANCE COMPARISON            
# ==============================================
# Unidirectional RNN Test Accuracy : 87.00%
# Bidirectional RNN  Test Accuracy : 90.50%
# Absolute Gain with Bi-RNN        : +3.50%
# ==============================================
# Conclusion: The Bidirectional RNN outperforms the Unidirectional RNN
# because it processes the sequence in both forward and backward directions,
# allowing the network to capture contextual dependencies present at both
# ends of the time series.

# ======================================================================
