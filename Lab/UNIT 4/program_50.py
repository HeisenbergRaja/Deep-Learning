# Program 50: Attention mechanism
# Question: Implement attention mechanism for sequence classification, visualize attention weights.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 1. Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 2. Synthetic Data Generation: Sequence with key impulse at random time step
n_samples = 1000
seq_len = 25
X_data_50 = np.random.randn(n_samples, seq_len, 1).astype(np.float32) * 0.2
y_data_50 = np.random.randint(0, 2, size=(n_samples,)).astype(np.int64)
spike_positions = []

for i in range(n_samples):
    idx = np.random.randint(5, 20)  # Random position for informative signal
    spike_positions.append(idx)
    if y_data_50[i] == 1:
        X_data_50[i, idx, 0] += 2.5   # Positive pulse for class 1
    else:
        X_data_50[i, idx, 0] -= 2.5   # Negative pulse for class 0

# Split 80/20
X_tr_50, X_te_50 = torch.tensor(X_data_50[:800]), torch.tensor(X_data_50[800:])
y_tr_50, y_te_50 = torch.tensor(y_data_50[:800]), torch.tensor(y_data_50[800:])
spike_pos_test = spike_positions[800:]

train_loader_50 = DataLoader(TensorDataset(X_tr_50, y_tr_50), batch_size=32, shuffle=True)
test_loader_50 = DataLoader(TensorDataset(X_te_50, y_te_50), batch_size=32, shuffle=False)

# 3. Attention Module (Bahdanau / Additive Attention)
class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, rnn_out):
        # rnn_out: (batch_size, seq_len, hidden_dim)
        scores = self.score_net(rnn_out)             # (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1)        # (batch_size, seq_len, 1)
        context = torch.sum(weights * rnn_out, dim=1) # (batch_size, hidden_dim)
        return context, weights

# 4. Attention-based Classifier
class AttentionSequenceClassifier(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_classes=2):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attention = AttentionBlock(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        context, weights = self.attention(rnn_out)
        logits = self.fc(context)
        return logits, weights

# 5. Training
model_50 = AttentionSequenceClassifier(input_dim=1, hidden_dim=32, num_classes=2)
criterion_50 = nn.CrossEntropyLoss()
optimizer_50 = optim.Adam(model_50.parameters(), lr=0.005)

print("\n--- Training Attention Classifier ---")
for epoch in range(1, 16):
    model_50.train()
    total_loss = 0.0
    for bx, by in train_loader_50:
        optimizer_50.zero_grad()
        logits, _ = model_50(bx)
        loss = criterion_50(logits, by)
        loss.backward()
        optimizer_50.step()
        total_loss += loss.item()
    if epoch % 5 == 0 or epoch == 15:
        print(f"Epoch [{epoch:2d}/15], Loss: {total_loss / len(train_loader_50):.4f}")

# 6. Evaluation
model_50.eval()
correct, total = 0, 0
with torch.no_grad():
    for bx, by in test_loader_50:
        logits, _ = model_50(bx)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == by).sum().item()
        total += by.size(0)

test_acc_50 = (correct / total) * 100.0
print(f"\nAttention Model Test Accuracy: {test_acc_50:.2f}%")

# 7. Visualize Attention Weights for a Test Sample
sample_idx = 0
sample_x = X_te_50[sample_idx:sample_idx+1]
sample_y = y_te_50[sample_idx].item()
sample_spike = spike_pos_test[sample_idx]

with torch.no_grad():
    logits, attn_weights = model_50(sample_x)
    pred_label = torch.argmax(logits, dim=1).item()
    weights_np = attn_weights.squeeze().numpy()

print(f"Sample #{sample_idx} -> True Class: {sample_y}, Predicted Class: {pred_label}")
print(f"Informative Spike at Time Step : t = {sample_spike}")
print(f"Maximum Attention Weight Peak  : t = {np.argmax(weights_np)} (Weight = {np.max(weights_np):.4f})")

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(sample_x.squeeze().numpy(), marker='o', color='royalblue', label='Input Signal')
plt.axvline(x=sample_spike, color='red', linestyle='--', label=f'True Spike (t={sample_spike})')
plt.title(f'Program 50: Input Sequence (True Class={sample_y}, Pred={pred_label})')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.bar(range(seq_len), weights_np, color='darkorange', alpha=0.8, label='Attention Weight')
plt.axvline(x=sample_spike, color='red', linestyle='--', label=f'Target Time Step (t={sample_spike})')
plt.title('Learned Attention Weights Across Time Steps')
plt.xlabel('Time Step')
plt.ylabel('Weight (Softmax)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('program50_attention_mechanism.png')
plt.show()

# Output comments:
# --- Training Attention Classifier ---
# Epoch [ 5/15], Loss: 0.0010
# Epoch [10/15], Loss: 0.0002
# Epoch [15/15], Loss: 0.0001
# 
# Attention Model Test Accuracy: 100.00%
# Sample #0 -> True Class: 1, Predicted Class: 1
# Informative Spike at Time Step : t = 14
# Maximum Attention Weight Peak  : t = 14 (Weight = 0.9842)
# Conclusion: The attention mechanism successfully learns to concentrate
# its weight distribution on the specific time step where the informative
# pattern occurs, filtering out irrelevant noise.

# ======================================================================
