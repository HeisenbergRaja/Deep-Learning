# Program 53: Next word prediction
# Question: RNN-based language model for next-word prediction.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 2. Synthetic Domain Text Corpus
text_corpus = [
    "deep learning is a subset of machine learning",
    "recurrent neural networks process sequential time series data",
    "neural networks learn rich representations from complex data",
    "long short term memory networks handle long term dependencies",
    "attention mechanisms focus dynamically on informative input tokens",
    "supervised learning trains predictive models on labeled training datasets",
    "transformers utilize self attention mechanisms for sequence modeling tasks",
    "deep learning models require extensive computational resources and data",
    "natural language processing models learn linguistic structure from text",
    "gradient descent optimizes neural network parameters during training"
]

# 3. Vocabulary Construction
words = []
for sentence in text_corpus:
    words.extend(sentence.lower().split())

vocabulary = sorted(list(set(words)))
word2idx = {w: i for i, w in enumerate(vocabulary)}
idx2word = {i: w for i, w in enumerate(vocabulary)}
vocab_size = len(vocabulary)

# 4. Create N-gram Training Pairs (Context length = 3 words -> target next word)
context_len = 3
input_sequences, target_words = [], []

for sentence in text_corpus:
    tokens = sentence.lower().split()
    for i in range(len(tokens) - context_len):
        input_sequences.append([word2idx[w] for w in tokens[i:i + context_len]])
        target_words.append(word2idx[tokens[i + context_len]])

X_text = torch.tensor(input_sequences, dtype=torch.long)
y_text = torch.tensor(target_words, dtype=torch.long)
text_loader = DataLoader(TensorDataset(X_text, y_text), batch_size=8, shuffle=True)

# 5. RNN Language Model Architecture
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_sz, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_sz)

    def forward(self, x):
        embedded = self.embedding(x)          # (batch, seq_len, embed_dim)
        rnn_out, _ = self.rnn(embedded)       # (batch, seq_len, hidden_dim)
        logits = self.fc(rnn_out[:, -1, :])   # (batch, vocab_sz)
        return logits

model_53 = RNNLanguageModel(vocab_sz=vocab_size, embed_dim=32, hidden_dim=64)
criterion_53 = nn.CrossEntropyLoss()
optimizer_53 = optim.Adam(model_53.parameters(), lr=0.01)

# 6. Training Loop
print("\n--- Training RNN Language Model ---")
epochs_53 = 60
for epoch in range(1, epochs_53 + 1):
    model_53.train()
    total_loss = 0.0
    for bx, by in text_loader:
        optimizer_53.zero_grad()
        logits = model_53(bx)
        loss = criterion_53(logits, by)
        loss.backward()
        optimizer_53.step()
        total_loss += loss.item()
    if epoch % 20 == 0 or epoch == epochs_53:
        print(f"Epoch [{epoch:2d}/{epochs_53}], Cross-Entropy Loss: {total_loss / len(text_loader):.4f}")

# 7. Next Word Prediction and Sentence Generation Functions
def predict_next_words(prompt, top_k=3):
    model_53.eval()
    tokens = prompt.lower().split()[-context_len:]
    indices = [word2idx[w] for w in tokens if w in word2idx]
    if len(indices) < context_len:
        return []
    inp = torch.tensor([indices], dtype=torch.long)
    with torch.no_grad():
        logits = model_53(inp)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top_probs, top_indices = torch.topk(probs, top_k)
    return [(idx2word[idx.item()], top_probs[i].item() * 100.0) for i, idx in enumerate(top_indices)]

def generate_continuation(seed_prompt, max_words=5):
    current = seed_prompt.lower().split()
    for _ in range(max_words):
        context = current[-context_len:]
        indices = [word2idx[w] for w in context if w in word2idx]
        if len(indices) < context_len:
            break
        inp = torch.tensor([indices], dtype=torch.long)
        with torch.no_grad():
            next_idx = torch.argmax(model_53(inp), dim=1).item()
        current.append(idx2word[next_idx])
    return " ".join(current)

# 8. Demonstration
print("\n" + "=" * 48)
print("           NEXT-WORD PREDICTION SAMPLES         ")
print("=" * 48)
prompts_to_test = [
    "deep learning is",
    "recurrent neural networks",
    "neural networks learn",
    "long short term"
]

for p in prompts_to_test:
    candidates = predict_next_words(p, top_k=3)
    cand_str = ", ".join([f"'{w}' ({prob:.1f}%)" for w, prob in candidates])
    print(f"Prompt : \"{p}\"")
    print(f"Top-3  : {cand_str}\n")

print("--- Text Generation Continuation ---")
for seed in ["deep learning is", "neural networks learn"]:
    generated = generate_continuation(seed, max_words=4)
    print(f"Seed: \"{seed}\" -> Generated: \"{generated}\"")
print("=" * 48)

# Output comments:
# --- Training RNN Language Model ---
# Epoch [20/60], Cross-Entropy Loss: 0.0088
# Epoch [40/60], Cross-Entropy Loss: 0.0034
# Epoch [60/60], Cross-Entropy Loss: 0.0018
# 
# ================================================
#            NEXT-WORD PREDICTION SAMPLES         
# ================================================
# Prompt : "deep learning is"
# Top-3  : 'a' (99.8%), 'require' (0.0%), 'series' (0.0%)
# 
# Prompt : "recurrent neural networks"
# Top-3  : 'process' (99.8%), 'parameters' (0.0%), 'computational' (0.0%)
# 
# Prompt : "neural networks learn"
# Top-3  : 'rich' (99.8%), 'sequential' (0.0%), 'linguistic' (0.0%)
# 
# Prompt : "long short term"
# Top-3  : 'memory' (99.8%), 'dependencies' (0.0%), 'and' (0.0%)
# 
# --- Text Generation Continuation ---
# Seed: "deep learning is" -> Generated: "deep learning is a subset of machine"
# Seed: "neural networks learn" -> Generated: "neural networks learn rich representations from complex"
# ================================================
# Conclusion: The RNN language model effectively learns the transitional
# grammar and sequential word associations, yielding confident and coherent
# next-token predictions.

# ======================================================================
