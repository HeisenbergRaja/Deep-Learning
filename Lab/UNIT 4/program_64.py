# Program 64: Tree-Structured Recursive Neural Network for Sentence Sentiment Classification
# Question: A natural language processing research team wants to classify sentences by considering their syntactic tree structures. Implement a recursive neural network for sentiment classification and analyse its predictions. (BL6 – Create)
# ======================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# Binary Parse Tree Data Structure
class BinaryTreeNode:
    def __init__(self, word=None, left=None, right=None, label=None):
        self.word = word        # Word string if leaf node, else None
        self.left = left        # Left child BinaryTreeNode
        self.right = right      # Right child BinaryTreeNode
        self.label = label      # Sentiment label (0: Negative, 1: Positive)
        self.word_id = None     # Vocabulary integer id if leaf
        self.hidden = None      # Computed hidden vector representation
        self.logits = None      # Sentiment classification logits

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

# Vocabulary Definition
vocab = {
    "<PAD>": 0, "not": 1, "good": 2, "terrible": 3,
    "great": 4, "acting": 5, "boring": 6, "plot": 7,
    "movie": 8, "brilliant": 9, "poor": 10, "script": 11
}

# Tree Builder Helpers
def leaf(word, label):
    node = BinaryTreeNode(word=word, label=label)
    node.word_id = vocab[word]
    return node

def parent(left_child, right_child, label):
    return BinaryTreeNode(left=left_child, right=right_child, label=label)

# Synthetic Syntactic Tree Dataset (Demonstrating Semantic Compositionality)
# 1. ("not" [-] + "good" [+]) -> Negative sentiment composition [0]
tree_1 = parent(leaf("not", 0), leaf("good", 1), label=0)

# 2. ("great" [+] + "acting" [+]) -> Positive sentiment composition [1]
tree_2 = parent(leaf("great", 1), leaf("acting", 1), label=1)

# 3. ("boring" [-] + "plot" [0]) -> Negative sentiment composition [0]
tree_3 = parent(leaf("boring", 0), leaf("plot", 0), label=0)

# 4. ("not" [-] + "terrible" [-]) -> Negation of negative yields Positive [1]
tree_4 = parent(leaf("not", 0), leaf("terrible", 0), label=1)

# 5. Complex Tree: (("boring" [-] + "script" [-]) [-] + ("brilliant" [+] + "acting" [+]) [+]) -> Positive [1]
tree_5 = parent(
    parent(leaf("boring", 0), leaf("script", 0), label=0),
    parent(leaf("brilliant", 1), leaf("acting", 1), label=1),
    label=1
)

# 6. Complex Tree: (("not" [-] + "good" [+]) [-] + ("movie" [0] + "plot" [0]) [0]) -> Negative [0]
tree_6 = parent(
    parent(leaf("not", 0), leaf("good", 1), label=0),
    parent(leaf("movie", 0), leaf("plot", 0), label=0),
    label=0
)

trees = [tree_1, tree_2, tree_3, tree_4, tree_5, tree_6]

# Recursive Neural Network (Tree-RNN Architecture)
class TreeRecursiveNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, num_classes=2):
        super(TreeRecursiveNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Composition matrix combines left and right child representations: p = tanh(W [h_L; h_R] + b)
        self.composition = nn.Linear(2 * embed_dim, embed_dim)
        # Node-level sentiment classifier: y = softmax(W_s * h + b_s)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, node):
        if node.is_leaf:
            # Leaf node: lookup word embedding and apply non-linearity
            word_idx = torch.tensor([node.word_id], dtype=torch.long)
            node.hidden = torch.tanh(self.embedding(word_idx))
        else:
            # Recursive bottom-up tree traversal
            h_left = self.forward(node.left)
            h_right = self.forward(node.right)
            # Concatenate children representations [h_L; h_R]
            combined = torch.cat([h_left, h_right], dim=-1)
            node.hidden = torch.tanh(self.composition(combined))

        node.logits = self.classifier(node.hidden)
        return node.hidden

    def compute_tree_loss(self, node, criterion):
        """Recursively compute cross-entropy loss across all labeled sub-nodes."""
        target = torch.tensor([node.label], dtype=torch.long)
        current_loss = criterion(node.logits, target)
        if not node.is_leaf:
            current_loss = current_loss + self.compute_tree_loss(node.left, criterion)
            current_loss = current_loss + self.compute_tree_loss(node.right, criterion)
        return current_loss

# Initialize Model and Training
model = TreeRecursiveNN(vocab_size=len(vocab), embed_dim=16, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)

print("Training Tree-Structured Recursive Neural Network (Tree-RNN)...")
for epoch in range(40):
    total_epoch_loss = 0.0
    for tree in trees:
        optimizer.zero_grad()
        model(tree) # Recursive forward pass
        loss = model.compute_tree_loss(tree, criterion) # Backpropagation through tree structure
        loss.backward()
        optimizer.step()
        total_epoch_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch [{epoch+1:2d}/40] - Tree Structure Loss: {total_epoch_loss:.5f}")

# Syntactic Composition Sentiment Analysis
sentiment_labels = {0: "Negative", 1: "Positive"}
print("\n" + "=" * 70)
print("        TREE-RNN SYNTACTIC SENTIMENT COMPOSITION ANALYSIS")
print("=" * 70)

def display_composition_analysis(node, depth=0):
    indent = "  " * depth
    probs = torch.softmax(node.logits, dim=1).detach().squeeze().numpy()
    pred_class = int(np.argmax(probs))
    pred_str = sentiment_labels[pred_class]
    conf = probs[pred_class] * 100.0

    if node.is_leaf:
        print(f"{indent}[Word: '{node.word}'] -> Sentiment: {pred_str} ({conf:.1f}%)")
    else:
        display_composition_analysis(node.left, depth + 1)
        display_composition_analysis(node.right, depth + 1)
        print(f"{indent}[Phrase Composition] -> Root Sentiment: {pred_str} ({conf:.1f}%) | Target: {sentiment_labels[node.label]}")

print("\n--- Test Case 1: Negation Composition ('not' + 'good') ---")
model(tree_1)
display_composition_analysis(tree_1)

print("\n--- Test Case 2: Double Negative Negation ('not' + 'terrible') ---")
model(tree_4)
display_composition_analysis(tree_4)

print("\n--- Test Case 3: Contrastive Conjunction ('boring script' vs 'brilliant acting') ---")
model(tree_5)
display_composition_analysis(tree_5)

print("\nArchitectural Synthesis (BL6 - Create):")
print("Unlike flat sequential RNNs, a Recursive Neural Network computes representations hierarchically")
print("over syntactic constituency parse trees. This enables precise compositionality:")
print("1. 'not' + 'good' successfully flips positive to negative.")
print("2. 'not' + 'terrible' correctly flips negative to positive.")
print("3. Complex phrases resolve sub-clause conflicts according to syntactic tree hierarchy.")

# Output comments:
# Training Tree-Structured Recursive Neural Network (Tree-RNN)...
#   Epoch [10/40] - Tree Structure Loss: 0.22036
#   Epoch [20/40] - Tree Structure Loss: 0.06230
#   Epoch [30/40] - Tree Structure Loss: 0.03186
#   Epoch [40/40] - Tree Structure Loss: 0.01922
# 
# ======================================================================
#         TREE-RNN SYNTACTIC SENTIMENT COMPOSITION ANALYSIS
# ======================================================================
# 
# --- Test Case 1: Negation Composition ('not' + 'good') ---
#   [Word: 'not'] -> Sentiment: Negative (99.9%)
#   [Word: 'good'] -> Sentiment: Positive (99.9%)
# [Phrase Composition] -> Root Sentiment: Negative (100.0%) | Target: Negative
# 
# --- Test Case 2: Double Negative Negation ('not' + 'terrible') ---
#   [Word: 'not'] -> Sentiment: Negative (99.9%)
#   [Word: 'terrible'] -> Sentiment: Negative (99.8%)
# [Phrase Composition] -> Root Sentiment: Positive (100.0%) | Target: Positive
# 
# --- Test Case 3: Contrastive Conjunction ('boring script' vs 'brilliant acting') ---
#     [Word: 'boring'] -> Sentiment: Negative (99.9%)
#     [Word: 'script'] -> Sentiment: Negative (99.6%)
#   [Phrase Composition] -> Root Sentiment: Negative (100.0%) | Target: Negative
#     [Word: 'brilliant'] -> Sentiment: Positive (99.9%)
#     [Word: 'acting'] -> Sentiment: Positive (99.9%)
#   [Phrase Composition] -> Root Sentiment: Positive (100.0%) | Target: Positive
# [Phrase Composition] -> Root Sentiment: Positive (100.0%) | Target: Positive
# 
# Architectural Synthesis (BL6 - Create):
# Unlike flat sequential RNNs, a Recursive Neural Network computes representations hierarchically
# over syntactic constituency parse trees. This enables precise compositionality:
# 1. 'not' + 'good' successfully flips positive to negative.
# 2. 'not' + 'terrible' correctly flips negative to positive.
# 3. Complex phrases resolve sub-clause conflicts according to syntactic tree hierarchy.




# ======================================================================
# CO5 PROGRAMS 65 TO 72
# ======================================================================
