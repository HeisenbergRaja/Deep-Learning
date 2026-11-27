# Program 63: RNN Text Classification for Customer Complaint Categorization
# Question: A customer service platform develops an RNN to classify incoming customer messages into different complaint categories. Evaluate the classification model using precision, recall, and F1-score. (BL5 – Evaluate)
# ======================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(42)
torch.manual_seed(42)

# Synthetic customer complaint corpus (4 classes, 10 samples each)
complaint_corpus = {
    0: [  # Billing & Payments
        "billed twice for subscription payment",
        "unauthorized charge on credit card statement",
        "incorrect billing amount on invoice",
        "requesting refund for double billing",
        "charged extra fee at checkout",
        "payment deducted but subscription inactive",
        "duplicate invoice transaction on statement",
        "overcharged on renewal membership",
        "billing error please refund card",
        "refund not credited to bank account"
    ],
    1: [  # Delivery & Shipping
        "package not arrived courier delivery late",
        "shipment tracking number not updating",
        "courier delivered package wrong address",
        "item arrived damaged box crushed shipping",
        "delivery delayed two weeks no update",
        "parcel marked delivered never received",
        "shipping courier lost package transit",
        "courier delivery stuck distribution facility",
        "damaged goods received shipping carrier",
        "delivery date passed parcel missing"
    ],
    2: [  # Product Defects
        "device stopped working after two days",
        "hardware defective screen flickering display",
        "battery drains within thirty minutes defect",
        "product power button malfunctioning broken",
        "low build quality defective plastic cracked",
        "hardware overheating shutting down abruptly",
        "internal speaker loud static crackling defect",
        "device cannot charge defective power connector",
        "faulty components inside unit defective",
        "defective unit broke during operation"
    ],
    3: [  # Account & Login Access
        "cannot login customer account invalid credentials",
        "password reset link not arriving email",
        "account locked multiple failed login attempts",
        "two factor authentication sms code missing",
        "unable access dashboard portal access denied",
        "forgot password account recovery not working",
        "user session expired cannot login profile",
        "security verification failed access account",
        "locked out of user dashboard restore access",
        "account profile locked password reset failed"
    ]
}

category_names = ["Billing", "Shipping", "Product Defect", "Account Access"]

# Build vocabulary
all_samples = []
for label, texts in complaint_corpus.items():
    for text in texts:
        all_samples.append((text, label))

vocab = {"<PAD>": 0, "<UNK>": 1}
for text, _ in all_samples:
    for word in text.split():
        if word not in vocab:
            vocab[word] = len(vocab)

def encode_sentence(text, max_len=10):
    tokens = [vocab.get(w, 1) for w in text.split()][:max_len]
    return tokens + [0] * max(0, max_len - len(tokens))

# Partition: 8 train samples, 2 test samples per class
train_corpus, test_corpus = [], []
for c in range(4):
    c_items = [item for item in all_samples if item[1] == c]
    train_corpus.extend(c_items[:8])
    test_corpus.extend(c_items[8:])

X_train = torch.tensor([encode_sentence(t) for t, _ in train_corpus], dtype=torch.long)
y_train = torch.tensor([lbl for _, lbl in train_corpus], dtype=torch.long)
X_test = torch.tensor([encode_sentence(t) for t, _ in test_corpus], dtype=torch.long)
y_test = torch.tensor([lbl for _, lbl in test_corpus], dtype=torch.long)

# Recurrent Text Classification Architecture with Masked Mean Pooling
class ComplaintClassifierRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=48, num_classes=4):
        super(ComplaintClassifierRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.rnn(embedded)
        # Mask out padding tokens to pool genuine sequence representations
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)

torch.manual_seed(42)
classifier = ComplaintClassifierRNN(vocab_size=len(vocab), embed_dim=32, hidden_dim=48, num_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.01)

print("Training RNN Text Classifier on Customer Complaint Corpus...")
for epoch in range(35):
    classifier.train()
    optimizer.zero_grad()
    logits = classifier(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()

# Evaluation on Test Set
classifier.eval()
with torch.no_grad():
    test_logits = classifier(X_test)
    y_pred = torch.argmax(test_logits, dim=1).numpy()
    y_true = y_test.numpy()

# Calculate Evaluation Metrics: Precision, Recall, F1-Score
def evaluate_classification(y_t, y_p, num_classes=4):
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    support_per_class = []

    for c in range(num_classes):
        tp = np.sum((y_t == c) & (y_p == c))
        fp = np.sum((y_t != c) & (y_p == c))
        fn = np.sum((y_t == c) & (y_p != c))
        sup = np.sum(y_t == c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        precision_per_class.append(prec)
        recall_per_class.append(rec)
        f1_per_class.append(f1)
        support_per_class.append(sup)

    macro_p = float(np.mean(precision_per_class))
    macro_r = float(np.mean(recall_per_class))
    macro_f1 = float(np.mean(f1_per_class))
    accuracy = float(np.mean(y_t == y_p))
    return precision_per_class, recall_per_class, f1_per_class, support_per_class, macro_p, macro_r, macro_f1, accuracy

precs, recs, f1s, sups, mac_p, mac_r, mac_f1, acc = evaluate_classification(y_true, y_pred, num_classes=4)

print("\n" + "=" * 65)
print("              CLASSIFICATION REPORT")
print("=" * 65)
print(f"{'Category':<18} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
print("-" * 65)
for i in range(4):
    print(f"{category_names[i]:<18} | {precs[i]:<10.4f} | {recs[i]:<10.4f} | {f1s[i]:<10.4f} | {sups[i]:<8}")
print("-" * 65)
print(f"{'Macro Average':<18} | {mac_p:<10.4f} | {mac_r:<10.4f} | {mac_f1:<10.4f} | {len(y_true):<8}")
print(f"{'Overall Accuracy':<18} | {acc * 100:.2f}%")
print("=" * 65)

# Confusion Matrix
cm = np.zeros((4, 4), dtype=int)
for t_lbl, p_lbl in zip(y_true, y_pred):
    cm[t_lbl, p_lbl] += 1
print("\nConfusion Matrix (Rows: Actual, Columns: Predicted):")
print(cm)

print("\nSample Test Inquiries and Automated Routing Predictions:")
for i in range(len(test_corpus)):
    msg, true_lbl = test_corpus[i]
    pred_lbl = y_pred[i]
    status = "CORRECT" if pred_lbl == true_lbl else "MISCLASSIFIED"
    print(f"Message: '{msg}'")
    print(f" -> Predicted: {category_names[pred_lbl]} | True: {category_names[true_lbl]} [{status}]")

# Output comments:
# Training RNN Text Classifier on Customer Complaint Corpus...
# 
# =================================================================
#               CLASSIFICATION REPORT
# =================================================================
# Category           | Precision  | Recall     | F1-Score   | Support 
# -----------------------------------------------------------------
# Billing            | 1.0000     | 1.0000     | 1.0000     | 2       
# Shipping           | 0.6667     | 1.0000     | 0.8000     | 2       
# Product Defect     | 1.0000     | 0.5000     | 0.6667     | 2       
# Account Access     | 1.0000     | 1.0000     | 1.0000     | 2       
# -----------------------------------------------------------------
# Macro Average      | 0.9167     | 0.8750     | 0.8667     | 8       
# Overall Accuracy   | 87.50%
# =================================================================
# 
# Confusion Matrix (Rows: Actual, Columns: Predicted):
# [[2 0 0 0]
#  [0 2 0 0]
#  [0 1 1 0]
#  [0 0 0 2]]
# 
# Sample Test Inquiries and Automated Routing Predictions:
# Message: 'billing error please refund card'
#  -> Predicted: Billing | True: Billing [CORRECT]
# Message: 'refund not credited to bank account'
#  -> Predicted: Billing | True: Billing [CORRECT]
# Message: 'damaged goods received shipping carrier'
#  -> Predicted: Shipping | True: Shipping [CORRECT]
# Message: 'delivery date passed parcel missing'
#  -> Predicted: Shipping | True: Shipping [CORRECT]
# Message: 'faulty components inside unit defective'
#  -> Predicted: Shipping | True: Product Defect [MISCLASSIFIED]
# Message: 'defective unit broke during operation'
#  -> Predicted: Product Defect | True: Product Defect [CORRECT]
# Message: 'locked out of user dashboard restore access'
#  -> Predicted: Account Access | True: Account Access [CORRECT]
# Message: 'account profile locked password reset failed'
#  -> Predicted: Account Access | True: Account Access [CORRECT]

# ======================================================================
