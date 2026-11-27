# Program 65: Seq2Seq translation
# Question: A language translation company wants to translate sentences from one language to another using a limited parallel dataset. Develop a sequence-to-sequence LSTM encoder–decoder model and evaluate its translation performance. (BL6 – Create)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Inline parallel corpus (English -> Spanish)
pairs = [
    ("how are you", "como estas"),
    ("i am fine", "estoy bien"),
    ("good morning", "buenos dias"),
    ("good night", "buenas noches"),
    ("thank you", "gracias"),
    ("see you later", "hasta luego"),
    ("welcome home", "bienvenido a casa"),
    ("hello friend", "hola amigo")
]

# 2. Vocabulary Construction
def build_vocab(sentences):
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for s in sentences:
        for word in s.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

src_sentences = [p[0] for p in pairs]
trg_sentences = [p[1] for p in pairs]
src_vocab = build_vocab(src_sentences)
trg_vocab = build_vocab(trg_sentences)
inv_trg_vocab = {idx: word for word, idx in trg_vocab.items()}

def encode(sentence, vocab, add_eos=True):
    tokens = [vocab.get(w, vocab["<UNK>"]) for w in sentence.split()]
    if add_eos:
        tokens.append(vocab["<EOS>"])
    return tokens

max_src_len = max(len(s.split()) for s in src_sentences) + 1
max_trg_len = max(len(s.split()) for s in trg_sentences) + 2

def pad(tokens, max_len, pad_val=0):
    return tokens + [pad_val] * (max_len - len(tokens))

X_data = torch.tensor([pad(encode(p[0], src_vocab), max_src_len) for p in pairs], dtype=torch.long)
Y_data = torch.tensor([pad([trg_vocab["<SOS>"]] + encode(p[1], trg_vocab), max_trg_len) for p in pairs], dtype=torch.long)

# 3. Model Architecture: LSTM Encoder-Decoder
class Encoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h, c) = self.lstm(embedded)
        return h, c

class Decoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_sz)

    def forward(self, x, h, c):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)
        out, (h, c) = self.lstm(embedded, (h, c))
        prediction = self.fc(out.squeeze(1))
        return prediction, h, c

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        vocab_sz = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, trg_len, vocab_sz)
        h, c = self.encoder(src)

        inp = trg[:, 0]  # First input is <SOS>
        for t in range(1, trg_len):
            out, h, c = self.decoder(inp, h, c)
            outputs[:, t] = out
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = out.argmax(1)
            inp = trg[:, t] if teacher_force else top1
        return outputs

# 4. Training
encoder = Encoder(len(src_vocab), emb_dim=16, hidden_dim=32)
decoder = Decoder(len(trg_vocab), emb_dim=16, hidden_dim=32)
model = Seq2Seq(encoder, decoder)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.03)

for epoch in range(150):
    model.train()
    optimizer.zero_grad()
    output = model(X_data, Y_data, teacher_forcing_ratio=0.5)
    loss = criterion(output[:, 1:].reshape(-1, len(trg_vocab)), Y_data[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1:3d}/150] - Loss: {loss.item():.4f}")

# 5. Translation Inference Function
def translate(model, sentence):
    model.eval()
    with torch.no_grad():
        src_toks = pad(encode(sentence, src_vocab), max_src_len)
        src_t = torch.tensor([src_toks], dtype=torch.long)
        h, c = model.encoder(src_t)
        inp = torch.tensor([trg_vocab["<SOS>"]], dtype=torch.long)
        result = []
        for _ in range(max_trg_len):
            out, h, c = model.decoder(inp, h, c)
            top1 = out.argmax(1).item()
            if top1 == trg_vocab["<EOS>"]:
                break
            result.append(inv_trg_vocab.get(top1, "<UNK>"))
            inp = torch.tensor([top1], dtype=torch.long)
    return " ".join(result)

# 6. Evaluation
print("\n--- Translation Evaluation ---")
for src, trg in pairs[:5]:
    pred = translate(model, src)
    print(f"Source: {src:<18} -> Target: {trg:<18} | Predicted: {pred}")

# Output:
# Epoch [ 50/150] - Loss: 0.0006
# Epoch [100/150] - Loss: 0.0004
# Epoch [150/150] - Loss: 0.0003
#
# --- Translation Evaluation ---
# Source: how are you        -> Target: como estas         | Predicted: como estas
# Source: i am fine          -> Target: estoy bien         | Predicted: estoy bien
# Source: good morning       -> Target: buenos dias        | Predicted: buenos dias
# Source: good night         -> Target: buenas noches      | Predicted: buenas noches
# Source: thank you          -> Target: gracias            | Predicted: gracias

# ======================================================================
