# Program 59: Demonstrating the Vanishing Gradient Problem in Simple RNN
# Question: A research team observes that a simple RNN fails to learn long-term dependencies in sequential data. Demonstrate the vanishing gradient problem and analyse how gradient values change across multiple time steps. (BL4 – Analyse)
# ======================================================================

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)

SEQUENCE_LENGTH = 25
HIDDEN_DIM = 16
INPUT_DIM = 8

# Synthetic sequence inputs
inputs = [torch.randn(1, INPUT_DIM) * 0.1 for _ in range(SEQUENCE_LENGTH)]

# 1. Simple RNN Setup (unrolled recurrence to track hidden gradients)
W_xh_rnn = nn.Parameter(torch.randn(HIDDEN_DIM, INPUT_DIM) * 0.1)
W_hh_rnn = nn.Parameter(torch.eye(HIDDEN_DIM) * 0.6) # Spectral radius 0.6 < 1.0
b_h_rnn = nn.Parameter(torch.zeros(HIDDEN_DIM))

h_rnn = torch.zeros(1, HIDDEN_DIM)
rnn_hidden_states = []

for t in range(SEQUENCE_LENGTH):
    # Recurrent transition: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)
    h_rnn = torch.tanh(inputs[t] @ W_xh_rnn.T + h_rnn @ W_hh_rnn.T + b_h_rnn)
    h_rnn.retain_grad()
    rnn_hidden_states.append(h_rnn)

# Loss computed solely at the final time step T = 25
target_rnn = torch.randn(1, HIDDEN_DIM)
loss_rnn = 0.5 * torch.sum((rnn_hidden_states[-1] - target_rnn) ** 2)
loss_rnn.backward()

# Extract gradient norm || dL / dh_t || for each time step
rnn_grad_norms = [rnn_hidden_states[t].grad.norm().item() for t in range(SEQUENCE_LENGTH)]

# 2. LSTM Setup for Comparative Gradient Flow Analysis
W_xi = nn.Parameter(torch.randn(HIDDEN_DIM, INPUT_DIM) * 0.1)
W_hi = nn.Parameter(torch.randn(HIDDEN_DIM, HIDDEN_DIM) * 0.1)
W_xf = nn.Parameter(torch.randn(HIDDEN_DIM, INPUT_DIM) * 0.1)
W_hf = nn.Parameter(torch.randn(HIDDEN_DIM, HIDDEN_DIM) * 0.1)
b_f = nn.Parameter(torch.ones(HIDDEN_DIM) * 2.5) # Active forget gate (f_t ~ 0.92)
W_xc = nn.Parameter(torch.randn(HIDDEN_DIM, INPUT_DIM) * 0.1)
W_hc = nn.Parameter(torch.randn(HIDDEN_DIM, HIDDEN_DIM) * 0.1)
W_xo = nn.Parameter(torch.randn(HIDDEN_DIM, INPUT_DIM) * 0.1)
W_ho = nn.Parameter(torch.randn(HIDDEN_DIM, HIDDEN_DIM) * 0.1)

c_lstm = torch.zeros(1, HIDDEN_DIM)
h_lstm = torch.zeros(1, HIDDEN_DIM)
lstm_cell_states = []

for t in range(SEQUENCE_LENGTH):
    x = inputs[t]
    i_gate = torch.sigmoid(x @ W_xi.T + h_lstm @ W_hi.T)
    f_gate = torch.sigmoid(x @ W_xf.T + h_lstm @ W_hf.T + b_f)
    c_cand = torch.tanh(x @ W_xc.T + h_lstm @ W_hc.T)
    c_lstm = f_gate * c_lstm + i_gate * c_cand
    c_lstm.retain_grad()
    lstm_cell_states.append(c_lstm)
    o_gate = torch.sigmoid(x @ W_xo.T + h_lstm @ W_ho.T)
    h_lstm = o_gate * torch.tanh(c_lstm)

loss_lstm = 0.5 * torch.sum((lstm_cell_states[-1] - target_rnn) ** 2)
loss_lstm.backward()

lstm_grad_norms = [lstm_cell_states[t].grad.norm().item() for t in range(SEQUENCE_LENGTH)]

# Display Analysis of Vanishing Gradients Across Time Steps
print("=== Gradient Value Analysis Across Time Steps (BPTT) ===")
print(f"{'Time Step (t)':<14} | {'Simple RNN ||dL/dh_t||':<24} | {'LSTM ||dL/dc_t||':<22} | {'Ratio (RNN/LSTM)':<16}")
print("=" * 82)
sampled_steps = [0, 4, 9, 14, 19, 24]
for s in sampled_steps:
    ratio = rnn_grad_norms[s] / (lstm_grad_norms[s] + 1e-12)
    print(f"t = {s+1:<10} | {rnn_grad_norms[s]:<24.6e} | {lstm_grad_norms[s]:<22.6e} | {ratio:<16.2e}")
print("=" * 82)

# Quantitative decay analysis
decay_factor_rnn = rnn_grad_norms[-1] / (rnn_grad_norms[0] + 1e-12)
decay_factor_lstm = lstm_grad_norms[-1] / (lstm_grad_norms[0] + 1e-12)

print(f"\nSimple RNN gradient decayed by a factor of {decay_factor_rnn:,.0f}x from step 25 to step 1!")
print(f"LSTM cell gradient decayed by only {decay_factor_lstm:.2f}x over the exact same span.")
print("\nAnalytical Insight (BL4 - Analyse):")
print("1. In Simple RNN: dh_T/dh_t = Prod_{k=t+1}^T diag(1 - tanh^2(a_k)) * W_hh^T.")
print("   Because the derivative of tanh is <= 1 and ||W_hh|| < 1, the gradient decays exponentially.")
print("2. In LSTM: dc_T/dc_t = Prod_{k=t+1}^T f_k. When forget gates f_k ~ 1, error gradients flow")
print("   unhindered through the additive cell state (the Constant Error Carousel).")

# Output comments:
# === Gradient Value Analysis Across Time Steps (BPTT) ===
# Time Step (t)  | Simple RNN ||dL/dh_t||   | LSTM ||dL/dc_t||       | Ratio (RNN/LSTM)
# ==================================================================================
# t = 1          | 2.052513e-05             | 2.673522e+00           | 7.68e-06        
# t = 5          | 1.586327e-04             | 2.571287e+00           | 6.17e-05        
# t = 10         | 2.051457e-03             | 2.563491e+00           | 8.00e-04        
# t = 15         | 2.645252e-02             | 2.689027e+00           | 9.84e-03        
# t = 20         | 3.415259e-01             | 3.049951e+00           | 1.12e-01        
# t = 25         | 4.415739e+00             | 4.453809e+00           | 9.91e-01        
# ==================================================================================
# 
# Simple RNN gradient decayed by a factor of 215,138x from step 25 to step 1!
# LSTM cell gradient decayed by only 1.67x over the exact same span.
# 
# Analytical Insight (BL4 - Analyse):
# 1. In Simple RNN: dh_T/dh_t = Prod_{k=t+1}^T diag(1 - tanh^2(a_k)) * W_hh^T.
#    Because the derivative of tanh is <= 1 and ||W_hh|| < 1, the gradient decays exponentially.
# 2. In LSTM: dc_T/dc_t = Prod_{k=t+1}^T f_k. When forget gates f_k ~ 1, error gradients flow
#    unhindered through the additive cell state (the Constant Error Carousel).

# ======================================================================
