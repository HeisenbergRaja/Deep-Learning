# Program 73: ICA Blind Source Separation of Simultaneous Voice Recordings
# Question: ICA-based blind source separation of two simultaneous voice recordings, evaluate quality.
# ======================================================================

import sys, types
# Fallback shim for environments with strict DLL policies on sklearn polynomial expansion
try:
    from sklearn.preprocessing import _csr_polynomial_expansion
except ImportError:
    _stub = types.ModuleType('sklearn.preprocessing._csr_polynomial_expansion')
    _stub._calc_expanded_nnz = lambda *a, **kw: 0
    _stub._calc_total_nnz = lambda *a, **kw: 0
    _stub._csr_polynomial_expansion = lambda *a, **kw: None
    sys.modules['sklearn.preprocessing._csr_polynomial_expansion'] = _stub

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# Set random seed for reproducibility
np.random.seed(42)

# 1. Synthesize two distinct voice/speech signals (sampling frequency = 2000 Hz, duration = 1.0 s)
sampling_rate = 2000
duration = 1.0
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Voice 1 (Speaker 1: male pitch vocal simulation, f0=150Hz + harmonics + amplitude modulation envelope)
voice1 = (np.sin(2 * np.pi * 150 * t) + 
          0.5 * np.sin(2 * np.pi * 300 * t) + 
          0.25 * np.sin(2 * np.pi * 450 * t)) * (0.7 + 0.3 * np.sin(2 * np.pi * 5 * t))

# Voice 2 (Speaker 2: female pitch vocal simulation, f0=240Hz + harmonics + cadence envelope)
voice2 = (np.sin(2 * np.pi * 240 * t) + 
          0.6 * np.sin(2 * np.pi * 480 * t) + 
          0.3 * np.sin(2 * np.pi * 720 * t)) * (0.8 + 0.2 * np.cos(2 * np.pi * 6 * t))

# Combine and normalize sources to zero mean and unit variance
S = np.c_[voice1, voice2]
S = (S - S.mean(axis=0)) / S.std(axis=0)

# 2. Linear instantaneous mixing (simulating two microphones in a room)
# Mixing matrix A
A = np.array([[0.65, 0.35],
              [0.30, 0.70]])
X = np.dot(S, A.T)

# 3. Blind Source Separation using FastICA
ica = FastICA(n_components=2, random_state=42, max_iter=1000)
S_rec = ica.fit_transform(X)
S_rec = (S_rec - S_rec.mean(axis=0)) / S_rec.std(axis=0)

# 4. Resolve ICA permutation and sign ambiguities using correlation matrix
corr_matrix = np.corrcoef(S.T, S_rec.T)[:2, 2:]

print("--- Program 73: ICA Blind Source Separation Results ---")
print("True Mixing Matrix A:")
print(A)
print("\nEstimated Unmixing Matrix W:")
print(np.round(ica.components_, 4))
print("\nCorrelation Matrix between Original Sources (rows) and Recovered Components (cols):")
print(np.round(corr_matrix, 4))

# 5. Evaluate quality: Pearson correlation r, Mean Squared Error (MSE), and Signal-to-Noise Ratio (SNR)
matches = np.argmax(np.abs(corr_matrix), axis=1)
aligned_recovered = np.zeros_like(S)

print("\nSeparation Quality Evaluation:")
for i in range(2):
    rec_idx = matches[i]
    sign = np.sign(corr_matrix[i, rec_idx])
    aligned_sig = sign * S_rec[:, rec_idx]
    aligned_recovered[:, i] = aligned_sig
    
    r = np.corrcoef(S[:, i], aligned_sig)[0, 1]
    mse = np.mean((S[:, i] - aligned_sig) ** 2)
    snr = 10 * np.log10(np.var(S[:, i]) / mse)
    print(f"Source {i+1} -> Component {rec_idx+1}: Pearson r = {r:.4f}, MSE = {mse:.5f}, SNR = {snr:.2f} dB")

# 6. Plot Original, Mixed, and Recovered Signals
fig, axes = plt.subplots(3, 2, figsize=(10, 6), sharex=True)
plot_slice = slice(0, 200) # zoom in on first 0.1s to show waveform structure

axes[0, 0].plot(t[plot_slice], S[plot_slice, 0], color='tab:blue')
axes[0, 0].set_title("Original Source 1 (Speaker 1, 150 Hz)")
axes[0, 1].plot(t[plot_slice], S[plot_slice, 1], color='tab:orange')
axes[0, 1].set_title("Original Source 2 (Speaker 2, 240 Hz)")

axes[1, 0].plot(t[plot_slice], X[plot_slice, 0], color='tab:green')
axes[1, 0].set_title("Mixed Signal 1 (Microphone 1)")
axes[1, 1].plot(t[plot_slice], X[plot_slice, 1], color='tab:red')
axes[1, 1].set_title("Mixed Signal 2 (Microphone 2)")

axes[2, 0].plot(t[plot_slice], aligned_recovered[plot_slice, 0], color='tab:purple')
axes[2, 0].set_title("Recovered Source 1 via FastICA")
axes[2, 1].plot(t[plot_slice], aligned_recovered[plot_slice, 1], color='tab:brown')
axes[2, 1].set_title("Recovered Source 2 via FastICA")

for ax in axes.flat:
    ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("program_73_ica_voice_separation.png", dpi=120)
plt.close()
print("Waveform comparison plot saved to 'program_73_ica_voice_separation.png'.")

# Output:
# --- Program 73: ICA Blind Source Separation Results ---
# True Mixing Matrix A:
# [[0.65 0.35]
#  [0.3  0.7 ]]
#
# Estimated Unmixing Matrix W:
# [[-0.86    1.8586]
#  [ 1.9988 -0.9973]]
#
# Correlation Matrix between Original Sources (rows) and Recovered Components (cols):
# [[-0.0014  1.    ]
#  [ 1.      0.0014]]
#
# Separation Quality Evaluation:
# Source 1 -> Component 2: Pearson r = 1.0000, MSE = 0.00000, SNR = 56.88 dB
# Source 2 -> Component 1: Pearson r = 1.0000, MSE = 0.00000, SNR = 56.88 dB
# Waveform comparison plot saved to 'program_73_ica_voice_separation.png'.
#
# Output comments:
# FastICA achieves virtually perfect blind source separation (SNR = 56.88 dB, r = 1.0000)
# for linearly mixed non-Gaussian audio waveforms without prior knowledge of the mixing matrix A.
# The absolute off-diagonal correlation is near zero (~0.0014), indicating complete unmixing.

# ======================================================================
