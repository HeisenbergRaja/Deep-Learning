# Program 72: ICA signal separation
# Question: A recording studio captures multiple sound sources simultaneously through microphones. Implement Independent Component Analysis to separate the mixed signals into their original independent components. (BL3 – Apply)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA

np.random.seed(42)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# 1. Generate 3 Original Independent Acoustic/Signal Sources
s1 = np.sin(2 * time)                   # Sinusoidal acoustic wave
s2 = np.sign(np.sin(3 * time))          # Square wave sound
s3 = 2 * (time % 1) - 1                 # Sawtooth audio waveform

S = np.c_[s1, s2, s3]
S /= S.std(axis=0)                      # Standardize sources

# 2. Linear Acoustic Mixing Matrix (Simulate 3 Microphones)
A = np.array([
    [1.0, 0.5, 0.5],
    [0.5, 1.0, 0.2],
    [0.3, 0.4, 1.0]
])
X = np.dot(S, A.T)                      # Observed mixed microphone recordings

# 3. Independent Component Analysis (FastICA Blind Source Separation)
ica = FastICA(n_components=3, random_state=42)
S_ica = ica.fit_transform(X)

# 4. Apply PCA for Direct Comparison (PCA only uncorrelates, fails to separate sources)
pca = PCA(n_components=3)
S_pca = pca.fit_transform(X)

# 5. Quantitative Evaluation: Absolute Pearson Correlation Matrix
corr_matrix_ica = np.zeros((3, 3))
corr_matrix_pca = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        corr_matrix_ica[i, j] = np.abs(np.corrcoef(S[:, i], S_ica[:, j])[0, 1])
        corr_matrix_pca[i, j] = np.abs(np.corrcoef(S[:, i], S_pca[:, j])[0, 1])

max_corr_ica = np.max(corr_matrix_ica, axis=1)
max_corr_pca = np.max(corr_matrix_pca, axis=1)

print("Correlation with Ground Truth Sources:")
print(f"{'Source Signal':<18} | {'ICA Correlation':<18} | {'PCA Correlation':<18}")
print("-" * 60)
for i in range(3):
    print(f"Source {i+1} (Tone {i+1})  | {max_corr_ica[i]:>16.4f} | {max_corr_pca[i]:>16.4f}")

print("\nConclusion: FastICA achieves near-perfect separation (>0.99 correlation), successfully separating non-Gaussian mixed sources where PCA fails.")

# 6. Visualization: Sources vs Mixed vs Separated Waveforms
fig, axes = plt.subplots(3, 3, figsize=(10, 6), sharex=True)
models = [S, X, S_ica]
titles = ["Original Audio Sources", "Mixed Microphone Signals", "ICA Separated Components"]
colors = ['steelblue', 'darkorange', 'forestgreen']

for row, (data, title, col) in enumerate(zip(models, titles, colors)):
    for i in range(3):
        axes[row, i].plot(time[:250], data[:250, i], color=col)
        if row == 0:
            axes[row, i].set_title(f"Channel {i+1}")
        if i == 0:
            axes[row, i].set_ylabel(title, fontsize=9)
        axes[row, i].grid(True)

plt.suptitle("Independent Component Analysis (ICA): Blind Source Separation", fontsize=12)
plt.tight_layout()
plt.show()

# Output:
# Correlation with Ground Truth Sources:
# Source Signal      | ICA Correlation    | PCA Correlation   
# ------------------------------------------------------------
# Source 1 (Tone 1)  |           0.9967 |           0.7275
# Source 2 (Tone 2)  |           0.9995 |           0.6242
# Source 3 (Tone 3)  |           0.9969 |           0.8688
#
# Conclusion: FastICA achieves near-perfect separation (>0.99 correlation), successfully separating non-Gaussian mixed sources where PCA fails.




# ======================================================================
# CO5 PROGRAMS 73 TO 80
# ======================================================================

# ======================================================================
