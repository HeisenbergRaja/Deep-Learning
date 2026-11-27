# Program 4: Search engine PageRank
# Question: Construct transition matrix, compute eigenvalues and eigenvectors to identify important web pages.

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Web Link Graph Representation
# ----------------------------------------------------------------------
# Web pages: 4 pages (A, B, C, D)
# Directed links:
# - Page A -> Page B, Page C
# - Page B -> Page C
# - Page C -> Page A
# - Page D -> Page A, Page B, Page C
pages = ['Page A', 'Page B', 'Page C', 'Page D']
n = len(pages)

# Adjacency matrix: adj[i, j] = 1 if Page i has an outgoing link to Page j
adj_matrix = np.array([
    [0, 1, 1, 0],  # Page A links to B, C
    [0, 0, 1, 0],  # Page B links to C
    [1, 0, 0, 0],  # Page C links to A
    [1, 1, 1, 0]   # Page D links to A, B, C
], dtype=float)

# ----------------------------------------------------------------------
# 2. Construct Column-Stochastic Transition Matrix M
# ----------------------------------------------------------------------
# M[j, i] represents the probability of transitioning from page i to page j
# Each column i sums to 1.0
out_degrees = np.sum(adj_matrix, axis=1)
M = np.zeros((n, n), dtype=float)
for i in range(n):
    if out_degrees[i] > 0:
        M[:, i] = adj_matrix[i, :] / out_degrees[i]
    else:
        M[:, i] = 1.0 / n  # Equal probability for dangling nodes

print("=" * 65)
print("SEARCH ENGINE PAGERANK: TRANSITION MATRIX & EIGEN ANALYSIS")
print("=" * 65)
print("Transition Probability Matrix M (Column-Stochastic):")
print(np.round(M, 4))
print("Column Sums Verification:", np.round(np.sum(M, axis=0), 4))

# ----------------------------------------------------------------------
# 3. Construct Google Matrix G with Damping Factor (d = 0.85)
# ----------------------------------------------------------------------
# G = d * M + (1 - d) / n * E (handles random surfer teleportation)
d = 0.85
E = np.ones((n, n)) / n
G = d * M + (1.0 - d) * E

print("-" * 65)
print(f"Google Matrix G (Damping Factor d = {d}):")
print(np.round(G, 4))

# ----------------------------------------------------------------------
# 4. Compute Eigenvalues and Eigenvectors
# ----------------------------------------------------------------------
eigenvalues, eigenvectors = np.linalg.eig(G)

# The stationary distribution corresponds to the principal eigenvalue lambda = 1.0
lambda_1_index = np.argmin(np.abs(eigenvalues - 1.0))
dominant_eigenvalue = eigenvalues[lambda_1_index]
stationary_eigenvector = np.real(eigenvectors[:, lambda_1_index])

# Normalize the eigenvector so the probabilities sum to 1.0
pagerank_scores = stationary_eigenvector / np.sum(stationary_eigenvector)

print("-" * 65)
print(f"Dominant Eigenvalue (lambda) : {dominant_eigenvalue.real:.4f}")
print("PageRank Scores (Normalized Eigenvector for lambda = 1):")
print("-" * 65)

# Sort pages in descending order of PageRank score
ranked_indices = np.argsort(pagerank_scores)[::-1]
for rank, idx in enumerate(ranked_indices, start=1):
    print(f"  Rank {rank}: {pages[idx]:<8} | Score: {pagerank_scores[idx]:.4f} ({pagerank_scores[idx]*100:.2f}%)")

print("-" * 65)
print(f"Most Important Web Page Identified: {pages[ranked_indices[0]]}")
print("=" * 65)

# ----------------------------------------------------------------------
# 5. Visualization of PageRank Scores using Matplotlib
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 5))
bars = plt.bar(pages, pagerank_scores, color=['#4285F4', '#EA4335', '#FBBC05', '#34A853'], edgecolor='black')
plt.title("PageRank Distribution across Web Pages (Principal Eigenvector)")
plt.xlabel("Web Pages")
plt.ylabel("PageRank Score (Stationary Probability)")
plt.ylim(0, max(pagerank_scores) * 1.2)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
# plt.show() # Display plot window (uncomment when running interactively)
plt.close()

# Output:
# =================================================================
# SEARCH ENGINE PAGERANK: TRANSITION MATRIX & EIGEN ANALYSIS
# =================================================================
# Transition Probability Matrix M (Column-Stochastic):
# [[0.     0.     1.     0.3333]
#  [0.5    0.     0.     0.3333]
#  [0.5    1.     0.     0.3333]
#  [0.     0.     0.     0.    ]]
# Column Sums Verification: [1. 1. 1. 1.]
# -----------------------------------------------------------------
# Google Matrix G (Damping Factor d = 0.85):
# [[0.0375 0.0375 0.8875 0.3208]
#  [0.4625 0.0375 0.0375 0.3208]
#  [0.4625 0.8875 0.0375 0.3208]
#  [0.0375 0.0375 0.0375 0.0375]]
# -----------------------------------------------------------------
# Dominant Eigenvalue (lambda) : 1.0000
# PageRank Scores (Normalized Eigenvector for lambda = 1):
# -----------------------------------------------------------------
#   Rank 1: Page C   | Score: 0.3825 (38.25%)
#   Rank 2: Page A   | Score: 0.3732 (37.32%)
#   Rank 3: Page B   | Score: 0.2068 (20.68%)
#   Rank 4: Page D   | Score: 0.0375 (3.75%)
# -----------------------------------------------------------------
# Most Important Web Page Identified: Page C
# =================================================================

# ======================================================================
