# Program 5: E-commerce
# Question: Calculate dot product and cosine similarity between customer preference vectors.

import numpy as np

# ----------------------------------------------------------------------
# 1. Customer Preference Profiles
# ----------------------------------------------------------------------
# Preference vectors across 5 product categories:
# [Electronics, Apparel, Books, Home & Kitchen, Sports]
categories = ['Electronics', 'Apparel', 'Books', 'Home & Kitchen', 'Sports']

customers = {
    'Customer 1 (Alice)'  : np.array([5, 4, 1, 0, 2], dtype=float),
    'Customer 2 (Bob)'    : np.array([10, 8, 2, 0, 4], dtype=float), # Exactly 2x Alice
    'Customer 3 (Charlie)': np.array([1, 1, 5, 4, 0], dtype=float), # Book & Home lover
    'Customer 4 (Diana)'  : np.array([4, 5, 0, 1, 3], dtype=float)  # High apparel & sports
}

print("=" * 70)
print("E-COMMERCE: DOT PRODUCT & COSINE SIMILARITY ANALYSIS")
print("=" * 70)
print(f"Categories: {categories}")
print("-" * 70)
for name, vec in customers.items():
    norm = np.linalg.norm(vec)
    print(f"{name:<22} : Vector = {vec} | Magnitude (L2) = {norm:.4f}")

# ----------------------------------------------------------------------
# 2. Dot Product & Cosine Similarity Computation
# ----------------------------------------------------------------------
def compute_vector_metrics(u, v):
    # Dot Product: u . v = sum(u_i * v_i)
    dot_product = np.dot(u, v)
    # Magnitudes (Euclidean Norms): ||u||, ||v||
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    # Cosine Similarity: (u . v) / (||u|| * ||v||)
    cosine_sim = dot_product / (norm_u * norm_v) if (norm_u * norm_v) > 0 else 0.0
    # Angle in degrees
    angle_rad = np.arccos(np.clip(cosine_sim, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return dot_product, cosine_sim, angle_deg

pairs_to_compare = [
    ('Customer 1 (Alice)', 'Customer 2 (Bob)'),
    ('Customer 1 (Alice)', 'Customer 3 (Charlie)'),
    ('Customer 1 (Alice)', 'Customer 4 (Diana)'),
    ('Customer 3 (Charlie)', 'Customer 4 (Diana)')
]

print("-" * 70)
print(f"{'Customer Pair':<43} | {'Dot Product':<11} | {'Cosine Sim':<10} | {'Angle':<8}")
print("-" * 70)

for c1_name, c2_name in pairs_to_compare:
    v1, v2 = customers[c1_name], customers[c2_name]
    dot_prod, cos_sim, deg = compute_vector_metrics(v1, v2)
    pair_label = f"{c1_name} vs {c2_name}"
    print(f"{pair_label:<43} | {dot_prod:>11.2f} | {cos_sim:>10.4f} | {deg:>6.2f}°")

# ----------------------------------------------------------------------
# 3. Analytical Interpretation
# ----------------------------------------------------------------------
print("-" * 70)
print("ANALYTICAL COMPARISON & E-COMMERCE INSIGHTS:")
print("-" * 70)
print("1. Alice vs Bob:")
print("   - Dot Product is large (92.00) because Bob's transaction volume is 2x Alice's.")
print("   - Cosine Similarity is exactly 1.0000 (0.00° angle), proving their preference")
print("     distribution is identical regardless of spending volume.")
print("2. Alice vs Charlie:")
print("   - Cosine Similarity is low (0.3148, angle 71.65°), indicating opposing interests.")
print("3. Why Cosine Similarity is standard for Recommendation Systems:")
print("   - Dot product is sensitive to magnitude (frequency/volume bias).")
print("   - Cosine similarity normalizes vector lengths, measuring purely directional")
print("     preference alignment (taste similarity).")
print("=" * 70)

# Output:
# ======================================================================
# E-COMMERCE: DOT PRODUCT & COSINE SIMILARITY ANALYSIS
# ======================================================================
# Categories: ['Electronics', 'Apparel', 'Books', 'Home & Kitchen', 'Sports']
# ----------------------------------------------------------------------
# Customer 1 (Alice)     : Vector = [5. 4. 1. 0. 2.] | Magnitude (L2) = 6.7823
# Customer 2 (Bob)       : Vector = [10.  8.  2.  0.  4.] | Magnitude (L2) = 13.5647
# Customer 3 (Charlie)   : Vector = [1. 1. 5. 4. 0.] | Magnitude (L2) = 6.5574
# Customer 4 (Diana)     : Vector = [4. 5. 0. 1. 3.] | Magnitude (L2) = 7.1414
# ----------------------------------------------------------------------
# Customer Pair                               | Dot Product | Cosine Sim | Angle   
# ----------------------------------------------------------------------
# Customer 1 (Alice) vs Customer 2 (Bob)      |       92.00 |     1.0000 |   0.00°
# Customer 1 (Alice) vs Customer 3 (Charlie)  |       14.00 |     0.3148 |  71.65°
# Customer 1 (Alice) vs Customer 4 (Diana)    |       46.00 |     0.9497 |  18.25°
# Customer 3 (Charlie) vs Customer 4 (Diana)  |       13.00 |     0.2776 |  73.88°
# ----------------------------------------------------------------------
# ANALYTICAL COMPARISON & E-COMMERCE INSIGHTS:
# ----------------------------------------------------------------------
# 1. Alice vs Bob:
#    - Dot Product is large (92.00) because Bob's transaction volume is 2x Alice's.
#    - Cosine Similarity is exactly 1.0000 (0.00° angle), proving their preference
#      distribution is identical regardless of spending volume.
# 2. Alice vs Charlie:
#    - Cosine Similarity is low (0.3148, angle 71.65°), indicating opposing interests.
# 3. Why Cosine Similarity is standard for Recommendation Systems:
#    - Dot product is sensitive to magnitude (frequency/volume bias).
#    - Cosine similarity normalizes vector lengths, measuring purely directional
#      preference alignment (taste similarity).
# ======================================================================

# ======================================================================
