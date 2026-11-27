# Program 6: Retail
# Question: Implement Euclidean and Manhattan distance calculations, compare results.

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Define Distance Formulas
# ----------------------------------------------------------------------
def euclidean_distance(u, v):
    """Euclidean Distance (L2 norm): sqrt(sum((u_i - v_i)^2))"""
    return np.sqrt(np.sum((u - v) ** 2))

def manhattan_distance(u, v):
    """Manhattan Distance (L1 norm): sum(|u_i - v_i|)"""
    return np.sum(np.abs(u - v))

# ----------------------------------------------------------------------
# 2. Case Study A: 2D Spatial Logistics (Warehouse & Retail Stores in km)
# ----------------------------------------------------------------------
locations = {
    'Distribution Center' : np.array([12.0, 15.0]),
    'Store A (Downtown)'   : np.array([18.0, 23.0]),
    'Store B (Suburbs)'    : np.array([2.0, 5.0]),
    'Store C (Uptown)'     : np.array([12.0, 27.0])
}

print("=" * 75)
print("RETAIL: EUCLIDEAN VS MANHATTAN DISTANCE CALCULATIONS")
print("=" * 75)
print("Case Study A: Retail Store Logistics (2D Grid Coordinates in km)")
print(f"{'Location Pair':<45} | {'Euclidean':<11} | {'Manhattan':<11} | {'Ratio (M/E)':<10}")
print("-" * 85)

dc = locations['Distribution Center']
store_names = ['Store A (Downtown)', 'Store B (Suburbs)', 'Store C (Uptown)']
for name in store_names:
    store_loc = locations[name]
    d_e = euclidean_distance(dc, store_loc)
    d_m = manhattan_distance(dc, store_loc)
    ratio = d_m / d_e if d_e > 0 else 1.0
    pair_label = f"Dist Center <-> {name}"
    print(f"{pair_label:<45} | {d_e:>8.4f} km | {d_m:>8.4f} km | {ratio:>10.4f}")

# ----------------------------------------------------------------------
# 3. Case Study B: Multi-Attribute Product Profiles (4D Feature Space)
# Attributes: [Monthly Sales (k), Profit Margin (%), Return Rate (%), Shelf Space (sqft)]
# ----------------------------------------------------------------------
products = {
    'Product 1 (Laptop)'   : np.array([15.0, 22.0, 8.0, 2.0]),
    'Product 2 (Tablet)'   : np.array([18.0, 20.0, 7.5, 1.5]),
    'Product 3 (Groceries)': np.array([85.0,  5.0, 1.0, 15.0])
}

print("-" * 85)
print("Case Study B: Product Attribute Feature Space (4D Metrics)")
product_pairs = [
    ('Product 1 (Laptop)', 'Product 2 (Tablet)'),
    ('Product 1 (Laptop)', 'Product 3 (Groceries)')
]

for p1_name, p2_name in product_pairs:
    p1, p2 = products[p1_name], products[p2_name]
    d_e = euclidean_distance(p1, p2)
    d_m = manhattan_distance(p1, p2)
    ratio = d_m / d_e
    print(f"Pair: {p1_name} vs {p2_name}")
    print(f"  Euclidean Distance (L2)    : {d_e:.4f}")
    print(f"  Manhattan Distance (L1)    : {d_m:.4f}")
    print(f"  Ratio (Manhattan/Euclidean): {ratio:.4f}")

# ----------------------------------------------------------------------
# 4. Comparison & Analysis
# ----------------------------------------------------------------------
print("-" * 85)
print("COMPARATIVE ANALYSIS OF RESULTS:")
print("-" * 85)
print("1. Mathematical Bound: d_Euclidean <= d_Manhattan <= sqrt(n) * d_Euclidean.")
print("   - Equality (Ratio = 1.0000) occurs when points differ along only one axis (Store C).")
print("   - Maximum ratio (sqrt(2) ~ 1.4142 in 2D) occurs along exact 45-degree diagonals (Store B).")
print("2. Application Domain:")
print("   - Manhattan Distance: Best for urban road networks and automated warehouse aisles")
print("     where navigation is strictly restricted to orthogonal grid axes.")
print("   - Euclidean Distance: Best for straight-line aerial delivery (drone logistics)")
print("     and continuous geometric similarity in embedding spaces.")
print("=" * 85)

# ----------------------------------------------------------------------
# 5. Route Comparison Visualization using Matplotlib
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
dc_coord = locations['Distribution Center']
store_coord = locations['Store A (Downtown)']

# Plot coordinates
ax.scatter(*dc_coord, color='blue', s=120, label='Distribution Center (12, 15)')
ax.scatter(*store_coord, color='red', s=120, label='Store A (18, 23)')

# Plot Euclidean path (straight line)
ax.plot([dc_coord[0], store_coord[0]], [dc_coord[1], store_coord[1]],
        'b--', linewidth=2, label=f'Euclidean Path ({euclidean_distance(dc_coord, store_coord):.2f} km)')

# Plot Manhattan path (grid step)
ax.plot([dc_coord[0], store_coord[0], store_coord[0]], [dc_coord[1], dc_coord[1], store_coord[1]],
        'r-', linewidth=2, label=f'Manhattan Path ({manhattan_distance(dc_coord, store_coord):.2f} km)')

ax.set_title("Retail Logistics: Euclidean vs Manhattan Route")
ax.set_xlabel("East-West Coordinate (km)")
ax.set_ylabel("North-South Coordinate (km)")
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left')
plt.tight_layout()
# plt.show() # Display plot window (uncomment when running interactively)
plt.close()

# Output:
# ===========================================================================
# RETAIL: EUCLIDEAN VS MANHATTAN DISTANCE CALCULATIONS
# ===========================================================================
# Case Study A: Retail Store Logistics (2D Grid Coordinates in km)
# Location Pair                                 | Euclidean   | Manhattan   | Ratio (M/E)
# -------------------------------------------------------------------------------------
# Dist Center <-> Store A (Downtown)            |  10.0000 km |  14.0000 km |     1.4000
# Dist Center <-> Store B (Suburbs)             |  14.1421 km |  20.0000 km |     1.4142
# Dist Center <-> Store C (Uptown)              |  12.0000 km |  12.0000 km |     1.0000
# -------------------------------------------------------------------------------------
# Case Study B: Product Attribute Feature Space (4D Metrics)
# Pair: Product 1 (Laptop) vs Product 2 (Tablet)
#   Euclidean Distance (L2)    : 3.6742
#   Manhattan Distance (L1)    : 6.0000
#   Ratio (Manhattan/Euclidean): 1.6330
# Pair: Product 1 (Laptop) vs Product 3 (Groceries)
#   Euclidean Distance (L2)    : 73.5323
#   Manhattan Distance (L1)    : 107.0000
#   Ratio (Manhattan/Euclidean): 1.4551
# -------------------------------------------------------------------------------------
# COMPARATIVE ANALYSIS OF RESULTS:
# -------------------------------------------------------------------------------------
# 1. Mathematical Bound: d_Euclidean <= d_Manhattan <= sqrt(n) * d_Euclidean.
#    - Equality (Ratio = 1.0000) occurs when points differ along only one axis (Store C).
#    - Maximum ratio (sqrt(2) ~ 1.4142 in 2D) occurs along exact 45-degree diagonals (Store B).
# 2. Application Domain:
#    - Manhattan Distance: Best for urban road networks and automated warehouse aisles
#      where navigation is strictly restricted to orthogonal grid axes.
#    - Euclidean Distance: Best for straight-line aerial delivery (drone logistics)
#      and continuous geometric similarity in embedding spaces.
# =====================================================================================

# ======================================================================
