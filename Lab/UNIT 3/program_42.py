# Program 42: CNN Filter Analysis for Defective Product Detection
# ==========================================================================
# Configuration                      | Parameters | Train Loss   | Test Acc 
# --------------------------------------------------------------------------
# Model 1 (3x3 Kernel, 16 Filters)   | 35346      | 0.0001       | 100.0   %
# Model 2 (5x5 Kernel, 16 Filters)   | 39698      | 0.0003       | 100.0   %
# Model 3 (3x3 Kernel, 32 Filters)   | 75202      | 0.0005       | 100.0   %
#
# Filter Analysis Observations:
# 1. Filter Size Effect: 5x5 filters expand the spatial receptive field per layer to capture broader defect textures, increasing parameter count from 35,346 to 39,698.
# 2. Filter Number Effect: Increasing filter depth from 16 to 32 doubles the feature channels, raising parameters to 75,202 to capture richer defect representations.

# ======================================================================
