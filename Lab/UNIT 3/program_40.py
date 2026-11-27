# Program 40: Max Pooling vs Average Pooling on Extracted Image Features
# Question: A wildlife monitoring organization uses CNNs to identify animals from camera images. Implement max pooling and average pooling and compare their effects on extracted image features. (BL5 – Evaluate)

import torch
import torch.nn.functional as F

# Simulate an extracted image feature map from a wildlife monitoring camera
# Dimensions: 1 batch, 1 channel, 8x8 spatial feature map
# Contains sharp animal features (tiger stripes / leopard spots / horns) with background foliage
torch.manual_seed(42)
animal_feature_map = torch.zeros(1, 1, 8, 8)

# Distinct animal contour and high-contrast stripe features
animal_feature_map[0, 0, 2:6, 3] = 0.90   # Sharp vertical stripe
animal_feature_map[0, 0, 4, 2:6] = 0.85   # Horizontal animal contour
animal_feature_map[0, 0, 1, 6] = 0.95     # Bright animal eye / reflective marker
# Low-level ambient vegetation noise
animal_feature_map += 0.10 * torch.rand(1, 1, 8, 8)

print("Original Wildlife Camera Feature Map (8x8):")
print(animal_feature_map[0, 0].numpy().round(3))

# 1. Apply Max Pooling (2x2 kernel, stride 2)
max_pooled = F.max_pool2d(animal_feature_map, kernel_size=2, stride=2)

# 2. Apply Average Pooling (2x2 kernel, stride 2)
avg_pooled = F.avg_pool2d(animal_feature_map, kernel_size=2, stride=2)

print("\nMax Pooled Feature Map (4x4):")
print(max_pooled[0, 0].numpy().round(3))

print("\nAverage Pooled Feature Map (4x4):")
print(avg_pooled[0, 0].numpy().round(3))

# Detailed Patch-by-Patch Comparison on a representative 2x2 region containing an animal stripe
patch_2x2 = animal_feature_map[0, 0, 2:4, 2:4]
patch_max = max_pooled[0, 0, 1, 1].item()
patch_avg = avg_pooled[0, 0, 1, 1].item()

print("\n=== Local 2x2 Feature Patch Analysis ===")
print("Input 2x2 Patch (Animal Stripe against Foliage):")
print(patch_2x2.numpy().round(3))
print(f"Max Pooling Output:     {patch_max:.4f} (Captures the dominant feature activation)")
print(f"Average Pooling Output: {patch_avg:.4f} (Averages feature with low background noise)")

# Statistical Comparison Table
stats = [
    ("Max Activation", animal_feature_map.max().item(), max_pooled.max().item(), avg_pooled.max().item()),
    ("Min Activation", animal_feature_map.min().item(), max_pooled.min().item(), avg_pooled.min().item()),
    ("Mean Activation", animal_feature_map.mean().item(), max_pooled.mean().item(), avg_pooled.mean().item()),
    ("Std Deviation", animal_feature_map.std().item(), max_pooled.std().item(), avg_pooled.std().item()),
    ("Contrast (Max-Min)", animal_feature_map.max().item() - animal_feature_map.min().item(),
     max_pooled.max().item() - max_pooled.min().item(), avg_pooled.max().item() - avg_pooled.min().item())
]

print("\n=== Quantitative Comparison: Max vs Average Pooling ===")
print(f"{'Statistic':<22} | {'Original Feature':<18} | {'Max Pooling (2x2)':<18} | {'Average Pooling (2x2)':<18}")
print("-" * 85)
for stat_name, orig, mp, ap in stats:
    print(f"{stat_name:<22} | {orig:<18.4f} | {mp:<18.4f} | {ap:<18.4f}")

print("\nComparative Evaluation for Wildlife Monitoring:")
print("1. Feature Salience: Max pooling preserves the strongest activation (0.95+), retaining sharp animal features")
print("   (whiskers, stripes, eyes) even in low-contrast, camouflaged forest scenes.")
print("2. Contrast Retention: Max pooling maintains high contrast (Max - Min = " + f"{max_pooled.max().item() - max_pooled.min().item():.4f}" + "),")
print("   whereas Average pooling dampens contrast to " + f"{avg_pooled.max().item() - avg_pooled.min().item():.4f}" + " by diluting sharp signals with background noise.")
print("3. Recommendation: Max Pooling is superior for animal pattern detection and species identification.")
print("   Average Pooling is preferred for smoothing and aggregate background context.")

# Output:
# Original Wildlife Camera Feature Map (8x8):
# [[8.800e-02 9.200e-02 3.800e-02 9.600e-02 3.900e-02 6.000e-02 2.600e-02
#   7.900e-02]
#  [9.400e-02 1.300e-02 9.300e-02 5.900e-02 8.700e-02 5.700e-02 1.024e+00
#   4.300e-02]
#  [8.900e-02 5.700e-02 2.700e-02 9.630e-01 2.700e-02 4.400e-02 3.000e-02
#   8.300e-02]
#  [1.100e-02 2.700e-02 3.600e-02 9.200e-01 5.500e-02 1.000e-03 9.500e-02
#   8.000e-03]
#  [8.900e-02 5.800e-02 8.840e-01 9.310e-01 9.080e-01 9.400e-01 5.500e-02
#   3.400e-02]
#  [6.300e-02 3.600e-02 7.100e-02 9.950e-01 7.900e-02 2.800e-02 7.900e-02
#   5.900e-02]
#  [7.500e-02 2.000e-02 1.000e-03 3.100e-02 1.200e-02 9.100e-02 6.400e-02
#   7.100e-02]
#  [6.600e-02 4.900e-02 8.900e-02 1.400e-02 5.300e-02 1.600e-02 6.500e-02
#   3.300e-02]]
# 
# Max Pooled Feature Map (4x4):
# [[0.094 0.096 0.087 1.024]
#  [0.089 0.963 0.055 0.095]
#  [0.089 0.995 0.94  0.079]
#  [0.075 0.089 0.091 0.071]]
# 
# Average Pooled Feature Map (4x4):
# [[0.072 0.072 0.061 0.293]
#  [0.046 0.486 0.032 0.054]
#  [0.062 0.72  0.489 0.057]
#  [0.052 0.034 0.043 0.058]]
# 
# === Local 2x2 Feature Patch Analysis ===
# Input 2x2 Patch (Animal Stripe against Foliage):
# [[0.027 0.963]
#  [0.036 0.92 ]]
# Max Pooling Output:     0.9627 (Captures the dominant feature activation)
# Average Pooling Output: 0.4863 (Averages feature with low background noise)
# 
# === Quantitative Comparison: Max vs Average Pooling ===
# Statistic              | Original Feature   | Max Pooling (2x2)  | Average Pooling (2x2)
# -------------------------------------------------------------------------------------
# Max Activation         | 1.0241             | 1.0241             | 0.7201            
# Min Activation         | 0.0005             | 0.0547             | 0.0316            
# Mean Activation        | 0.1644             | 0.3082             | 0.1644            
# Std Deviation          | 0.2992             | 0.4013             | 0.2135            
# Contrast (Max-Min)     | 1.0236             | 0.9694             | 0.6885            
# 
# Comparative Evaluation for Wildlife Monitoring:
# 1. Feature Salience: Max pooling preserves the strongest activation (0.95+), retaining sharp animal features
#    (whiskers, stripes, eyes) even in low-contrast, camouflaged forest scenes.
# 2. Contrast Retention: Max pooling maintains high contrast (Max - Min = 0.9694),
#    whereas Average pooling dampens contrast to 0.6885 by diluting sharp signals with background noise.
# 3. Recommendation: Max Pooling is superior for animal pattern detection and species identification.
#    Average Pooling is preferred for smoothing and aggregate background context.



# ======================================================================
# CO3 PROGRAMS 41 TO 48
# ======================================================================
