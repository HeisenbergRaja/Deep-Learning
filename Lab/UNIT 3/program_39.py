# Program 39: Convolution and Pooling Operations for Surveillance Image Feature Extraction
# Question: A security company wants to extract important features from surveillance images. Implement convolution and pooling operations on an input image and analyse the resulting feature maps. (BL4 – Analyse)

import torch
import torch.nn.functional as F

# Simulate a 32x32 surveillance image with security features:
# Perimeter boundary, security fence, entrance gate, building structure, and an intruder highlight.
image = torch.zeros(1, 1, 32, 32)
image[0, 0, 6:10, :] = 0.8       # Horizontal perimeter boundary
image[0, 0, :, 14:18] = 0.9      # Vertical security fence
image[0, 0, 20:28, 20:28] = 0.6  # Building structure
image[0, 0, 8, 16] = 1.0         # Highly reflective intruder point target

print(f"Surveillance Input Image Shape: {list(image.shape)} (Batch=1, Channel=1, H=32, W=32)")

# Define classic surveillance feature extraction filters (3x3 kernels)
# 1. Horizontal Edge Detection (Sobel-H) - detects perimeter fences & barriers
sobel_h = torch.tensor([
    [-1.0, -2.0, -1.0],
    [ 0.0,  0.0,  0.0],
    [ 1.0,  2.0,  1.0]
]).view(1, 1, 3, 3)

# 2. Vertical Edge Detection (Sobel-V) - detects pillars & vertical gate posts
sobel_v = torch.tensor([
    [-1.0,  0.0,  1.0],
    [-2.0,  0.0,  2.0],
    [-1.0,  0.0,  1.0]
]).view(1, 1, 3, 3)

# 3. Laplacian Filter - omnidirectional edge & contour detector
laplacian = torch.tensor([
    [ 0.0,  1.0,  0.0],
    [ 1.0, -4.0,  1.0],
    [ 0.0,  1.0,  0.0]
]).view(1, 1, 3, 3)

# 4. Sharpening Filter - highlights fine perimeter textures & intruder details
sharpen = torch.tensor([
    [ 0.0, -1.0,  0.0],
    [-1.0,  5.0, -1.0],
    [ 0.0, -1.0,  0.0]
]).view(1, 1, 3, 3)

# Combine into a 4-channel convolution kernel bank: shape (4, 1, 3, 3)
filter_bank = torch.cat([sobel_h, sobel_v, laplacian, sharpen], dim=0)
filter_names = ["Sobel Horizontal", "Sobel Vertical", "Laplacian Edge", "Sharpen Filter"]

# 1. Convolution Operation (padding=1 preserves spatial dimensions)
conv_feature_maps = F.conv2d(image, filter_bank, padding=1)

# Apply ReLU to retain positive feature activations (rectified feature maps)
relu_feature_maps = F.relu(conv_feature_maps)

# 2. Pooling Operation (Max Pooling: 2x2 window with stride 2 for spatial downsampling)
pooled_feature_maps = F.max_pool2d(relu_feature_maps, kernel_size=2, stride=2)

print(f"Convolved Feature Maps Shape:   {list(conv_feature_maps.shape)} (Channels=4, H=32, W=32)")
print(f"Pooled Feature Maps Shape:      {list(pooled_feature_maps.shape)} (Channels=4, H=16, W=16)")

print("\n=== Quantitative Analysis of Surveillance Feature Maps ===")
print(f"{'Filter Type':<18} | {'Min Val':<8} | {'Max Val':<8} | {'Mean Act':<10} | {'Variance':<10} | {'Sparsity (% zeros)':<18}")
print("-" * 85)

for idx, name in enumerate(filter_names):
    f_conv = relu_feature_maps[0, idx]
    f_pool = pooled_feature_maps[0, idx]
    
    min_v = f_pool.min().item()
    max_v = f_pool.max().item()
    mean_v = f_pool.mean().item()
    var_v = f_pool.var().item()
    sparsity = (f_pool == 0).float().mean().item() * 100

    print(f"{name:<18} | {min_v:<8.2f} | {max_v:<8.2f} | {mean_v:<10.4f} | {var_v:<10.4f} | {sparsity:<17.1f}%")

print("\nFeature Map Analysis Insights:")
print("1. Sobel Horizontal: Strongly activates on horizontal boundaries (Max = 3.60), isolating perimeter barriers.")
print("2. Sobel Vertical: Strongly activates on vertical structures (Max = 3.60), isolating vertical fence posts.")
print("3. Laplacian: Highlights boundary transitions omnidirectionally, exhibiting high sparsity.")
print("4. Max Pooling: Downsamples spatial dimensions from 32x32 to 16x16 (75% data reduction) while preserving the strongest edge activations.")

# Output:
# Surveillance Input Image Shape: [1, 1, 32, 32] (Batch=1, Channel=1, H=32, W=32)
# Convolved Feature Maps Shape:   [1, 4, 32, 32] (Channels=4, H=32, W=32)
# Pooled Feature Maps Shape:      [1, 4, 16, 16] (Channels=4, H=16, W=16)
# 
# === Quantitative Analysis of Surveillance Feature Maps ===
# Filter Type        | Min Val  | Max Val  | Mean Act   | Variance   | Sparsity (% zeros)
# -------------------------------------------------------------------------------------
# Sobel Horizontal   | 0.00     | 3.60     | 0.4820     | 1.1955     | 71.1             %
# Sobel Vertical     | 0.00     | 3.60     | 0.5281     | 1.4373     | 81.2             %
# Laplacian Edge     | 0.00     | 1.70     | 0.2168     | 0.1518     | 72.3             %
# Sharpen Filter     | 0.00     | 2.70     | 0.4906     | 0.6300     | 70.3             %
# 
# Feature Map Analysis Insights:
# 1. Sobel Horizontal: Strongly activates on horizontal boundaries (Max = 3.60), isolating perimeter barriers.
# 2. Sobel Vertical: Strongly activates on vertical structures (Max = 3.60), isolating vertical fence posts.
# 3. Laplacian: Highlights boundary transitions omnidirectionally, exhibiting high sparsity.
# 4. Max Pooling: Downsamples spatial dimensions from 32x32 to 16x16 (75% data reduction) while preserving the strongest edge activations.

# ======================================================================
