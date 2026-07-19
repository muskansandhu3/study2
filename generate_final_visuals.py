#!/usr/bin/env python3
"""
Quick evaluation using epoch 30 trained cGAN checkpoint
Generates best visualizations for submission
"""

import torch
import os
import sys
os.chdir('/Users/muskansandhu/Downloads/gen1')

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}\n")

from model.cgan import Generator
from utils.checkpoint import load_checkpoint
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

print("=" * 80)
print("GENERATING FINAL VISUALIZATIONS USING EPOCH 30 cGAN")
print("=" * 80 + "\n")

z_dim = 100
num_classes = 10

# Load cGAN from epoch 30
cgan_gen = Generator(z_dim=z_dim, num_classes=num_classes, img_shape=(1, 28, 28)).to(device)

try:
    # Try local checkpoints first
    if os.path.exists('./checkpoints/cgan_generator_epoch30.pt'):
        load_checkpoint(cgan_gen, path="./checkpoints/cgan_generator_epoch30.pt", map_location=device)
        print("✓ Loaded cGAN Generator (epoch 30 from local)")
    else:
        load_checkpoint(cgan_gen, path="/Users/muskansandhu/Downloads/checkpoints/cgan_generator_epoch30.pt", map_location=device)
        print("✓ Loaded cGAN Generator (epoch 30 from parent)")
except Exception as e:
    print(f"✗ Error loading cGAN: {e}")
    sys.exit(1)

cgan_gen.eval()

# ============================================================================
# HIGH-QUALITY VISUALIZATION - All 10 Classes
# ============================================================================
print("\nGenerating high-quality grid with all 10 digit classes (80 samples)...")

with torch.no_grad():
    z = torch.randn(80, z_dim, device=device)
    labels = torch.tensor([i for i in range(10) for _ in range(8)], device=device)
    cgan_samples = cgan_gen(z, labels).cpu()

# Create a professional grid
fig, axes = plt.subplots(8, 10, figsize=(16, 13))
fig.suptitle('cGAN Generated Handwritten Digits (Epoch 30) - All Classes 0-9', 
             fontsize=18, fontweight='bold', y=0.995)

for col in range(10):
    for row in range(8):
        ax = axes[row, col]
        img = cgan_samples[col * 8 + row].squeeze()
        ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Add subtle border
        for spine in ax.spines.values():
            spine.set_edgecolor('lightgray')
            spine.set_linewidth(0.5)
        
        # Label columns with digit class
        if row == 0:
            ax.set_title(f'Class {col}', fontsize=12, fontweight='bold', pad=5)

plt.tight_layout()
output_file = 'cgan_samples_final_all_classes.png'
plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
print(f"✓ Saved to {output_file}")
plt.close()

# ============================================================================
# COMPARISON: Real vs cGAN
# ============================================================================
print("Generating Real vs cGAN comparison (one sample per class)...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Get one real sample per class
real_samples = []
for cls in range(10):
    for img, lab in mnist_test:
        if lab == cls:
            real_samples.append(img)
            break
real_samples = torch.stack(real_samples)

# Generate cGAN samples
with torch.no_grad():
    z_comp = torch.randn(10, z_dim, device=device)
    labels_comp = torch.arange(10, device=device)
    cgan_comp = cgan_gen(z_comp, labels_comp).cpu()

# Create side-by-side grid
fig, axes = plt.subplots(2, 10, figsize=(18, 5))
fig.suptitle('Real MNIST (Top) vs cGAN Generated (Bottom) - Digit Classes 0-9', 
             fontsize=16, fontweight='bold')

for col in range(10):
    # Real
    axes[0, col].imshow(real_samples[col].squeeze(), cmap='gray', vmin=-1, vmax=1)
    axes[0, col].set_title(f'{col}', fontsize=11, fontweight='bold')
    axes[0, col].set_xticks([])
    axes[0, col].set_yticks([])
    if col == 0:
        axes[0, col].set_ylabel('Real MNIST', fontsize=12, fontweight='bold')
    
    # cGAN
    axes[1, col].imshow(cgan_comp[col].squeeze(), cmap='gray', vmin=-1, vmax=1)
    axes[1, col].set_xticks([])
    axes[1, col].set_yticks([])
    if col == 0:
        axes[1, col].set_ylabel('cGAN Generated', fontsize=12, fontweight='bold')

plt.tight_layout()
comparison_file = 'real_vs_cgan_final.png'
plt.savefig(comparison_file, dpi=200, bbox_inches='tight', facecolor='white')
print(f"✓ Saved to {comparison_file}")
plt.close()

# ============================================================================
# STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("GENERATION STATISTICS")
print("=" * 80)
print(f"\nSamples Generated: {len(cgan_samples)}")
print(f"Classes Represented: {len(torch.unique(labels))}")
print(f"Value Range: [{cgan_samples.min():.3f}, {cgan_samples.max():.3f}]")
print(f"Mean Value: {cgan_samples.mean():.3f}")
print(f"Std Dev: {cgan_samples.std():.3f}")

print("\n✅ HIGH-QUALITY VISUALIZATIONS GENERATED")
print("\nFiles created:")
print(f"  1. {output_file} (80 samples, all classes)")
print(f"  2. {comparison_file} (Real vs Generated)")
print("\n" + "=" * 80)
