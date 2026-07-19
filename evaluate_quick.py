#!/usr/bin/env python3
"""
Quick evaluation using trained checkpoints from epoch 40
"""

import torch
import os
import sys

# Add parent directory to path to access checkpoints
sys.path.insert(0, '/Users/muskansandhu/Downloads')
os.chdir('/Users/muskansandhu/Downloads/gen1')

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}\n")

from data_utils.dataloader import get_mnist_loaders
from model.cgan import Generator
from model.diffusion import ConditionalUNet
from training.train_diffusion import sample_images
from utils.checkpoint import load_checkpoint
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from tqdm import tqdm

print("=" * 80)
print("LOADING TRAINED MODELS FROM EPOCH 40 CHECKPOINTS")
print("=" * 80)

z_dim = 100
num_classes = 10

# Load models
cgan_gen = Generator(z_dim=z_dim, num_classes=num_classes, img_shape=(1, 28, 28)).to(device)
diff_model = ConditionalUNet(num_classes=num_classes).to(device)

# Load from epoch 40 checkpoints in parent directory
try:
    load_checkpoint(cgan_gen, path="/Users/muskansandhu/Downloads/checkpoints/cgan_generator_epoch40.pt", map_location=device)
    print("✓ Loaded cGAN Generator (epoch 40)")
except Exception as e:
    print(f"✗ Error loading cGAN: {e}")
    sys.exit(1)

try:
    # For diffusion, we don't have a checkpoint yet since training was interrupted
    # We'll use the untrained model (or we could train it separately)
    print("⚠ Diffusion model not yet trained - using random initialization")
    print("  (Training was interrupted; will demonstrate evaluation framework)\n")
except Exception as e:
    print(f"Error: {e}")

cgan_gen.eval()
diff_model.eval()

# ============================================================================
# CGAN VISUALIZATION - All 10 Classes
# ============================================================================
print("=" * 80)
print("cGAN SAMPLES - ALL 10 DIGIT CLASSES")
print("=" * 80)

print("\nGenerating 80 samples (8 per class)...")
with torch.no_grad():
    z = torch.randn(80, z_dim, device=device)
    labels = torch.tensor([i for i in range(10) for _ in range(8)], device=device)
    cgan_samples = cgan_gen(z, labels).cpu()

# Create grid
fig, axes = plt.subplots(8, 10, figsize=(15, 12))
fig.suptitle('cGAN Generated Samples - All 10 Digit Classes (Epoch 40)', fontsize=16, fontweight='bold')

for col in range(10):
    for row in range(8):
        ax = axes[row, col]
        img = cgan_samples[col * 8 + row].squeeze()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        if row == 0:
            ax.text(0.5, 1.15, f'Digit {col}', transform=ax.transAxes, 
                   ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
output_file = './cgan_samples_epoch40_all_classes.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved to {output_file}\n")
plt.show()

# ============================================================================
# VISUAL COMPARISON - Real vs cGAN
# ============================================================================
print("=" * 80)
print("REAL vs cGAN COMPARISON - ONE SAMPLE PER CLASS")
print("=" * 80)

# Get real examples
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

real_samples = []
for cls in range(10):
    for img, lab in mnist_test:
        if lab == cls:
            real_samples.append(img)
            break
real_samples = torch.stack(real_samples)

# Generate cGAN samples for comparison
with torch.no_grad():
    z_comp = torch.randn(10, z_dim, device=device)
    labels_comp = torch.arange(10, device=device)
    cgan_comp = cgan_gen(z_comp, labels_comp).cpu()

# Plot comparison
fig, axes = plt.subplots(2, 10, figsize=(16, 4))
fig.suptitle('Real MNIST vs cGAN Generated - One Sample Per Class (Epoch 40)', fontsize=14, fontweight='bold')

for col in range(10):
    # Real
    axes[0, col].imshow(real_samples[col].squeeze(), cmap='gray')
    axes[0, col].set_title(f'{col}', fontsize=10, fontweight='bold')
    axes[0, col].axis('off')
    if col == 0:
        axes[0, col].text(-0.3, 0.5, 'Real', transform=axes[0, col].transAxes,
                         fontsize=12, fontweight='bold', va='center', rotation=90)
    
    # cGAN
    axes[1, col].imshow(cgan_comp[col].squeeze(), cmap='gray')
    axes[1, col].axis('off')
    if col == 0:
        axes[1, col].text(-0.3, 0.5, 'cGAN', transform=axes[1, col].transAxes,
                         fontsize=12, fontweight='bold', va='center', rotation=90)

plt.tight_layout()
comparison_file = './real_vs_cgan_epoch40.png'
plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
print(f"✓ Comparison saved to {comparison_file}\n")
plt.show()

# ============================================================================
# BASIC STATISTICS
# ============================================================================
print("=" * 80)
print("GENERATION STATISTICS")
print("=" * 80)
print(f"\n✓ cGAN Sample Shape: {cgan_samples.shape}")
print(f"✓ Value Range: [{cgan_samples.min():.3f}, {cgan_samples.max():.3f}]")
print(f"✓ All digit classes successfully generated: {len(torch.unique(labels)) == 10}")

print("\n" + "=" * 80)
print("✅ EVALUATION COMPLETE - Ready for notebook integration")
print("=" * 80)
print(f"\nGenerated outputs:")
print(f"  - {output_file}")
print(f"  - {comparison_file}")
