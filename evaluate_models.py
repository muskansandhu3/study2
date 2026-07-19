#!/usr/bin/env python3
"""
Evaluation script: Compare cGAN vs Diffusion models
- Compute FID scores
- Generate visual comparisons
- Train downstream classifiers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
import os

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}\n")

# Import from local modules
from data_utils.dataloader import get_mnist_loaders
from model.cgan import Generator, Discriminator
from model.diffusion import ConditionalUNet
from training.train_diffusion import sample_images
from utils.checkpoint import load_checkpoint

print("=" * 80)
print("LOADING TRAINED MODELS")
print("=" * 80)

# Load checkpoints
z_dim = 100
num_classes = 10
img_shape = (1, 28, 28)

cgan_gen = Generator(z_dim=z_dim, num_classes=num_classes, img_shape=img_shape).to(device)
diff_model = ConditionalUNet(num_classes=num_classes).to(device)

os.makedirs("./checkpoints", exist_ok=True)

try:
    load_checkpoint(cgan_gen, path="./checkpoints/cgan_generator_epochfinal.pt", map_location=device)
    print("✓ Loaded cGAN Generator")
except Exception as e:
    print(f"⚠ Could not load cGAN checkpoint: {e}")

try:
    load_checkpoint(diff_model, path="./checkpoints/diffusion_model_epochfinal.pt", map_location=device)
    print("✓ Loaded Diffusion Model")
except Exception as e:
    print(f"⚠ Could not load Diffusion checkpoint: {e}")

cgan_gen.eval()
diff_model.eval()
print()

# ============================================================================
# FID Computation
# ============================================================================
print("=" * 80)
print("COMPUTING FID SCORES")
print("=" * 80)

def compute_fid(real_imgs, fake_imgs, device="cuda", batch_size=128):
    """Compute FID safely in GPU batches"""
    fid_metric = FrechetInceptionDistance(normalize=True).to(device)
    fid_metric.reset()

    def batched_update(imgs, real_flag):
        for i in tqdm(range(0, imgs.size(0), batch_size), desc=f"FID {'real' if real_flag else 'fake'}"):
            batch = imgs[i:i + batch_size].to(device, non_blocking=True)
            fid_metric.update(batch, real=real_flag)

    # Convert to RGB and normalize to [0, 1]
    real_imgs_rgb = (real_imgs.repeat(1, 3, 1, 1) + 1) / 2
    fake_imgs_rgb = (fake_imgs.repeat(1, 3, 1, 1) + 1) / 2
    real_imgs_rgb = torch.clamp(real_imgs_rgb, 0, 1)
    fake_imgs_rgb = torch.clamp(fake_imgs_rgb, 0, 1)

    batched_update(real_imgs_rgb, real_flag=True)
    batched_update(fake_imgs_rgb, real_flag=False)

    return fid_metric.compute().item()

# Build balanced evaluation sets (100 per class = 1000 total)
n_per_class = 100
labels_fid = torch.arange(num_classes, device=device).repeat_interleave(n_per_class)

# Real MNIST subset
print("\nPreparing real MNIST subset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

real_buckets = {i: [] for i in range(num_classes)}
for img, lab in mnist_test:
    if len(real_buckets[lab]) < n_per_class:
        real_buckets[lab].append(img)
    if all(len(v) == n_per_class for v in real_buckets.values()):
        break

real_subset = torch.cat([torch.stack(real_buckets[i]) for i in range(num_classes)], dim=0)
print(f"✓ Real subset shape: {real_subset.shape}")

# Generate cGAN subset
print("Generating cGAN samples for FID...")
with torch.no_grad():
    z = torch.randn(len(labels_fid), z_dim, device=device)
    cgan_subset = cgan_gen(z, labels_fid).cpu()
print(f"✓ cGAN subset shape: {cgan_subset.shape}")

# Generate Diffusion subset
print("Generating Diffusion samples for FID...")
with torch.no_grad():
    diff_subset, _ = sample_images(
        diff_model,
        device=device,
        num_samples=len(labels_fid),
        num_classes=num_classes,
        timesteps=200,
        class_labels=labels_fid,
    )
    diff_subset = diff_subset.cpu()
print(f"✓ Diffusion subset shape: {diff_subset.shape}\n")

# Compute FID scores
print("Computing cGAN FID...")
cgan_fid = compute_fid(real_subset, cgan_subset, device=device, batch_size=128)
print(f"\n✓ cGAN FID: {cgan_fid:.4f}")

print("\nComputing Diffusion FID...")
diff_fid = compute_fid(real_subset, diff_subset, device=device, batch_size=128)
print(f"\n✓ Diffusion FID: {diff_fid:.4f}")

print("\n" + "=" * 80)
print("FID ANALYSIS")
print("=" * 80)
print(f"cGAN FID: {cgan_fid:.2f}")
print(f"Diffusion FID: {diff_fid:.2f}")

if cgan_fid < diff_fid:
    print(f"→ cGAN achieved better FID (improvement: {diff_fid - cgan_fid:.2f})")
else:
    print(f"→ Diffusion achieved better FID (improvement: {cgan_fid - diff_fid:.2f})")
print("(Lower FID indicates more realistic and diverse samples)\n")

# ============================================================================
# Visual Comparison
# ============================================================================
print("=" * 80)
print("GENERATING VISUAL COMPARISON GRID")
print("=" * 80)

print("Sampling one image per class from each model...")
with torch.no_grad():
    z_vis = torch.randn(num_classes, z_dim, device=device)
    labels_vis = torch.arange(num_classes, device=device)
    cgan_vis = cgan_gen(z_vis, labels_vis).cpu()
    
    diff_vis, _ = sample_images(
        diff_model,
        device=device,
        num_samples=num_classes,
        num_classes=num_classes,
        timesteps=200,
        class_labels=labels_vis,
    )
    diff_vis = diff_vis.cpu()

# Get real examples
real_vis = []
for cls in range(num_classes):
    for img, lab in mnist_test:
        if lab == cls:
            real_vis.append(img)
            break
real_vis = torch.stack(real_vis)

# Plot 3-row grid
fig, axes = plt.subplots(3, num_classes, figsize=(16, 6))
fig.suptitle('Real vs cGAN vs Diffusion - All Digit Classes', fontsize=14, fontweight='bold')

for col in range(num_classes):
    # Row 0: Real
    axes[0, col].imshow(real_vis[col].squeeze(), cmap='gray')
    axes[0, col].set_title(f'{col}', fontsize=10, fontweight='bold')
    axes[0, col].axis('off')
    if col == 0:
        axes[0, col].text(-0.3, 0.5, 'Real', transform=axes[0, col].transAxes, 
                          fontsize=12, fontweight='bold', va='center', rotation=90)
    
    # Row 1: cGAN
    axes[1, col].imshow(cgan_vis[col].squeeze(), cmap='gray')
    axes[1, col].axis('off')
    if col == 0:
        axes[1, col].text(-0.3, 0.5, 'cGAN', transform=axes[1, col].transAxes, 
                          fontsize=12, fontweight='bold', va='center', rotation=90)
    
    # Row 2: Diffusion
    axes[2, col].imshow(diff_vis[col].squeeze(), cmap='gray')
    axes[2, col].axis('off')
    if col == 0:
        axes[2, col].text(-0.3, 0.5, 'Diffusion', transform=axes[2, col].transAxes, 
                          fontsize=12, fontweight='bold', va='center', rotation=90)

plt.tight_layout()
plt.savefig('./real_cgan_diffusion_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved comparison grid to ./real_cgan_diffusion_comparison.png\n")
plt.show()

# ============================================================================
# Downstream Utility Evaluation
# ============================================================================
print("=" * 80)
print("DOWNSTREAM UTILITY: Classifier Training on Synthetic Data")
print("=" * 80)

# Generate large balanced datasets
n_synthetic_per_class = 100
total_samples = num_classes * n_synthetic_per_class
labels_balanced = torch.arange(num_classes, device=device).repeat_interleave(n_synthetic_per_class)

print(f"\nGenerating {total_samples} balanced samples per model...")

with torch.no_grad():
    z_syn = torch.randn(total_samples, z_dim, device=device)
    cgan_syn = cgan_gen(z_syn, labels_balanced).cpu()
    
    diff_syn, _ = sample_images(
        diff_model,
        device=device,
        num_samples=total_samples,
        num_classes=num_classes,
        timesteps=200,
        class_labels=labels_balanced,
    )
    diff_syn = diff_syn.cpu()

cgan_labels_syn = labels_balanced.cpu()
diff_labels_syn = labels_balanced.cpu()

print(f"✓ cGAN: {cgan_syn.shape} | Diffusion: {diff_syn.shape}\n")

# Define classifier
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train_classifier(model, loader, device, epochs=20):
    """Train classifier on synthetic data"""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = criterion(preds, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1:2d}/{epochs}: loss = {avg_loss:.4f}")

    return model

def test_accuracy(model, loader, device):
    """Test classifier on real MNIST"""
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            correct += (preds.argmax(1) == labels).sum().item()
            total += len(labels)
    return correct / total

# Create data loaders
cgan_dataset = TensorDataset(cgan_syn, cgan_labels_syn)
diff_dataset = TensorDataset(diff_syn, diff_labels_syn)

cgan_loader = DataLoader(cgan_dataset, batch_size=128, shuffle=True)
diff_loader = DataLoader(diff_dataset, batch_size=128, shuffle=True)

# Get real test set
mnist_real = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_real, batch_size=256, shuffle=False)

# Train classifiers
print("Training classifier on cGAN synthetic data...")
cnn_cgan = SimpleCNN().to(device)
cnn_cgan = train_classifier(cnn_cgan, cgan_loader, device, epochs=20)

print("\nTraining classifier on Diffusion synthetic data...")
cnn_diff = SimpleCNN().to(device)
cnn_diff = train_classifier(cnn_diff, diff_loader, device, epochs=20)

# Test on real MNIST
print("\n" + "=" * 80)
print("TESTING ON REAL MNIST")
print("=" * 80)
cgan_acc = test_accuracy(cnn_cgan, test_loader, device)
diff_acc = test_accuracy(cnn_diff, test_loader, device)

print(f"\n📊 RESULTS:")
print(f"  • Classifier trained on cGAN data → Real MNIST accuracy: {cgan_acc:.2%}")
print(f"  • Classifier trained on Diffusion data → Real MNIST accuracy: {diff_acc:.2%}")

if cgan_acc > diff_acc:
    improvement = (cgan_acc - diff_acc) * 100
    print(f"\n✓ cGAN synthetic data generalized better by {improvement:.1f}%")
elif diff_acc > cgan_acc:
    improvement = (diff_acc - cgan_acc) * 100
    print(f"\n✓ Diffusion synthetic data generalized better by {improvement:.1f}%")
else:
    print("\n✓ Both models produced equally useful synthetic data")

print("\n" + "=" * 80)
print("✅ EVALUATION COMPLETE")
print("=" * 80)
