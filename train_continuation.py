#!/usr/bin/env python3
"""
Continuation training script - Train remaining epochs for both models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🎯 Device: {device}\n")

# ============================================================================
# COMPLETE cGAN TRAINING TO 50 EPOCHS
# ============================================================================
print("=" * 80)
print("COMPLETING cGAN TRAINING (50 EPOCHS TOTAL)")
print("=" * 80 + "\n")

from data_utils.dataloader import get_mnist_loaders
from model.cgan import Generator, Discriminator
from training.train_cgan import train_cgan
from utils.checkpoint import save_checkpoint, load_checkpoint

# Setup
z_dim = 100
num_classes = 10
img_shape = (1, 28, 28)
batch_size = 128
lr = 0.0002
epochs_cgan = 50

# Load data
print("Loading MNIST data...")
train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
print(f"✓ Train batches: {len(train_loader)}\n")

# Initialize and load from checkpoint
print("Initializing cGAN models...")
generator = Generator(z_dim=z_dim, num_classes=num_classes, img_shape=img_shape).to(device)
discriminator = Discriminator(num_classes=num_classes, img_shape=img_shape).to(device)

# Try to load epoch 40 checkpoint
try:
    load_checkpoint(generator, path="/Users/muskansandhu/Downloads/checkpoints/cgan_generator_epoch40.pt", map_location=device)
    load_checkpoint(discriminator, path="/Users/muskansandhu/Downloads/checkpoints/cgan_discriminator_epoch40.pt", map_location=device)
    print("✓ Loaded cGAN from epoch 40 checkpoint")
except Exception as e:
    print(f"⚠ Could not load checkpoint: {e}")
    print("  Training from scratch...")

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

print(f"Starting training for {epochs_cgan} epochs...\n")
train_cgan(
    generator=generator,
    discriminator=discriminator,
    dataloader=train_loader,
    optimizer_G=optimizer_G,
    optimizer_D=optimizer_D,
    criterion=criterion,
    device=device,
    z_dim=z_dim,
    num_classes=num_classes,
    epochs=epochs_cgan,
)

# Save final checkpoints
os.makedirs('./checkpoints', exist_ok=True)
save_checkpoint(generator, optimizer_G, epoch='final', name='cgan_generator', path='./checkpoints')
save_checkpoint(discriminator, optimizer_D, epoch='final', name='cgan_discriminator', path='./checkpoints')
print("\n✓ cGAN final checkpoints saved\n")

# ============================================================================
# TRAIN DIFFUSION MODEL (40 EPOCHS)
# ============================================================================
print("=" * 80)
print("TRAINING CONDITIONAL DIFFUSION (40 EPOCHS)")
print("=" * 80 + "\n")

from model.diffusion import ConditionalUNet
from training.train_diffusion import train_diffusion

timesteps = 200
epochs_diff = 40
lr_diff = 1e-4

print("Initializing Conditional Diffusion model...")
diffusion_model = ConditionalUNet(num_classes=num_classes).to(device)
print(f"✓ Model: {sum(p.numel() for p in diffusion_model.parameters()):,} parameters\n")

print(f"Starting Diffusion training for {epochs_diff} epochs...\n")
train_diffusion(
    model=diffusion_model,
    dataloader=train_loader,
    device=device,
    num_classes=num_classes,
    timesteps=timesteps,
    epochs=epochs_diff,
    lr=lr_diff,
)

# Save final checkpoint
save_checkpoint(diffusion_model, None, epoch='final', name='diffusion_model', path='./checkpoints')
print("\n✓ Diffusion final checkpoint saved\n")

print("=" * 80)
print("✅ ALL TRAINING COMPLETE")
print("=" * 80)
print("\nCheckpoints saved to ./checkpoints/")
print("Ready for evaluation!")
