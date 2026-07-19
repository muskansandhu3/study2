#!/usr/bin/env python3
"""
Train cGAN and Diffusion models with proper output logging.
This script trains both models to completion and saves checkpoints.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Setup device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🎯 Device: {device}\n")

# ============================================================================
# PART 1: TRAIN cGAN
# ============================================================================
print("=" * 80)
print("PART 1: TRAINING cGAN (50 EPOCHS)")
print("=" * 80 + "\n")

from data_utils.dataloader import get_mnist_loaders
from model.cgan import Generator, Discriminator
from training.train_cgan import train_cgan
from utils.checkpoint import save_checkpoint

# Hyperparameters
z_dim = 100
num_classes = 10
img_shape = (1, 28, 28)
batch_size = 128
lr = 0.0002
epochs_cgan = 50

# Load data
print("Loading MNIST data...")
train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
print(f"✓ Train batches: {len(train_loader)} | Test batches: {len(test_loader)}\n")

# Initialize models
print("Initializing cGAN models...")
generator = Generator(z_dim=z_dim, num_classes=num_classes, img_shape=img_shape).to(device)
discriminator = Discriminator(num_classes=num_classes, img_shape=img_shape).to(device)
print(f"✓ Generator: {sum(p.numel() for p in generator.parameters()):,} parameters")
print(f"✓ Discriminator: {sum(p.numel() for p in discriminator.parameters()):,} parameters\n")

# Setup optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Train
print(f"Starting cGAN training for {epochs_cgan} epochs...\n")
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

# Save checkpoints
os.makedirs('./checkpoints', exist_ok=True)
save_checkpoint(generator, optimizer_G, epoch='final', name='cgan_generator', path='./checkpoints')
save_checkpoint(discriminator, optimizer_D, epoch='final', name='cgan_discriminator', path='./checkpoints')
print("\n✓ cGAN checkpoints saved\n")

# ============================================================================
# PART 2: TRAIN DIFFUSION MODEL
# ============================================================================
print("=" * 80)
print("PART 2: TRAINING CONDITIONAL DIFFUSION (40 EPOCHS)")
print("=" * 80 + "\n")

from model.diffusion import ConditionalUNet
from training.train_diffusion import train_diffusion

# Hyperparameters
timesteps = 200
epochs_diff = 40
lr_diff = 1e-4

# Initialize diffusion model
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

# Save checkpoint
save_checkpoint(diffusion_model, None, epoch='final', name='diffusion_model', path='./checkpoints')
print("\n✓ Diffusion checkpoint saved\n")

print("=" * 80)
print("✅ ALL TRAINING COMPLETE")
print("=" * 80)
