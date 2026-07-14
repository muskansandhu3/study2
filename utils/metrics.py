import torch
from torchmetrics.image.fid import FrechetInceptionDistance


def compute_fid(real_imgs, fake_imgs, device="cpu", batch_size=128):
    """Compute FID between two normalized MNIST tensors in [-1, 1]."""
    fid = FrechetInceptionDistance(normalize=True).to(device)

    real_imgs_rgb = torch.clamp((real_imgs.repeat(1, 3, 1, 1) + 1) / 2, 0, 1)
    fake_imgs_rgb = torch.clamp((fake_imgs.repeat(1, 3, 1, 1) + 1) / 2, 0, 1)

    for i in range(0, real_imgs_rgb.size(0), batch_size):
        fid.update(real_imgs_rgb[i:i + batch_size].to(device), real=True)
    for i in range(0, fake_imgs_rgb.size(0), batch_size):
        fid.update(fake_imgs_rgb[i:i + batch_size].to(device), real=False)

    return fid.compute().item()
