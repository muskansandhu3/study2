# Conditional Generative Handwriting for CAPTCHA Robustness

This project trains two class-conditional generative models on MNIST:
- **Conditional GAN (cGAN)**
- **Conditional Diffusion Model (DDPM-style Conditional U-Net)**

It then compares them with:
- Visual quality/diversity checks
- **FID** (Frechet Inception Distance)
- Downstream utility: classifier trained on synthetic data, tested on real MNIST

## Project Layout

- [data_utils/dataloader.py](data_utils/dataloader.py) - MNIST loading helpers
- [model/cgan.py](model/cgan.py) - Conditional Generator/Discriminator
- [model/diffusion.py](model/diffusion.py) - Conditional U-Net denoiser
- [training/train_cgan.py](training/train_cgan.py) - cGAN training loop + checkpoints
- [training/train_diffusion.py](training/train_diffusion.py) - diffusion training and sampling
- [utils/checkpoint.py](utils/checkpoint.py) - save/load checkpoints
- [utils/visualize.py](utils/visualize.py) - batch visualization
- [utils/metrics.py](utils/metrics.py) - FID helper
- [00_data_preparation.ipynb](00_data_preparation.ipynb)
- [01_cGAN_training.ipynb](01_cGAN_training.ipynb)
- [02_diffusion_training.ipynb](02_diffusion_training.ipynb)
- [03_evaluation.ipynb](03_evaluation.ipynb)

## Environment

Use your workspace venv:

- Python: 3.11+
- Installed: `torch`, `torchvision`, `torchmetrics`, `tqdm`, `matplotlib`, `numpy`, `scipy`

## Run Order (Notebooks)

1. [00_data_preparation.ipynb](00_data_preparation.ipynb)
2. [01_cGAN_training.ipynb](01_cGAN_training.ipynb)
3. [02_diffusion_training.ipynb](02_diffusion_training.ipynb)
4. [03_evaluation.ipynb](03_evaluation.ipynb)

## Checkpoints

Notebooks now save/load checkpoints from `./checkpoints`:
- `cgan_generator_epochfinal.pt`
- `cgan_discriminator_epochfinal.pt`
- `diffusion_unet_epochfinal.pt`

## Smoke Test Status

A minimal smoke test was run successfully on a tiny MNIST subset:
- cGAN: 1 epoch train pass completed
- Diffusion: 1 epoch train pass completed
- Diffusion sampler produced valid `(N, 1, 28, 28)` tensors

## Notes

- Full training with default notebook epochs can take significant time, especially diffusion.
- FID is computed on balanced real/synthetic subsets and should be interpreted alongside visual inspection + downstream classifier accuracy.
