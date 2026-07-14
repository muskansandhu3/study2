import os

import torch

from utils.checkpoint import save_checkpoint


def train_cgan(
    generator,
    discriminator,
    dataloader,
    optimizer_G,
    optimizer_D,
    criterion,
    device,
    z_dim,
    num_classes,
    epochs=50,
    checkpoint_dir="../checkpoints",
    save_every=10,
):
    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            batch_size = imgs.size(0)

            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, z_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            gen_imgs = generator(z, gen_labels)
            g_loss = criterion(discriminator(gen_imgs, gen_labels), valid)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(imgs, labels), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        print(f"[Epoch {epoch + 1}/{epochs}] D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")

        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model=generator,
                optimizer=optimizer_G,
                epoch=epoch + 1,
                loss=g_loss.item(),
                name="cgan_generator",
                path=checkpoint_dir,
            )
            save_checkpoint(
                model=discriminator,
                optimizer=optimizer_D,
                epoch=epoch + 1,
                loss=d_loss.item(),
                name="cgan_discriminator",
                path=checkpoint_dir,
            )

    save_checkpoint(model=generator, optimizer=optimizer_G, epoch="final", loss=g_loss.item(), name="cgan_generator", path=checkpoint_dir)
    save_checkpoint(model=discriminator, optimizer=optimizer_D, epoch="final", loss=d_loss.item(), name="cgan_discriminator", path=checkpoint_dir)
