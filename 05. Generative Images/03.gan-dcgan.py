"""
03. Generative Adversarial Networks (GAN & DCGAN)
=================================================

Implementation of standard Multi-Layer Perceptron GAN and Deep Convolutional GAN (DCGAN).

Architecture Diagram / Layout (DCGAN):
    Generator:
        Input z [Batch, Latent_Dim, 1, 1]
            -> ConvTranspose2d (k=7, s=1, p=0) -> BatchNorm2d -> ReLU [Batch, 128, 7, 7]
            -> ConvTranspose2d (k=4, s=2, p=1) -> BatchNorm2d -> ReLU [Batch, 64, 14, 14]
            -> ConvTranspose2d (k=4, s=2, p=1) -> Tanh [Batch, 1, 28, 28]
    Discriminator:
        Input Image [Batch, 1, 28, 28]
            -> Conv2d (k=4, s=2, p=1) -> LeakyReLU [Batch, 64, 14, 14]
            -> Conv2d (k=4, s=2, p=1) -> BatchNorm2d -> LeakyReLU [Batch, 128, 7, 7]
            -> Conv2d (k=7, s=1, p=0) -> Sigmoid [Batch, 1, 1, 1]

Key insights / educational takeaways:
    * Minimax adversarial game training pushes the generator to map simple noise distributions to complex data spaces.
    * Batch normalization and LeakyReLU are crucial for stabilizer dynamics.

Run:
    python "03.gan-dcgan.py" --variant dcgan --epochs 5
    python "03.gan-dcgan.py" --variant mlp --epochs 5
    python "03.gan-dcgan.py" --limit 2000 --epochs 2
"""

import os
import torch
import torch.nn as nn
import gen_common as mc


# --------------------------------------------------------------------------- #
# MLP Variant Models
# --------------------------------------------------------------------------- #
class MLPGenerator(nn.Module):
    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        # Flattened spatial representation
        out_flat = self.net(z)
        return out_flat.view(-1, 1, 28, 28)


class MLPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x) # Outputs raw logits


# --------------------------------------------------------------------------- #
# DCGAN Variant Models (Radford et al. 2015)
# --------------------------------------------------------------------------- #
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size: (128) x 7 x 7
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size: (64) x 14 x 14
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size: (1) x 28 x 28
        )

    def forward(self, z):
        # Expects z to be [B, latent_dim] or [B, latent_dim, 1, 1]
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # input is (1) x 28 x 28
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (64) x 14 x 14
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (128) x 7 x 7
            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=0, bias=False)
            # state size: (1) x 1 x 1
        )

    def forward(self, x):
        # squeeze to [B, 1]
        return self.net(x).view(-1, 1)


# Weights initialization as recommended in the DCGAN paper
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main():
    p = mc.build_argparser("Adversarial GAN Models")
    args = p.parse_args()

    # Learning rates should ideally be 2e-4 for stable DCGAN training
    lr_g = 2e-4 if args.lr == 1e-3 else args.lr
    lr_d = 2e-4 if args.lr == 1e-3 else args.lr

    variant = args.variant or "dcgan"
    if variant not in ["mlp", "dcgan"]:
        variant = "dcgan"

    device = mc.get_device(args.device)

    print(f"Loading {args.dataset} dataset (limit={args.limit})...")
    train_loader, _, _ = mc.get_dataloaders(
        name=args.dataset, batch_size=args.batch_size, limit=args.limit
    )

    if variant == "dcgan":
        generator = DCGANGenerator(latent_dim=args.latent_dim).to(device)
        discriminator = DCGANDiscriminator().to(device)
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        model_name = "DCGAN"
    else:
        generator = MLPGenerator(latent_dim=args.latent_dim).to(device)
        discriminator = MLPDiscriminator().to(device)
        model_name = "MLP-GAN"

    opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    n_params = sum(p.numel() for p in generator.parameters()) + sum(p.numel() for p in discriminator.parameters())
    print(f"Variant: {model_name} | Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        loss_d_accum = 0.0
        loss_g_accum = 0.0
        total_batches = 0

        for real_imgs, _ in train_loader:
            b_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            opt_d.zero_grad()

            # Pass real images: target = 1
            real_labels = torch.ones(b_size, 1, device=device)
            out_real = discriminator(real_imgs)
            loss_real = criterion(out_real, real_labels)

            # Pass fake images: target = 0
            z = torch.randn(b_size, args.latent_dim, device=device)
            fake_imgs = generator(z)
            fake_labels = torch.zeros(b_size, 1, device=device)
            out_fake = discriminator(fake_imgs.detach())
            loss_fake = criterion(out_fake, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            opt_d.step()

            # ---------------------
            #  Train Generator
            # ---------------------
            opt_g.zero_grad()

            # Generator wants discriminator to output 1 for fakes
            out_fake_g = discriminator(fake_imgs)
            loss_g = criterion(out_fake_g, real_labels)

            loss_g.backward()
            opt_g.step()

            loss_d_accum += loss_d.item()
            loss_g_accum += loss_g.item()
            total_batches += 1

        epoch_d_loss = loss_d_accum / total_batches
        epoch_g_loss = loss_g_accum / total_batches

        print(f"Epoch {epoch:2d}/{args.epochs} | loss_D {epoch_d_loss:.4f} | loss_G {epoch_g_loss:.4f}")

    print("-" * 64)

    # Save visual outputs
    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))

        # Generate a grid of random generated samples
        generator.eval()
        with torch.no_grad():
            z_random = torch.randn(64, args.latent_dim, device=device)
            gen_imgs = generator(z_random)

            gen_path = os.path.join(save_dir, f"{variant}_generated_samples.png")
            mc.save_grid_png(gen_imgs, gen_path, nrows=8, ncols=8)

        # Save latent space walk
        walk_path = os.path.join(save_dir, f"{variant}_latent_walk.png")
        mc.save_latent_walk_png(generator, walk_path, args.latent_dim, device=device)


if __name__ == "__main__":
    main()
