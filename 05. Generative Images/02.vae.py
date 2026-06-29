"""
02. Variational Autoencoder (VAE)
=================================

A Variational Autoencoder (VAE) utilizing fully connected mapping layers.

Architecture Diagram / Layout:
    Input [Batch, 1, 28, 28] -> Flatten [Batch, 784]
        -> Encoder: Linear [784, 256] -> ReLU
        -> Parameter Projections: mu [Batch, Latent_Dim], logvar [Batch, Latent_Dim]
        -> Reparameterization Trick: z = mu + exp(0.5 * logvar) * eps (where eps ~ N(0, I))
        -> Decoder: Linear [Latent_Dim, 256] -> ReLU -> Linear [256, 784] -> Tanh
        -> Output [Batch, 1, 28, 28]

Key insights / educational takeaways:
    * The reparameterization trick allows gradients to flow backwards through a stochastic bottleneck.
    * The Kullback-Leibler (KL) divergence penalizes deviation from a standard Gaussian prior, forcing a smooth latent space.

Run:
    python "02.vae.py" --epochs 5
    python "02.vae.py" --limit 2000 --epochs 2
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gen_common as mc


class VAE(nn.Module):
    """Variational Autoencoder with normal distribution parameter projections."""
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder_base = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Tanh()
        )

    def encode(self, x):
        h = self.encoder_base(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def main():
    p = mc.build_argparser("Variational Autoencoder Model")
    args = p.parse_args()

    device = mc.get_device(args.device)

    print(f"Loading {args.dataset} dataset (limit={args.limit})...")
    train_loader, test_loader, _ = mc.get_dataloaders(
        name=args.dataset, batch_size=args.batch_size, limit=args.limit
    )

    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_recon = 0.0
        running_kl = 0.0
        total_samples = 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)

            # Sum losses over feature dimensions, average over batch
            recon_loss = F.mse_loss(recon, x, reduction="sum")
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (recon_loss + kl_loss) / x.size(0)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_recon += (recon_loss.item() / x.size(0)) * x.size(0)
            running_kl += (kl_loss.item() / x.size(0)) * x.size(0)
            total_samples += x.size(0)

        train_loss = running_loss / total_samples
        train_recon = running_recon / total_samples
        train_kl = running_kl / total_samples

        # Eval
        model.eval()
        test_loss = 0.0
        test_samples = 0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                recon_loss = F.mse_loss(recon, x, reduction="sum")
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = (recon_loss + kl_loss) / x.size(0)
                test_loss += loss.item() * x.size(0)
                test_samples += x.size(0)
        val_loss = test_loss / test_samples

        print(f"Epoch {epoch:2d}/{args.epochs} | loss {train_loss:.2f} (recon {train_recon:.2f}, kl {train_kl:.2f}) | test_loss {val_loss:.2f}")

    print("-" * 64)

    # Save visual outputs
    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))

        model.eval()
        with torch.no_grad():
            # 1. Reconstructions
            x_test_batch, _ = next(iter(test_loader))
            x_test_batch = x_test_batch[:8].to(device)
            recons, _, _ = model(x_test_batch)
            comparison = torch.cat([x_test_batch, recons], dim=0)

            comp_path = os.path.join(save_dir, "vae_reconstructions.png")
            mc.save_grid_png(comparison, comp_path, nrows=2, ncols=8)

            # 2. Random Generative Sampling
            z_random = torch.randn(64, args.latent_dim, device=device)
            gen_imgs = model.decode(z_random)

            gen_path = os.path.join(save_dir, "vae_generated_samples.png")
            mc.save_grid_png(gen_imgs, gen_path, nrows=8, ncols=8)

        # 3. Latent space walk
        walk_path = os.path.join(save_dir, "vae_latent_walk.png")
        mc.save_latent_walk_png(model.decode, walk_path, args.latent_dim, device=device)


if __name__ == "__main__":
    main()
