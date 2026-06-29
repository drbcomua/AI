"""
01. Autoencoder (AE)
====================

A standard undercomplete Autoencoder using fully connected (MLP) layers.

Architecture Diagram / Layout:
    Input [Batch, 1, 28, 28] -> Flatten [Batch, 784]
        -> Encoder: Linear [784, 256] -> ReLU -> Linear [256, Latent_Dim] (z)
        -> Decoder: Linear [Latent_Dim, 256] -> ReLU -> Linear [256, 784] -> Tanh
        -> Output [Batch, 1, 28, 28]

Key insights / educational takeaways:
    * The bottleneck forces the model to compress representations, learning the most salient features.
    * Autoencoders perform dimensionality reduction, finding a non-linear manifold mapping of data distribution.

Run:
    python "01.autoencoder.py" --epochs 5
    python "01.autoencoder.py" --limit 2000 --epochs 2
"""

import os
import torch
import torch.nn as nn
import gen_common as mc


class Autoencoder(nn.Module):
    """Undercomplete fully connected autoencoder."""
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Tanh() # Matches pixel normalization range [-1, 1]
        )

    def forward(self, x):
        z = self.encoder(x)
        out_flat = self.decoder(z)
        return out_flat.view(-1, 1, 28, 28)

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)


def main():
    p = mc.build_argparser("MLP Autoencoder Model")
    args = p.parse_args()

    device = mc.get_device(args.device)

    print(f"Loading {args.dataset} dataset (limit={args.limit})...")
    train_loader, test_loader, _ = mc.get_dataloaders(
        name=args.dataset, batch_size=args.batch_size, limit=args.limit
    )

    model = Autoencoder(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        train_loss = running_loss / total_samples

        # Eval on test
        model.eval()
        test_loss = 0.0
        test_samples = 0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                recon = model(x)
                loss = criterion(recon, x)
                test_loss += loss.item() * x.size(0)
                test_samples += x.size(0)
        val_loss = test_loss / test_samples

        print(f"Epoch {epoch:2d}/{args.epochs} | train_mse {train_loss:.6f} | test_mse {val_loss:.6f}")

    print("-" * 64)

    # Save visual outputs
    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))

        # Generate reconstructions comparison (16 samples: original in top row, reconstructed in bottom)
        model.eval()
        with torch.no_grad():
            x_test_batch, _ = next(iter(test_loader))
            x_test_batch = x_test_batch[:8].to(device)
            recons = model(x_test_batch)

            # Stack original and recons
            comparison = torch.cat([x_test_batch, recons], dim=0) # [16, 1, 28, 28]

            comp_path = os.path.join(save_dir, "ae_reconstructions.png")
            mc.save_grid_png(comparison, comp_path, nrows=2, ncols=8)

        # Save a latent walk
        walk_path = os.path.join(save_dir, "ae_latent_walk.png")
        mc.save_latent_walk_png(model.decode, walk_path, args.latent_dim, device=device)


if __name__ == "__main__":
    main()
