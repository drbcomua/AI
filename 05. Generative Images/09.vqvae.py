"""
09. VQ-VAE — Vector-Quantized VAE (van den Oord et al., 2017)
============================================================

A VAE whose latent space is **discrete**: the encoder output is snapped to the
nearest vector in a learned codebook of K embeddings, so each image becomes a
small grid of integer codes. This discreteness is the foundation of modern
token-based image generation (DALL·E, VQGAN, ...).

Architecture Diagram / Layout:
    Encoder: [B,1,28,28] -> conv/stride -> z_e [B, D, 7, 7]
    Quantize: each spatial D-vector -> nearest codebook entry -> z_q [B, D, 7, 7]
              (straight-through estimator copies gradients z_q -> z_e)
    Decoder: z_q -> convT -> reconstruction [B,1,28,28]
    Loss = MSE(recon, x) + ||sg[z_e]-e||^2 + beta*||z_e-sg[e]||^2

Key insights / educational takeaways:
    * Discrete codes + straight-through gradients: backprop through a hard argmin.
    * Codebook + commitment losses keep encoder outputs near codebook vectors.
    * VQ-VAE has no built-in prior; to *generate* you need a prior over the codes.
      Here we sample codes from their per-position training marginal (a weak prior);
      a real system fits an autoregressive prior (e.g. PixelCNN, script 07) instead.

Run:
    python "09.vqvae.py" --epochs 5
    python "09.vqvae.py" --limit 2000 --epochs 2
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gen_common as mc


class VectorQuantizer(nn.Module):
    def __init__(self, num_codes=64, dim=32, beta=0.25):
        super().__init__()
        self.beta = beta
        self.embedding = nn.Embedding(num_codes, dim)
        self.embedding.weight.data.uniform_(-1 / num_codes, 1 / num_codes)

    def forward(self, z_e):                                 # [B, D, H, W]
        B, D, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)       # [B*H*W, D]
        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ self.embedding.weight.t()
             + self.embedding.weight.pow(2).sum(1))
        idx = d.argmin(1)                                   # nearest code
        z_q = self.embedding(idx).view(B, H, W, D).permute(0, 3, 1, 2)
        loss = F.mse_loss(z_q.detach(), z_e) + self.beta * F.mse_loss(z_q, z_e.detach())
        z_q = z_e + (z_q - z_e).detach()                    # straight-through
        return z_q, loss, idx.view(B, H, W)


class VQVAE(nn.Module):
    def __init__(self, dim=32, num_codes=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),    # 28 -> 14
            nn.Conv2d(32, dim, 4, stride=2, padding=1), nn.ReLU(),  # 14 -> 7
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        self.vq = VectorQuantizer(num_codes, dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, 32, 4, stride=2, padding=1), nn.ReLU(),  # 7 -> 14
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Tanh(),    # 14 -> 28
        )
        self.dim, self.num_codes = dim, num_codes

    def forward(self, x):
        z_q, vq_loss, idx = self.vq(self.encoder(x))
        return self.decoder(z_q), vq_loss, idx

    def decode_codes(self, idx):                            # idx: [B, 7, 7] long
        z_q = self.vq.embedding(idx).permute(0, 3, 1, 2)
        return self.decoder(z_q)


def main():
    args = mc.build_argparser("VQ-VAE").parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader, _ = mc.get_dataloaders(
        name=args.dataset, batch_size=args.batch_size, limit=args.limit)

    model = VQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Device: {device} | trainable params: {sum(q.numel() for q in model.parameters()):,}")
    print("-" * 64)

    code_counts = None                                      # per-position code histogram (weak prior)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = total = 0.0
        code_counts = torch.zeros(7, 7, model.num_codes)
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            recon, vq_loss, idx = model(imgs)
            loss = F.mse_loss(recon, imgs) + vq_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item() * imgs.size(0); total += imgs.size(0)
            for h in range(7):
                for w in range(7):
                    code_counts[h, w] += torch.bincount(idx[:, h, w].cpu(), minlength=model.num_codes)
        print(f"Epoch {epoch:2d}/{args.epochs} | loss {running / total:.5f}")
    print("-" * 64)

    # Sample novel images from the per-position code marginal (weak independent prior)
    prior = (code_counts + 1e-6)
    prior = (prior / prior.sum(-1, keepdim=True)).to(device)     # [7,7,K]
    n = 1024
    idx = torch.stack([torch.multinomial(prior[h, w], n, replacement=True)
                       for h in range(7) for w in range(7)], dim=1).view(n, 7, 7).to(device)
    with torch.no_grad():
        gen = model.decode_codes(idx)
    fid = mc.compute_fid(mc.get_real_images(test_loader, n), gen, train_loader, device)
    print(f"FID (prior samples): {fid:.2f}")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        mc.save_grid_png(gen[:64], os.path.join(save_dir, "vqvae_generated_samples.png"))
        # Reconstructions: top rows real, bottom rows reconstructed
        real = mc.get_real_images(test_loader, 32).to(device)
        with torch.no_grad():
            recon, _, _ = model(real)
        pair = torch.cat([real[:32], recon[:32]], dim=0)
        mc.save_grid_png(pair, os.path.join(save_dir, "vqvae_reconstructions.png"), nrows=8, ncols=8)


if __name__ == "__main__":
    main()
