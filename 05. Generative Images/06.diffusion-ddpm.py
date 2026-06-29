"""
06. Denoising Diffusion Probabilistic Models (DDPM)
===================================================

A Denoising Diffusion Probabilistic Model using a time-conditioned U-Net.

Architecture Diagram / Layout:
    UNet Noise Predictor:
        Input x_t [Batch, 1, 28, 28], Timestep t [Batch]
            -> Time Embedding MLP [Batch, Emb_Dim]
            -> Down Block 1: Conv2d -> Add Time Proj -> Downsample [Batch, 32, 14, 14]
            -> Down Block 2: Conv2d -> Add Time Proj -> Downsample [Batch, 64, 7, 7]
            -> Middle Block: Conv2d -> Add Time Proj [Batch, 64, 7, 7]
            -> Up Block 1: ConvTranspose2d -> Concat Down2 -> Add Time Proj [Batch, 32, 14, 14]
            -> Up Block 2: ConvTranspose2d -> Concat Down1 -> Add Time Proj [Batch, 1, 28, 28]

Key insights / educational takeaways:
    * Forward diffusion maps data to pure Gaussian noise by incrementally adding noise over T steps.
    * Reverse diffusion trains a model to predict the added noise at step t, allowing iterative synthesis from pure noise.

Run:
    python "06.diffusion-ddpm.py" --epochs 5
    python "06.diffusion-ddpm.py" --limit 2000 --epochs 2
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gen_common as mc


# --------------------------------------------------------------------------- #
# Time Conditioning Module & UNet Components
# --------------------------------------------------------------------------- #
class TimeEmbedding(nn.Module):
    """Sinusoidal positional embeddings for timesteps, followed by an MLP projection."""
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.emb_dim = emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, t):
        # t shape: [B]
        device = t.device
        half_dim = self.emb_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(-1).float() * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class UNetBlock(nn.Module):
    """Conv block with time-conditioning projection adding."""
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 128):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(emb_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t_emb):
        h = F.relu(self.conv(x))
        # Project time embedding to match out_channels spatial dimensions
        proj = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + proj # Condition features with time step representation
        return F.relu(self.conv2(h))


# --------------------------------------------------------------------------- #
# Time-Conditioned U-Net Noise Predictor
# --------------------------------------------------------------------------- #
class MiniUNet(nn.Module):
    """Lightweight U-Net with skip-connections and time-conditioning."""
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.t_emb = TimeEmbedding(emb_dim)

        self.down1 = UNetBlock(1, 32, emb_dim)
        self.down2 = UNetBlock(32, 64, emb_dim)

        self.mid = UNetBlock(64, 64, emb_dim)

        # Upsample blocks
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) # 7x7 -> 14x14
        self.up_block1 = UNetBlock(96, 32, emb_dim) # concatenated with h2 (32 + 64)

        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2) # 14x14 -> 28x28
        self.up_block2 = UNetBlock(17, 16, emb_dim) # concatenated with input (16 + 1)
        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x, t):
        # x: [B, 1, 28, 28], t: [B]
        t_emb = self.t_emb(t) # [B, emb_dim]

        # Downsample path
        h1 = self.down1(x, t_emb) # [B, 32, 28, 28]
        h1_pool = F.max_pool2d(h1, 2) # [B, 32, 14, 14]

        h2 = self.down2(h1_pool, t_emb) # [B, 64, 14, 14]
        h2_pool = F.max_pool2d(h2, 2) # [B, 64, 7, 7]

        # Bottleneck middle
        m = self.mid(h2_pool, t_emb) # [B, 64, 7, 7]

        # Upsample path
        u1 = self.up1(m) # [B, 32, 14, 14]
        # Concat skip connection
        u1_cat = torch.cat([u1, h2], dim=1) # [B, 96, 14, 14]
        u1_out = self.up_block1(u1_cat, t_emb) # [B, 32, 14, 14]

        u2 = self.up2(u1_out) # [B, 16, 28, 28]
        # Concat skip with original input image
        u2_cat = torch.cat([u2, x], dim=1) # [B, 17, 28, 28]
        h_out = self.up_block2(u2_cat, t_emb) # [B, 16, 28, 28]
        out_noise = self.out_conv(h_out) # [B, 1, 28, 28]

        return out_noise


# --------------------------------------------------------------------------- #
# Diffusion Scheduler Pipeline
# --------------------------------------------------------------------------- #
class DDPMPipeline:
    """Manages forward noise addition schedules and reverse generation sampling."""
    def __init__(self, T: int = 200, beta_start: float = 1e-4, beta_end: float = 0.02, device=None):
        self.T = T
        self.device = device or mc.get_device()

        # Variance schedules
        self.beta = torch.linspace(beta_start, beta_end, T, device=self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward_diffusion(self, x0, t, noise=None):
        """Add noise to original image x0 at step t."""
        # x0: [B, 1, 28, 28], t: [B]
        if noise is None:
            noise = torch.randn_like(x0)

        # Extract values for specific timesteps
        ab_t = self.alpha_bar[t].view(-1, 1, 1, 1)

        xt = torch.sqrt(ab_t) * x0 + torch.sqrt(1.0 - ab_t) * noise
        return xt

    @torch.no_grad()
    def sample(self, model, n_samples: int = 64):
        """Generate images starting from pure Gaussian noise."""
        model.eval()
        # Start from pure noise
        x = torch.randn(n_samples, 1, 28, 28, device=self.device)

        for step in reversed(range(self.T)):
            t = torch.full((n_samples,), step, dtype=torch.long, device=self.device)
            predicted_noise = model(x, t)

            beta_t = self.beta[step]
            alpha_t = self.alpha[step]
            ab_t = self.alpha_bar[step]

            # Mean formula
            coef = beta_t / torch.sqrt(1.0 - ab_t)
            mean = (1.0 / torch.sqrt(alpha_t)) * (x - coef * predicted_noise)

            if step > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = mean + sigma_t * noise
            else:
                x = mean

        return x

    @torch.no_grad()
    def sample_denoising_walk(self, model):
        """Generate a series showing progressive denoisings (e.g. 10 steps)."""
        model.eval()
        x = torch.randn(1, 1, 28, 28, device=self.device)
        walk_steps = []

        interval = self.T // 10
        for step in reversed(range(self.T)):
            t = torch.full((1,), step, dtype=torch.long, device=self.device)
            predicted_noise = model(x, t)

            beta_t = self.beta[step]
            alpha_t = self.alpha[step]
            ab_t = self.alpha_bar[step]

            coef = beta_t / torch.sqrt(1.0 - ab_t)
            mean = (1.0 / torch.sqrt(alpha_t)) * (x - coef * predicted_noise)

            if step > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = mean + sigma_t * noise
            else:
                x = mean

            if step % interval == 0 or step == 0:
                walk_steps.append(x.clone())

        # Return concatenated steps
        return torch.cat(walk_steps[:10], dim=0)


def main():
    p = mc.build_argparser("Denoising Diffusion Model")
    args = p.parse_args()

    device = mc.get_device(args.device)

    print(f"Loading {args.dataset} dataset (limit={args.limit})...")
    train_loader, _, _ = mc.get_dataloaders(
        name=args.dataset, batch_size=args.batch_size, limit=args.limit
    )

    # Use T = 200 for fast educational training/sampling speeds
    T = 200
    pipeline = DDPMPipeline(T=T, device=device)
    model = MiniUNet().to(device)

    lr = 2e-4 if args.lr == 1e-3 else args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for real_imgs, _ in train_loader:
            b_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # Sample random step t for each batch item
            t = torch.randint(0, T, (b_size,), device=device)
            noise = torch.randn_like(real_imgs)

            # Forward diffusion: corrupt image with noise
            xt = pipeline.forward_diffusion(real_imgs, t, noise)

            optimizer.zero_grad()
            # Predict the noise
            pred_noise = model(xt, t)
            loss = criterion(pred_noise, noise)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * b_size
            total_samples += b_size

        train_loss = running_loss / total_samples
        print(f"Epoch {epoch:2d}/{args.epochs} | loss {train_loss:.6f}")

    print("-" * 64)

    # Save visual outputs
    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))

        # 1. Generate final generated grid from random noise
        print("Sampling 64 images from trained DDPM...")
        gen_imgs = pipeline.sample(model, n_samples=64)
        gen_path = os.path.join(save_dir, "diffusion_generated_samples.png")
        mc.save_grid_png(gen_imgs, gen_path, nrows=8, ncols=8)

        # 2. Save a progressive denoising walk (10 steps)
        print("Sampling progressive denoising walk...")
        denoise_walk = pipeline.sample_denoising_walk(model)
        walk_path = os.path.join(save_dir, "diffusion_denoising_walk.png")

        # Reuse latent walk saver layout
        # Or simple plot 1x10 row grid manually
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            walk_imgs = (denoise_walk.detach().cpu().numpy() + 1.0) / 2.0
            walk_imgs = np.clip(walk_imgs, 0.0, 1.0)

            fig, axes = plt.subplots(1, 10, figsize=(12, 1.5))
            for i in range(10):
                axes[i].imshow(walk_imgs[i, 0], cmap="gray")
                axes[i].axis("off")
                axes[i].set_title(f"t={T - i * (T//10)}", fontsize=8)

            fig.tight_layout(pad=0.2)
            fig.savefig(walk_path, dpi=120)
            plt.close(fig)
            print(f"Saved denoising progress walk -> {walk_path}")
        except Exception as e:
            print(f"(failed to save denoising progress walk: {e})")


if __name__ == "__main__":
    main()
