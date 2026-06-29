"""
04. Wasserstein GAN with Gradient Penalty (WGAN-GP)
===================================================

WGAN-GP replaces classification logits with a Critic score and enforces 1-Lipschitz continuity.

Architecture Diagram / Layout:
    Generator:
        Same layout as DCGAN (ConvTranspose2d blocks).
    Critic (No Batch Normalization to prevent batch sample leakage during penalty):
        Input [Batch, 1, 28, 28]
            -> Conv2d (k=4, s=2, p=1) -> LeakyReLU [Batch, 64, 14, 14]
            -> Conv2d (k=4, s=2, p=1) -> LeakyReLU [Batch, 128, 7, 7]
            -> Conv2d (k=7, s=1, p=0) -> Flat score [Batch, 1]

Key insights / educational takeaways:
    * The Wasserstein distance prevents mode collapse and offers a meaningful loss metric correlated with image quality.
    * Gradient Penalty forces the Critic's gradient norm to be close to 1 along the interpolation path between real and fake samples.

Run:
    python "04.wgan-gp.py" --epochs 5
    python "04.wgan-gp.py" --limit 2000 --epochs 2
"""

import os
import torch
import torch.nn as nn
import gen_common as mc


class WGANGenerator(nn.Module):
    """Generator same as DCGAN."""
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        return self.net(z)


class WGANCritic(nn.Module):
    """Critic without Batch Normalization (uses no norm or LayerNorm to keep GP clean)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


def main():
    p = mc.build_argparser("Wasserstein GAN with Gradient Penalty")
    args = p.parse_args()

    device = mc.get_device(args.device)

    print(f"Loading {args.dataset} dataset (limit={args.limit})...")
    train_loader, _, _ = mc.get_dataloaders(
        name=args.dataset, batch_size=args.batch_size, limit=args.limit
    )

    generator = WGANGenerator(latent_dim=args.latent_dim).to(device)
    critic = WGANCritic().to(device)
    generator.apply(weights_init)
    critic.apply(weights_init)

    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))

    lambda_gp = 10.0
    n_critic = 5 # Update Critic 5 times per Generator step

    n_params = sum(p.numel() for p in generator.parameters()) + sum(p.numel() for p in critic.parameters())
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    # Convert dataloader to infinite iterator to support n_critic steps easily
    train_iter = iter(train_loader)

    # Calculate total epochs based steps
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs

    for step in range(1, total_steps + 1):
        critic.train()

        # ---------------------
        #  Train Critic
        # ---------------------
        for _ in range(n_critic):
            try:
                real_imgs, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                real_imgs, _ = next(train_iter)

            b_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            opt_c.zero_grad()

            # Critic real score
            real_scores = critic(real_imgs)

            # Critic fake score
            z = torch.randn(b_size, args.latent_dim, device=device)
            fake_imgs = generator(z)
            fake_scores = critic(fake_imgs.detach())

            # Gradient Penalty (GP) calculation
            alpha = torch.rand(b_size, 1, 1, 1, device=device)
            interpolates = alpha * real_imgs + (1.0 - alpha) * fake_imgs.detach()
            interpolates.requires_grad_(True)

            d_interpolates = critic(interpolates)
            grad_outputs = torch.ones_like(d_interpolates)

            gradients = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            gradients = gradients.view(b_size, -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()

            # WGAN-GP Loss
            loss_c = fake_scores.mean() - real_scores.mean() + lambda_gp * gradient_penalty
            loss_c.backward()
            opt_c.step()

        # ---------------------
        #  Train Generator
        # ---------------------
        generator.train()
        opt_g.zero_grad()

        # Generate fake images
        z = torch.randn(b_size, args.latent_dim, device=device)
        fake_imgs = generator(z)
        fake_scores_g = critic(fake_imgs)

        # Generator wants to maximize Critic's score (minimize -score)
        loss_g = -fake_scores_g.mean()
        loss_g.backward()
        opt_g.step()

        # Log epoch-wise checkpoints
        if step % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            print(f"Epoch {epoch:2d}/{args.epochs} | loss_C {loss_c.item():.4f} | loss_G {loss_g.item():.4f} | gp {gradient_penalty.item():.4f}")

    print("-" * 64)

    # Save visual outputs
    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))

        generator.eval()
        with torch.no_grad():
            z_random = torch.randn(64, args.latent_dim, device=device)
            gen_imgs = generator(z_random)

            gen_path = os.path.join(save_dir, "wgan_generated_samples.png")
            mc.save_grid_png(gen_imgs, gen_path, nrows=8, ncols=8)

        # Save latent space walk
        walk_path = os.path.join(save_dir, "wgan_latent_walk.png")
        mc.save_latent_walk_png(generator, walk_path, args.latent_dim, device=device)


if __name__ == "__main__":
    main()
