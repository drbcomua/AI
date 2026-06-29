"""
10. DDIM — Denoising Diffusion Implicit Models (Song et al., 2021)
=================================================================

Same trained noise-prediction network as DDPM (06), but a smarter *sampler*. DDPM
must walk back through all T noising steps stochastically; DDIM defines a
non-Markovian reverse process that is **deterministic** (eta=0) and can skip
steps, producing comparable images in 10-50 steps instead of hundreds.

Sampling rule (eta = 0), for a chosen descending subsequence of timesteps:
    x0_hat = (x_t - sqrt(1 - abar_t) * eps_theta(x_t, t)) / sqrt(abar_t)
    x_{t_prev} = sqrt(abar_{t_prev}) * x0_hat + sqrt(1 - abar_{t_prev}) * eps_theta

Key insights / educational takeaways:
    * Decouples the number of *sampling* steps from the number of *training* steps.
    * Determinism makes the latent->image map reproducible (great for interpolation).
    * Far fewer network evaluations than DDPM at similar quality — the practical
      speed-up that made diffusion usable.

Run:
    python "10.ddim.py" --epochs 5 --ddim-steps 25
    python "10.ddim.py" --limit 2000 --epochs 2 --ddim-steps 20
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gen_common as mc


class TimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device) * (torch.log(torch.tensor(10000.0)) / (half - 1)))
        emb = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return self.mlp(torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1))


class Block(nn.Module):
    def __init__(self, in_c, out_c, emb=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.time = nn.Linear(emb, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)

    def forward(self, x, t_emb):
        h = F.relu(self.conv1(x)) + self.time(t_emb).unsqueeze(-1).unsqueeze(-1)
        return F.relu(self.conv2(h))


class MiniUNet(nn.Module):
    def __init__(self, emb=128):
        super().__init__()
        self.t_emb = TimeEmbedding(emb)
        self.down1, self.down2 = Block(1, 32, emb), Block(32, 64, emb)
        self.mid = Block(64, 64, emb)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.upb1 = Block(96, 32, emb)
        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.upb2 = Block(17, 16, emb)
        self.out = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x, t):
        te = self.t_emb(t)
        h1 = self.down1(x, te)
        h2 = self.down2(F.max_pool2d(h1, 2), te)
        m = self.mid(F.max_pool2d(h2, 2), te)
        u1 = self.upb1(torch.cat([self.up1(m), h2], dim=1), te)
        u2 = self.upb2(torch.cat([self.up2(u1), x], dim=1), te)
        return self.out(u2)


class Diffusion:
    def __init__(self, T=200, device=None):
        self.T = T
        self.device = device or mc.get_device()
        beta = torch.linspace(1e-4, 0.02, T, device=self.device)
        self.alpha_bar = torch.cumprod(1 - beta, dim=0)

    def add_noise(self, x0, t, noise):
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise

    @torch.no_grad()
    def ddim_sample(self, model, n, steps, eta=0.0):
        model.eval()
        seq = torch.linspace(self.T - 1, 0, steps, device=self.device).long()
        x = torch.randn(n, 1, 28, 28, device=self.device)
        for i in range(len(seq)):
            t = torch.full((n,), seq[i].item(), dtype=torch.long, device=self.device)
            ab_t = self.alpha_bar[seq[i]]
            eps = model(x, t)
            x0 = ((x - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)).clamp(-1, 1)  # thresholding
            if i < len(seq) - 1:
                ab_prev = self.alpha_bar[seq[i + 1]]
                x = torch.sqrt(ab_prev) * x0 + torch.sqrt(1 - ab_prev) * eps   # eta=0, deterministic
            else:
                x = x0
        return x


def main():
    p = mc.build_argparser("DDIM Diffusion (fast deterministic sampling)")
    p.add_argument("--ddim-steps", type=int, default=25, help="sampling steps (<< training T)")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader, _ = mc.get_dataloaders(
        name=args.dataset, batch_size=args.batch_size, limit=args.limit)

    T = 200
    diff = Diffusion(T=T, device=device)
    model = MiniUNet().to(device)
    lr = 2e-4 if args.lr == 1e-3 else args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Device: {device} | trainable params: {sum(q.numel() for q in model.parameters()):,}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = total = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            t = torch.randint(0, T, (imgs.size(0),), device=device)
            noise = torch.randn_like(imgs)
            pred = model(diff.add_noise(imgs, t, noise), t)
            loss = F.mse_loss(pred, noise)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item() * imgs.size(0); total += imgs.size(0)
        print(f"Epoch {epoch:2d}/{args.epochs} | loss {running / total:.6f}")
    print("-" * 64)

    print(f"DDIM sampling with {args.ddim_steps} steps (training T={T})...")
    gen = diff.ddim_sample(model, 1024, args.ddim_steps)
    fid = mc.compute_fid(mc.get_real_images(test_loader, len(gen)), gen, train_loader, device)
    print(f"FID ({args.ddim_steps} DDIM steps): {fid:.2f}")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        mc.save_grid_png(gen[:64], os.path.join(save_dir, "ddim_generated_samples.png"))


if __name__ == "__main__":
    main()
