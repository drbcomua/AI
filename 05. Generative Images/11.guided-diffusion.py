"""
11. Classifier-Free Guided Diffusion (Ho & Salimans, 2022)
==========================================================

Conditional diffusion that can be *steered* toward a class without a separate
classifier. A single noise-prediction network is trained both conditionally (given
the class label) and unconditionally (label randomly dropped to a null token). At
sampling time the two predictions are combined and extrapolated:

    eps = eps_uncond + guidance * (eps_cond - eps_uncond)

Larger `guidance` pushes samples to be more strongly class-typical (sharper, less
diverse) — the trade-off behind every modern text-to-image system. This does for
diffusion what the Conditional GAN (05) did for GANs.

Architecture Diagram / Layout:
    Noise U-Net conditioned on (timestep embedding + class embedding).
    Train: with prob p_drop, replace the label with a learned null embedding.
    Sample: run twice per step (cond + uncond), combine with the guidance scale.

Key insights / educational takeaways:
    * One network learns both p(x|class) and p(x); guidance interpolates/extrapolates.
    * No auxiliary classifier needed (unlike classifier guidance).
    * Output grid: each row is a class, showing controllable synthesis.

Run:
    python "11.guided-diffusion.py" --epochs 5 --guidance 3.0
    python "11.guided-diffusion.py" --limit 2000 --epochs 2
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
        self.cond = nn.Linear(emb, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)

    def forward(self, x, c_emb):
        h = F.relu(self.conv1(x)) + self.cond(c_emb).unsqueeze(-1).unsqueeze(-1)
        return F.relu(self.conv2(h))


class CondUNet(nn.Module):
    """Time-conditioned U-Net with an added class embedding (null index = num_classes)."""
    def __init__(self, num_classes=10, emb=128):
        super().__init__()
        self.t_emb = TimeEmbedding(emb)
        self.label_emb = nn.Embedding(num_classes + 1, emb)   # +1 = null/unconditional
        self.null_idx = num_classes
        self.down1, self.down2 = Block(1, 32, emb), Block(32, 64, emb)
        self.mid = Block(64, 64, emb)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.upb1 = Block(96, 32, emb)
        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.upb2 = Block(17, 16, emb)
        self.out = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x, t, y):
        c = self.t_emb(t) + self.label_emb(y)                 # combine time + class
        h1 = self.down1(x, c)
        h2 = self.down2(F.max_pool2d(h1, 2), c)
        m = self.mid(F.max_pool2d(h2, 2), c)
        u1 = self.upb1(torch.cat([self.up1(m), h2], dim=1), c)
        u2 = self.upb2(torch.cat([self.up2(u1), x], dim=1), c)
        return self.out(u2)


class Diffusion:
    def __init__(self, T=200, device=None):
        self.T = T
        self.device = device or mc.get_device()
        self.beta = torch.linspace(1e-4, 0.02, T, device=self.device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x0, t, noise):
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise

    @torch.no_grad()
    def sample(self, model, labels, guidance=3.0):
        model.eval()
        n = labels.size(0)
        null = torch.full((n,), model.null_idx, dtype=torch.long, device=self.device)
        x = torch.randn(n, 1, 28, 28, device=self.device)
        for step in reversed(range(self.T)):
            t = torch.full((n,), step, dtype=torch.long, device=self.device)
            eps_c = model(x, t, labels)
            eps_u = model(x, t, null)
            eps = eps_u + guidance * (eps_c - eps_u)          # classifier-free guidance
            beta_t, alpha_t, ab_t = self.beta[step], self.alpha[step], self.alpha_bar[step]
            ab_prev = self.alpha_bar[step - 1] if step > 0 else torch.tensor(1.0, device=self.device)
            # predict x0 and clamp (static thresholding) — keeps guidance from diverging
            x0 = ((x - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)).clamp(-1, 1)
            coef_x0 = torch.sqrt(ab_prev) * beta_t / (1 - ab_t)
            coef_xt = torch.sqrt(alpha_t) * (1 - ab_prev) / (1 - ab_t)
            mean = coef_x0 * x0 + coef_xt * x
            x = mean + (torch.sqrt(beta_t) * torch.randn_like(x) if step > 0 else 0)
        return x


def main():
    p = mc.build_argparser("Classifier-Free Guided Diffusion")
    p.add_argument("--guidance", type=float, default=3.0, help="guidance scale (0 = unconditional)")
    p.add_argument("--p-drop", type=float, default=0.1, help="prob. of dropping the label in training")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader, num_classes = mc.get_dataloaders(
        name=args.dataset, batch_size=args.batch_size, limit=args.limit)

    T = 200
    diff = Diffusion(T=T, device=device)
    model = CondUNet(num_classes=num_classes).to(device)
    lr = 2e-4 if args.lr == 1e-3 else args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Device: {device} | trainable params: {sum(q.numel() for q in model.parameters()):,}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = total = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # randomly drop labels to the null token for unconditional training
            drop = torch.rand(labels.size(0), device=device) < args.p_drop
            labels = labels.clone()
            labels[drop] = model.null_idx
            t = torch.randint(0, T, (imgs.size(0),), device=device)
            noise = torch.randn_like(imgs)
            pred = model(diff.add_noise(imgs, t, noise), t, labels)
            loss = F.mse_loss(pred, noise)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item() * imgs.size(0); total += imgs.size(0)
        print(f"Epoch {epoch:2d}/{args.epochs} | loss {running / total:.6f}")
    print("-" * 64)

    # Class-conditioned grid: each row = one class
    print(f"Sampling class-conditioned grid (guidance={args.guidance})...")
    rows = num_classes
    labels = torch.arange(rows, device=device).repeat_interleave(8)
    grid = diff.sample(model, labels, guidance=args.guidance)

    # FID on a class-balanced batch
    fid_labels = torch.arange(num_classes, device=device).repeat(100)[:1000]
    gen = diff.sample(model, fid_labels, guidance=args.guidance)
    fid = mc.compute_fid(mc.get_real_images(test_loader, len(gen)), gen, train_loader, device)
    print(f"FID: {fid:.2f}")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        mc.save_grid_png(grid, os.path.join(save_dir, "guided_generated_samples.png"),
                         nrows=rows, ncols=8)


if __name__ == "__main__":
    main()
