"""
08. Normalizing Flow — RealNVP (Dinh et al., 2017)
==================================================

The flow paradigm: learn an *invertible* mapping f between data x and a simple
base distribution z ~ N(0, I). Because f is invertible with a tractable Jacobian,
the exact data likelihood is available via the change-of-variables formula:

    log p(x) = log N(f(x); 0, I) + log |det df/dx|

RealNVP builds f from **affine coupling layers**: split the dimensions in two; pass
one half through unchanged and use it to predict a scale & shift for the other
half. The Jacobian is triangular, so its log-det is just the sum of the scales —
cheap to compute, and trivially invertible for sampling.

Architecture Diagram / Layout:
    x (784) -> [Coupling layer x K, alternating masks] -> z (784)
       each coupling: y = mask*x + (1-mask)*(x * exp(s(mask*x)) + t(mask*x))
    Sample: z ~ N(0,I) -> run the couplings in reverse -> image.

Key insights / educational takeaways:
    * The only paradigm here giving BOTH exact likelihood and exact latent inference.
    * Invertibility constrains the architecture (equal-size in/out, easy Jacobian).
    * Trained by plain maximum likelihood (no adversary, no reconstruction term).

Run:
    python "08.normalizing-flow.py" --epochs 5
    python "08.normalizing-flow.py" --limit 2000 --epochs 2
"""

import math
import os
import torch
import torch.nn as nn
import gen_common as mc

DIM = 28 * 28


class CouplingLayer(nn.Module):
    def __init__(self, dim, mask, hidden=256):
        super().__init__()
        self.register_buffer("mask", mask)
        self.scale = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.trans = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, x):                                   # data -> latent, returns logdet
        xm = x * self.mask
        s = torch.tanh(self.scale(xm)) * (1 - self.mask)   # tanh keeps scales stable
        t = self.trans(xm) * (1 - self.mask)
        y = xm + (1 - self.mask) * (x * torch.exp(s) + t)
        return y, s.sum(dim=1)

    def inverse(self, y):                                   # latent -> data
        ym = y * self.mask
        s = torch.tanh(self.scale(ym)) * (1 - self.mask)
        t = self.trans(ym) * (1 - self.mask)
        return ym + (1 - self.mask) * ((y - t) * torch.exp(-s))


class RealNVP(nn.Module):
    def __init__(self, dim=DIM, n_layers=8, hidden=256):
        super().__init__()
        masks = []
        base = torch.arange(dim) % 2                       # checkerboard on the flattened vector
        for i in range(n_layers):
            masks.append((base if i % 2 == 0 else 1 - base).float())
        self.layers = nn.ModuleList([CouplingLayer(dim, m, hidden) for m in masks])

    def log_prob(self, x):
        z, logdet = x, 0.0
        for layer in self.layers:
            z, ld = layer(z)
            logdet = logdet + ld
        log_pz = (-0.5 * (z ** 2) - 0.5 * math.log(2 * math.pi)).sum(dim=1)
        return log_pz + logdet

    @torch.no_grad()
    def sample(self, n, device):
        z = torch.randn(n, DIM, device=device)
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z


def main():
    args = mc.build_argparser("RealNVP Normalizing Flow").parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader, _ = mc.get_dataloaders(
        name=args.dataset, batch_size=args.batch_size, limit=args.limit)

    model = RealNVP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Device: {device} | trainable params: {sum(q.numel() for q in model.parameters()):,}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = total = 0.0
        for imgs, _ in train_loader:
            x = imgs.to(device).view(imgs.size(0), -1)
            x = x + 0.02 * torch.randn_like(x)             # dequantize (avoid infinite density)
            loss = -model.log_prob(x).mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.item() * x.size(0); total += x.size(0)
        print(f"Epoch {epoch:2d}/{args.epochs} | neg_log_likelihood {running / total:.2f}")
    print("-" * 64)

    gen = model.sample(1024, device).view(-1, 1, 28, 28).clamp(-1, 1)
    fid = mc.compute_fid(mc.get_real_images(test_loader, len(gen)), gen, train_loader, device)
    print(f"FID: {fid:.2f}")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        mc.save_grid_png(gen[:64], os.path.join(save_dir, "flow_generated_samples.png"))


if __name__ == "__main__":
    main()
