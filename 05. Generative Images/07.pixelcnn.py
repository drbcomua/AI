"""
07. PixelCNN — Autoregressive Image Model (van den Oord et al., 2016)
====================================================================

The autoregressive paradigm: model the image as a product of per-pixel
conditionals, p(x) = prod_i p(x_i | x_<i), in raster-scan order. Causality is
enforced by **masked convolutions** — each filter is zeroed out for the current
and future pixels — so a single forward pass predicts a categorical distribution
over intensities for every pixel given only those above-and-to-the-left.

Architecture Diagram / Layout:
    Input image [B, 1, 28, 28] (in [0,1])
       -> Masked Conv (type A, 7x7)   (excludes the center pixel)
       -> N x [Masked Conv (type B, 7x7) + ReLU]   (include the center)
       -> 1x1 Conv -> logits [B, levels, 28, 28]
    Generation: sample pixel-by-pixel in raster order (slow but exact).

Key insights / educational takeaways:
    * Exact, tractable likelihood (trained with plain cross-entropy over intensities).
    * Mask type 'A' on the first layer is what prevents a pixel from seeing itself.
    * Sampling is inherently sequential (one network pass per pixel) — the price of
      exact autoregressive modeling.

Run:
    python "07.pixelcnn.py" --epochs 5
    python "07.pixelcnn.py" --limit 2000 --epochs 2 --gen-samples 64
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gen_common as mc


class MaskedConv2d(nn.Conv2d):
    """Convolution whose kernel is masked so output (i,j) sees only earlier pixels."""
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.ones_like(self.weight))
        _, _, kh, kw = self.weight.shape
        self.mask[:, :, kh // 2, kw // 2 + (mask_type == "B"):] = 0   # right of center
        self.mask[:, :, kh // 2 + 1:, :] = 0                          # rows below center

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class PixelCNN(nn.Module):
    def __init__(self, levels: int = 256, channels: int = 64, n_layers: int = 6):
        super().__init__()
        self.levels = levels
        layers = [MaskedConv2d("A", 1, channels, 7, padding=3), nn.ReLU()]
        for _ in range(n_layers):
            layers += [MaskedConv2d("B", channels, channels, 7, padding=3),
                       nn.BatchNorm2d(channels), nn.ReLU()]
        layers += [nn.Conv2d(channels, levels, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):                                   # x in [0,1], [B,1,28,28]
        return self.net(x - 0.5)                            # center the input


@torch.no_grad()
def sample(model, n, device, temperature=1.0):
    model.eval()
    levels = model.levels
    img = torch.zeros(n, 1, 28, 28, device=device)
    for i in range(28):
        for j in range(28):
            logits = model(img)[:, :, i, j] / temperature   # [n, levels]
            probs = torch.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).squeeze(1)     # [n]
            img[:, 0, i, j] = idx.float() / (levels - 1)
    return img * 2 - 1                                       # [0,1] -> [-1,1]


def main():
    p = mc.build_argparser("PixelCNN Autoregressive Image Model")
    p.add_argument("--gen-samples", type=int, default=256,
                   help="images to generate (for the grid + FID)")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader, _ = mc.get_dataloaders(
        name=args.dataset, batch_size=args.batch_size, limit=args.limit)

    levels = 256
    model = PixelCNN(levels=levels).to(device)
    lr = 3e-4 if args.lr == 1e-3 else args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Device: {device} | trainable params: {sum(q.numel() for q in model.parameters()):,}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = total = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            x01 = (imgs + 1) / 2
            target = (x01 * (levels - 1)).round().long().clamp(0, levels - 1).squeeze(1)
            logits = model(x01)                              # [B, levels, 28, 28]
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item() * imgs.size(0); total += imgs.size(0)
        print(f"Epoch {epoch:2d}/{args.epochs} | nll(bits/dim≈) {running / total / 0.6931:.4f}")
    print("-" * 64)

    print(f"Sampling {args.gen_samples} images (raster-scan, 784 passes)...")
    gen = sample(model, args.gen_samples, device)
    fid = mc.compute_fid(mc.get_real_images(test_loader, len(gen)), gen, train_loader, device)
    print(f"FID: {fid:.2f}")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        mc.save_grid_png(gen[:64], os.path.join(save_dir, "pixelcnn_generated_samples.png"))


if __name__ == "__main__":
    main()
