"""
26. ConvMixer  (Trockman & Kolter, 2022 — "Patches Are All You Need?")
======================================================================

ConvMixer is a deliberately minimal answer to ViT/MLP-Mixer: keep the *patch
embedding* idea, but do all the mixing with plain convolutions. The entire model
is a patch-embed stem followed by repeated blocks of

    depthwise conv (mixes *spatial* info, with a large kernel and a residual)
    pointwise 1x1 conv (mixes *channel* info)

each with GELU + BatchNorm. That's it — no attention, no token-mixing MLP, no
downsampling. Despite its simplicity it is competitive with ViT, which was the
paper's provocative point: maybe patches, not attention, are what matter.

    --variant s   dim 128, depth 8
    --variant b   dim 256, depth 8

Run:
    python "26.convmixer.py" --variant s --epochs 5
    python "26.convmixer.py" --variant b --limit 2000
"""

import os

import torch.nn as nn

import mnist_common as mc

VARIANTS = {"s": (128, 8), "b": (256, 8)}


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel=5, patch=4, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(                        # patch embedding: 28 -> 7x7 tokens
            nn.Conv2d(1, dim, kernel_size=patch, stride=patch),
            nn.GELU(), nn.BatchNorm2d(dim),
        )
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                Residual(nn.Sequential(                   # depthwise: spatial mixing + residual
                    nn.Conv2d(dim, dim, kernel, groups=dim, padding=kernel // 2),
                    nn.GELU(), nn.BatchNorm2d(dim),
                )),
                nn.Conv2d(dim, dim, 1),                    # pointwise: channel mixing
                nn.GELU(), nn.BatchNorm2d(dim),
            ) for _ in range(depth)
        ])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(dim, num_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def main():
    p = mc.build_argparser("ConvMixer on MNIST")
    p.add_argument("--variant", choices=list(VARIANTS), default="s")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    dim, depth = VARIANTS[args.variant]
    model = ConvMixer(dim, depth)

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name=f"ConvMixer-{args.variant.upper()}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
