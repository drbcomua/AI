"""
27. PoolFormer / MetaFormer  (Yu et al., 2022 — "MetaFormer Is Actually What You Need")
=======================================================================================

PoolFormer makes a pointed argument: the success of Transformers may owe less to
*attention* than to the general "MetaFormer" template — token-mixer + channel-MLP,
each with a residual and normalization. To prove it, they replace attention with
the cheapest token mixer imaginable: **average pooling**.

    token mixer  =  AvgPool3x3(x) - x      (a parameter-free local mixer)
    channel MLP  =  1x1 conv -> GELU -> 1x1 conv

Each is wrapped in normalization, a residual, and a learnable LayerScale, and the
network is hierarchical (patch embeddings downsample between stages) like a
convnet. That a pooling-only "transformer" still works well is the whole lesson —
a sharp counterpoint to ViT, Swin, and MLP-Mixer.

    --variant s   depths (2, 2)        --variant m   depths (2, 6)

Run:
    python "27.poolformer.py" --variant s --epochs 5
    python "27.poolformer.py" --variant m --limit 2000
"""

import os

import torch
import torch.nn as nn

import mnist_common as mc

VARIANTS = {"s": (2, 2), "m": (2, 6)}


class Pooling(nn.Module):
    """The token mixer: subtract the input from its local average (pool(x) - x)."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class PoolFormerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, layer_scale=1e-5):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)                 # GroupNorm(1) == LayerNorm over C,H,W
        self.token_mixer = Pooling()
        self.norm2 = nn.GroupNorm(1, dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.GELU(), nn.Conv2d(hidden, dim, 1))
        self.ls1 = nn.Parameter(layer_scale * torch.ones(dim, 1, 1))
        self.ls2 = nn.Parameter(layer_scale * torch.ones(dim, 1, 1))

    def forward(self, x):
        x = x + self.ls1 * self.token_mixer(self.norm1(x))
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class PoolFormer(nn.Module):
    def __init__(self, depths, dims=(64, 128), num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(1, dims[0], kernel_size=2, stride=2)         # 28 -> 14
        self.stage1 = nn.Sequential(*[PoolFormerBlock(dims[0]) for _ in range(depths[0])])
        self.down = nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)   # 14 -> 7
        self.stage2 = nn.Sequential(*[PoolFormerBlock(dims[1]) for _ in range(depths[1])])
        self.norm = nn.GroupNorm(1, dims[1])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(dims[1], num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down(x)
        x = self.stage2(x)
        return self.head(self.norm(x))


def main():
    p = mc.build_argparser("PoolFormer on MNIST")
    p.add_argument("--variant", choices=list(VARIANTS), default="s")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = PoolFormer(VARIANTS[args.variant])

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name=f"PoolFormer-{args.variant.upper()}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
