"""
20. MLP-Mixer  (Tolstikhin et al., 2021 — "An all-MLP Architecture for Vision")
===============================================================================

MLP-Mixer asks: if ViT already replaced convolutions with attention, do we even
need attention? Mixer says no — pure MLPs suffice. It splits the image into
patch tokens (like ViT) and then alternates two kinds of MLP:

  * **Token-mixing MLP** — applied across the *patch* dimension (transpose the
    table first). This is where spatial information gets mixed between locations,
    the job attention does in ViT.
  * **Channel-mixing MLP** — applied across the *feature* dimension, mixing
    information within each token (a standard per-token MLP).

Each is wrapped in LayerNorm + a residual connection. No convolutions, no
attention — just matrix multiplies, transposes, and GELU.

    --variant s   dim 64, 4 blocks      --variant b   dim 128, 8 blocks

Run:
    python "20.mlp-mixer.py" --variant s --epochs 10
    python "20.mlp-mixer.py" --variant b --limit 4000
"""

import os

import torch.nn as nn

import mnist_common as mc

VARIANTS = {"s": (64, 4), "b": (128, 8)}


class MlpBlock(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, dim, n_tokens, token_hidden, channel_hidden):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = MlpBlock(n_tokens, token_hidden)     # mixes across patches
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = MlpBlock(dim, channel_hidden)      # mixes across channels

    def forward(self, x):                                     # x: (B, tokens, dim)
        y = self.norm1(x).transpose(1, 2)                    # -> (B, dim, tokens)
        x = x + self.token_mlp(y).transpose(1, 2)            # token mixing + residual
        x = x + self.channel_mlp(self.norm2(x))              # channel mixing + residual
        return x


class MLPMixer(nn.Module):
    def __init__(self, dim, depth, img_size=28, patch=7, num_classes=10):
        super().__init__()
        n_tokens = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.blocks = nn.Sequential(*[MixerBlock(dim, n_tokens, dim * 2, dim * 4)
                                      for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)   # (B, tokens, dim)
        x = self.norm(self.blocks(x))
        return self.head(x.mean(dim=1))                      # global average over tokens


def main():
    p = mc.build_argparser("MLP-Mixer on MNIST", epochs=10)
    p.add_argument("--variant", choices=list(VARIANTS), default="s")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    dim, depth = VARIANTS[args.variant]
    model = MLPMixer(dim, depth)

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name=f"MLP-Mixer-{args.variant.upper()}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
