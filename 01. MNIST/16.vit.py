"""
16. Vision Transformer (ViT)  (Dosovitskiy et al., 2021 — "An Image is Worth 16x16 Words")
=========================================================================================

ViT drops convolutions entirely. It cuts the image into a grid of fixed-size
patches, linearly embeds each patch as a "token" (exactly like a word embedding),
prepends a learnable [class] token, adds positional embeddings, and feeds the
sequence through a standard Transformer encoder. The class token's final state is
classified.

This is a small but faithful ViT for 28x28 digits:

    patch size 7  ->  4x4 = 16 patch tokens (+1 class token)
    embedding dim 64, 6 transformer blocks, 4 attention heads, MLP ratio 2
    pre-norm blocks, GELU, learnable positional embedding.

On MNIST a from-scratch ViT trains fine and reaches ~98-99%; ViTs normally crave
huge datasets, so the gap to convnets that bake in locality is part of the lesson.

Run:
    python "16.vit.py" --epochs 10
    python "16.vit.py" --limit 4000
"""

import os

import torch
import torch.nn as nn

import mnist_common as mc


class PatchEmbed(nn.Module):
    """Split into patches and linearly project — a strided conv does both at once."""

    def __init__(self, img_size=28, patch=7, in_ch=1, dim=64):
        super().__init__()
        self.n_patches = (img_size // patch) ** 2
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)                      # (B, dim, H/p, W/p)
        return x.flatten(2).transpose(1, 2)   # (B, n_patches, dim)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]     # pre-norm self-attention
        x = x + self.mlp(self.norm2(x))                       # pre-norm MLP
        return x


class ViT(nn.Module):
    def __init__(self, img_size=28, patch=7, dim=64, depth=6, heads=4,
                 mlp_ratio=2.0, num_classes=10, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch, 1, dim)
        n = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads, mlp_ratio, dropout)
                                      for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])             # classify the [class] token


def main():
    args = mc.build_argparser("Vision Transformer on MNIST", epochs=10).parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = ViT()

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name="ViT",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
