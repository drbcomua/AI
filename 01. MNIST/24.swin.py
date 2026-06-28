"""
24. Swin Transformer  (Liu et al., 2021 — "Hierarchical Vision Transformer using Shifted Windows")
=================================================================================================

ViT runs global self-attention over every patch (quadratic cost) and keeps one
resolution throughout. Swin fixes both, making transformers behave like a
convnet backbone:

  * **Windowed attention (W-MSA).** Attention is computed only *within* small,
    non-overlapping windows -> linear cost in image size.
  * **Shifted windows (SW-MSA).** Every other block shifts the window grid by
    half a window so information crosses window boundaries; an attention mask
    keeps the rolled-over edges from attending to each other.
  * **Hierarchy via patch merging.** Between stages, 2x2 neighbouring tokens are
    concatenated and projected, halving resolution and doubling channels — the
    transformer analogue of pooling, giving a feature pyramid.

Also faithful here: relative position bias inside each window.

    --variant t (tiny: depths 2,2)      --variant s (small: depths 2,6)

MNIST-scaled: patch size 2 -> 14x14 tokens, window 7, two stages.

Run:
    python "24.swin.py" --variant t --epochs 10
    python "24.swin.py" --variant s --limit 4000
"""

import os

import torch
import torch.nn as nn

import mnist_common as mc

VARIANTS = {"t": (2, 2), "s": (2, 6)}


def window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(windows, ws, H, W):
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, ws, num_heads):
        super().__init__()
        self.ws, self.num_heads = ws, num_heads
        self.scale = (dim // num_heads) ** -0.5
        # relative position bias table + index (one bias per head per relative offset)
        self.bias_table = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, num_heads))
        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij"))
        coords = coords.flatten(1)                                   # 2, ws*ws
        rel = coords[:, :, None] - coords[:, None, :]               # 2, N, N
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer("rel_index", rel.sum(-1))             # N, N
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)             # B_, heads, N, N
        bias = self.bias_table[self.rel_index.view(-1)].view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, ws=7, shift=0, mlp_ratio=4.0):
        super().__init__()
        self.input_resolution = input_resolution
        # If the window covers the whole feature map, plain windowed attention suffices.
        if min(input_resolution) <= ws:
            ws, shift = min(input_resolution), 0
        self.ws, self.shift = ws, shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, ws, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

        attn_mask = None
        if shift > 0:
            H, W = input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            spans = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
            cnt = 0
            for h in spans:
                for w in spans:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mw = window_partition(img_mask, ws).view(-1, ws * ws)
            attn_mask = mw.unsqueeze(1) - mw.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        windows = window_partition(x, self.ws).view(-1, self.ws * self.ws, C)
        windows = self.attn(windows, self.attn_mask)
        x = window_reverse(windows.view(-1, self.ws, self.ws, C), self.ws, H, W)
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        x = shortcut + x.view(B, H * W, C)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    """Concatenate 2x2 neighbouring tokens -> halve resolution, double channels."""

    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]; x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]; x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class SwinStage(nn.Module):
    def __init__(self, dim, resolution, depth, num_heads, ws, downsample):
        super().__init__()
        self.blocks = nn.Sequential(*[
            SwinBlock(dim, resolution, num_heads, ws, shift=0 if i % 2 == 0 else ws // 2)
            for i in range(depth)
        ])
        self.downsample = PatchMerging(resolution, dim) if downsample else None

    def forward(self, x):
        x = self.blocks(x)
        return self.downsample(x) if self.downsample is not None else x


class SwinTransformer(nn.Module):
    def __init__(self, depths, img_size=28, patch=2, dim=48, ws=7,
                 heads=(3, 6), num_classes=10):
        super().__init__()
        res = img_size // patch                                   # 14
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.pe_norm = nn.LayerNorm(dim)
        self.stage1 = SwinStage(dim, (res, res), depths[0], heads[0], ws, downsample=True)
        self.stage2 = SwinStage(dim * 2, (res // 2, res // 2), depths[1], heads[1], ws,
                                downsample=False)
        self.norm = nn.LayerNorm(dim * 2)
        self.head = nn.Linear(dim * 2, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)        # (B, res*res, dim)
        x = self.pe_norm(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.norm(x)
        return self.head(x.mean(dim=1))                           # global average over tokens


def main():
    p = mc.build_argparser("Swin Transformer on MNIST", epochs=10)
    p.add_argument("--variant", choices=list(VARIANTS), default="t")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = SwinTransformer(VARIANTS[args.variant])

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name=f"Swin-{args.variant.upper()}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
