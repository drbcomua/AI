"""
06. SegFormer (Xie et al., 2021)
================================

A simple, lightweight, and highly efficient transformer-based semantic segmentation network.

Architecture Diagram / Layout:
    Input [3 x 128 x 128] 
         |  Stage 1: Overlap Patch Embed (7x7, stride 4) + Transformer Block (SR=8, heads=1)
         v
    Stage 1 Features [16 x 32 x 32] (sequence: [B, 1024, 16])
         |  Stage 2: Overlap Patch Merge (3x3, stride 2) + Transformer Block (SR=4, heads=2)
         v
    Stage 2 Features [32 x 16 x 16] (sequence: [B, 256, 32])
         |  Stage 3: Overlap Patch Merge (3x3, stride 2) + Transformer Block (SR=2, heads=4)
         v
    Stage 3 Features [64 x 8 x 8]   (sequence: [B, 64, 64])
         |  Stage 4: Overlap Patch Merge (3x3, stride 2) + Transformer Block (SR=1, heads=8)
         v
    Stage 4 Features [128 x 4 x 4]  (sequence: [B, 16, 128])
         |
         v
    Decoder (All-MLP):
         - Project Stage 1-4 features to common embedding dimension (64 channels)
         - Bilinear upsample Stage 2-4 features to 32x32 spatial resolution
         - Concatenate stage features -> [256 x 32 x 32]
         - Fuse linear layer -> [64 x 32 x 32]
         - Classifier linear layer -> [1 x 32 x 32]
         - Upsample 4x -> Output Mask [1 x 128 x 128]

Key insights / educational takeaways:
    * Hierarchical Transformer Encoder (MiT): Generates multi-resolution feature maps resembling conventional CNNs, capturing both high-resolution local details and low-resolution global context.
    * Positional Encoding Free: Implicitly incorporates positional information via 3x3 depth-wise convolutions in the feed-forward network (MixFFN), allowing the model to adapt seamlessly to varying input dimensions at inference time.
    * Efficient Attention: Reduces the sequence length of keys and values using a spatial reduction (SR) ratio, maintaining computational tractability on larger feature maps.
    * Lightweight Decoder: Eliminates the need for complex, heavy decoders by projecting and combining multi-resolution features using simple MLP layers.
    * Downscaling Design: Re-scales Stage 1-4 channel dimensions (`[16, 32, 64, 128]`) and SR ratios (`[8, 4, 2, 1]`) to suit the $128 \times 128$ small-scale dataset and keep parameters lightweight.

Run:
    python "06.segformer.py" --epochs 10
    python "06.segformer.py" --limit 100 --epochs 1        # fast smoke test
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import segmentation_common as mc


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding (or patch merging for downstream stages)."""
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2).contiguous()  # -> shape [B, H*W, C]
        x = self.norm(x)
        return x, h, w


class MixFFN(nn.Module):
    """Feed-forward network incorporating implicit positional information via depth-wise conv."""
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1,
                                groups=hidden_features, bias=False)
        self.fc2 = nn.Conv2d(hidden_features, in_features, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x, h, w):
        B, N, C = x.shape
        # Restore sequence to 2D grid shape for convolution
        x = x.transpose(1, 2).contiguous().reshape(B, C, h, w)
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        # Flatten back to sequence shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class EfficientAttention(nn.Module):
    """Multi-head self-attention with sequence length reduction for keys and values."""
    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, bias=False)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, h, w):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            x_grid = x.transpose(1, 2).contiguous().reshape(B, C, h, w)
            x_reduced = self.sr(x_grid).flatten(2).transpose(1, 2).contiguous()
            x_reduced = self.norm(x_reduced)
            kv = self.kv(x_reduced).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()

        k, v = kv[0].contiguous(), kv[1].contiguous()  # shape: [B, num_heads, SeqLen_kv, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Mix Transformer block applying Self-Attention and MixFFN with LayerNorm and residuals."""
    def __init__(self, dim, num_heads, mlp_ratio=4, sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio))

    def forward(self, x, h, w):
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x), h, w)
        return x


class SegFormerDecoder(nn.Module):
    """Lightweight all-MLP decoder for SegFormer."""
    def __init__(self, dims: list[int], dec_dim: int = 64, out_channels: int = 1):
        super().__init__()
        self.linear1 = nn.Conv2d(dims[0], dec_dim, kernel_size=1)
        self.linear2 = nn.Conv2d(dims[1], dec_dim, kernel_size=1)
        self.linear3 = nn.Conv2d(dims[2], dec_dim, kernel_size=1)
        self.linear4 = nn.Conv2d(dims[3], dec_dim, kernel_size=1)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(dec_dim * 4, dec_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dec_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(dec_dim, out_channels, kernel_size=1)

    def forward(self, c1, c2, c3, c4):
        h, w = c1.size(2), c1.size(3)

        # Project multi-resolution features to same channel count
        _c1 = self.linear1(c1)
        _c2 = F.interpolate(self.linear2(c2), size=(h, w), mode="bilinear", align_corners=True)
        _c3 = F.interpolate(self.linear3(c3), size=(h, w), mode="bilinear", align_corners=True)
        _c4 = F.interpolate(self.linear4(c4), size=(h, w), mode="bilinear", align_corners=True)

        # Concatenate and fuse
        fused = torch.cat([_c1, _c2, _c3, _c4], dim=1)
        out = self.linear_fuse(fused)
        return self.classifier(out)


class SegFormer(nn.Module):
    """SegFormer model linking Hierarchical Transformer Encoder and Lightweight Decoder."""
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        dims = [16, 32, 64, 128]
        
        # Overlap Patch Embeddings/Merges
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_channels, embed_dim=dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=dims[0], embed_dim=dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=dims[1], embed_dim=dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=dims[2], embed_dim=dims[3])

        # Mix Transformer Blocks
        self.block1 = TransformerBlock(dim=dims[0], num_heads=1, mlp_ratio=4, sr_ratio=8)
        self.block2 = TransformerBlock(dim=dims[1], num_heads=2, mlp_ratio=4, sr_ratio=4)
        self.block3 = TransformerBlock(dim=dims[2], num_heads=4, mlp_ratio=4, sr_ratio=2)
        self.block4 = TransformerBlock(dim=dims[3], num_heads=8, mlp_ratio=4, sr_ratio=1)

        # Decoder
        self.decoder = SegFormerDecoder(dims=dims, dec_dim=64, out_channels=out_channels)

    def forward(self, x):
        h_in, w_in = x.size(2), x.size(3)

        # Stage 1
        x, h1, w1 = self.patch_embed1(x)
        x = self.block1(x, h1, w1)
        c1 = x.transpose(1, 2).contiguous().reshape(x.size(0), x.size(2), h1, w1)

        # Stage 2
        x, h2, w2 = self.patch_embed2(c1)
        x = self.block2(x, h2, w2)
        c2 = x.transpose(1, 2).contiguous().reshape(x.size(0), x.size(2), h2, w2)

        # Stage 3
        x, h3, w3 = self.patch_embed3(c2)
        x = self.block3(x, h3, w3)
        c3 = x.transpose(1, 2).contiguous().reshape(x.size(0), x.size(2), h3, w3)

        # Stage 4
        x, h4, w4 = self.patch_embed4(c3)
        x = self.block4(x, h4, w4)
        c4 = x.transpose(1, 2).contiguous().reshape(x.size(0), x.size(2), h4, w4)

        # Decoder fusion & classification
        out = self.decoder(c1, c2, c3, c4)
        
        # Bilinear upsample output by 4x back to input size
        return F.interpolate(out, size=(h_in, w_in), mode="bilinear", align_corners=True)


def main():
    p = mc.build_argparser("SegFormer Semantic Segmentation", epochs=10)
    args = p.parse_args()

    device = mc.get_device(args.device)

    # Generate synthetic image-mask dataset
    images, masks = mc.generate_segmentation_dataset(num_samples=args.limit or 1000)

    # Split (80/20)
    split_idx = int(len(images) * 0.8)
    train_img, train_mask = images[:split_idx], masks[:split_idx]
    test_img, test_mask = images[split_idx:], masks[split_idx:]

    model = SegFormer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Binary segmentation: Soft Dice Loss combined with BCE Loss
    bce_criterion = nn.BCEWithLogitsLoss()
    dice_criterion = mc.SoftDiceLoss()

    print("Training SegFormer...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    train_dataset = TensorDataset(train_img, train_mask)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_iou = 0.0
        total = 0

        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)

            optimizer.zero_grad()
            logits = model(img)

            # Combined Loss
            loss_bce = bce_criterion(logits, mask)
            loss_dice = dice_criterion(logits, mask)
            loss = 0.5 * loss_bce + 0.5 * loss_dice

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * img.size(0)
            epoch_iou += mc.compute_iou(logits, mask) * img.size(0)
            total += img.size(0)

        train_loss_avg = epoch_loss / total
        train_iou_avg = epoch_iou / total
        print(f"Epoch {epoch:2d}/{args.epochs} | loss: {train_loss_avg:.4f} | mIoU: {train_iou_avg * 100:.2f}%")

    print("-" * 64)

    # Evaluation on unseen test set
    test_dataset = TensorDataset(test_img, test_mask)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    test_loss = 0.0
    test_iou = 0.0
    total = 0

    with torch.no_grad():
        for img, mask in test_loader:
            img, mask = img.to(device), mask.to(device)
            logits = model(img)
            loss = 0.5 * bce_criterion(logits, mask) + 0.5 * dice_criterion(logits, mask)

            test_loss += loss.item() * img.size(0)
            test_iou += mc.compute_iou(logits, mask) * img.size(0)
            total += img.size(0)

    test_loss_avg = test_loss / total
    test_iou_avg = test_iou / total

    print(f"Test Loss: {test_loss_avg:.4f}")
    print(f"Test Mean IoU: {test_iou_avg * 100:.2f}%")

    # Plot comparative predictions
    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        grid_path = os.path.join(save_dir, "segformer_segmentation_results.png")

        # Capture logits for the first 3 test samples
        val_img = test_img[:3].to(device)
        with torch.no_grad():
            val_logits = model(val_img)

        mc.plot_segmentation_results(test_img[:3], test_mask[:3], val_logits.cpu(), grid_path)


if __name__ == "__main__":
    main()
