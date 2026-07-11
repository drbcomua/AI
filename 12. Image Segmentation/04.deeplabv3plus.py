"""
04. DeepLabV3+ (Chen et al., 2018)
==================================

Implements a powerful semantic segmentation network combining Atrous Spatial Pyramid Pooling (ASPP) and an encoder-decoder structure.

Architecture Diagram / Layout:
    Input [3 x 128 x 128] -> Layer 1 (Conv/BN/ReLU) -> [16 x 128 x 128]
                          -> MaxPool2d (2x) -> [16 x 64 x 64]
                          -> Layer 2 (Conv/BN/ReLU) -> [32 x 64 x 64]
                          -> MaxPool2d (2x) -> [32 x 32 x 32]
                          -> Layer 3 (Conv/BN/ReLU) -> [64 x 32 x 32] (Low-Level Features)
                          -> MaxPool2d (2x) -> [64 x 16 x 16] (ASPP Input)
                          -> ASPP (1x1 Conv + Dilated 3x3 Convs [dilation=2, 4, 6] + Image Pooling)
                             Concatenated & Projected to [64 x 16 x 16]
                          -> Decoder:
                             * Upsample ASPP features 2x -> [64 x 32 x 32]
                             * Project Low-Level features (1x1 Conv) -> [16 x 32 x 32]
                             * Concatenate -> [80 x 32 x 32]
                             * Decoder Conv (3x3 blocks) -> [48 x 32 x 32]
                             * Classifier (1x1 Conv) -> [1 x 32 x 32]
                          -> Bilinear Interpolation (4x Upsample) -> Output Mask [1 x 128 x 128]

Key insights / educational takeaways:
    * Atrous Spatial Pyramid Pooling (ASPP): Captures multi-scale contextual features by applying dilated convolutions with varying dilation rates.
    * Encoder-Decoder Skip Connection: Projects early high-resolution features and merges them with upsampled deep features, resolving boundary details lost in downsampling.
    * Rate Adaptation: Rates are scaled from standard ImageNet scales `[6, 12, 18]` down to `[2, 4, 6]` to match the smaller $16 \times 16$ latent feature map size, avoiding filter collapse.
    * Backbone Simplification: A plain max-pooling backbone is used instead of the dilated/atrous ResNet backbone (output-stride 8 or 16) described in the original paper to keep the network lightweight and suitable for $128 \times 128$ educational training.

Run:
    python "04.deeplabv3plus.py" --epochs 10
    python "04.deeplabv3plus.py" --limit 100 --epochs 1        # fast smoke test
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import segmentation_common as mc


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) module with scaled dilation rates."""
    def __init__(self, in_channels: int, out_channels: int, rates: list[int] = [2, 4, 6]):
        super().__init__()
        self.stages = nn.ModuleList()

        # 1. 1x1 Conv branch
        self.stages.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # 2. Dilated Conv branches
        for rate in rates:
            self.stages.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # 3. Image Pooling branch (global context)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Output projection
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        out = []
        for stage in self.stages:
            out.append(stage(x))

        pool_feat = self.image_pool(x)
        pool_feat = F.interpolate(pool_feat, size=(h, w), mode="bilinear", align_corners=True)
        out.append(pool_feat)

        concatenated = torch.cat(out, dim=1)
        return self.project(concatenated)


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ architecture adapted for 128x128 resolution."""
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        # Encoder Backbone layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)  # -> 64x64

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)  # -> 32x32

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)  # -> 16x16

        # ASPP module on high-level feature map
        self.aspp = ASPP(in_channels=64, out_channels=64, rates=[2, 4, 6])

        # Low-level feature projection conv in decoder (typically 1x1 Conv)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Decoder processing blocks after skip-connection concat
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(64 + 16, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(48, out_channels, kernel_size=1)

    def forward(self, x):
        h, w = x.size(2), x.size(3)

        # 1. Encoder Contracting Path
        x1 = self.layer1(x)
        p1 = self.pool1(x1)

        x2 = self.layer2(p1)
        p2 = self.pool2(x2)

        # Low-level feature is x3 (stride 4, shape [B, 64, 32, 32])
        x3 = self.layer3(p2)
        p3 = self.pool3(x3)  # -> shape [B, 64, 16, 16]

        # 2. ASPP Module
        aspp_out = self.aspp(p3)  # -> shape [B, 64, 16, 16]

        # 3. Decoder Expanding Path
        # Bilinear upsample ASPP outputs by 2x to match low-level feature spatial resolution
        aspp_upsampled = F.interpolate(aspp_out, size=x3.shape[2:], mode="bilinear", align_corners=True)

        # Project low-level features
        low_level_feat = self.low_level_conv(x3)  # -> shape [B, 16, 32, 32]

        # Concatenate projected features
        decoder_input = torch.cat([aspp_upsampled, low_level_feat], dim=1)  # -> shape [B, 80, 32, 32]
        decoder_out = self.decoder_conv(decoder_input)  # -> shape [B, 48, 32, 32]
        out = self.final_conv(decoder_out)  # -> shape [B, 1, 32, 32]

        # Final bilinear upsample by 4x to input resolution
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)


def main():
    p = mc.build_argparser("DeepLabV3+ Semantic Segmentation", epochs=10)
    args = p.parse_args()

    device = mc.get_device(args.device)

    # Generate synthetic image-mask dataset
    images, masks = mc.generate_segmentation_dataset(num_samples=args.limit or 1000)

    # Split (80/20)
    split_idx = int(len(images) * 0.8)
    train_img, train_mask = images[:split_idx], masks[:split_idx]
    test_img, test_mask = images[split_idx:], masks[split_idx:]

    model = DeepLabV3Plus().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # Binary segmentation: Soft Dice Loss combined with BCE Loss
    bce_criterion = nn.BCEWithLogitsLoss()
    dice_criterion = mc.SoftDiceLoss()

    print("Training DeepLabV3+...")
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

    # Plot comparative segmentation predictions
    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        grid_path = os.path.join(save_dir, "deeplabv3plus_segmentation_results.png")

        # Capture logits for the first 3 test samples
        val_img = test_img[:3].to(device)
        with torch.no_grad():
            val_logits = model(val_img)

        mc.plot_segmentation_results(test_img[:3], test_mask[:3], val_logits.cpu(), grid_path)


if __name__ == "__main__":
    main()
