"""
02. U-Net
=========

Symmetric contracting-expanding segmentation architecture with skip connections (Ronneberger et al., 2015).

Architecture Diagram / Layout:
    Input [3 x 128 x 128] ----> DoubleConv (16) -------------------------------+ (Skip)
                                     | MaxPool2d                                |
                                     v                                          v
                                DoubleConv (32) -------------------> +---- Concatenate & DoubleConv (16) -> Conv 1x1 [1 x 128 x 128]
                                     | MaxPool2d                     |          ^
                                     v                               v          | ConvTranspose2d (upsample)
                                DoubleConv (64) -----------> +---- Concatenate & DoubleConv (32)
                                     | MaxPool2d             |          ^
                                     v                       v          | ConvTranspose2d (upsample)
                                DoubleConv (128) ------> ConvTranspose2d (upsample)

Key insights / educational takeaways:
    * Implements concatenation-based skip connections between corresponding encoder and decoder levels.
    * Allows high-resolution spatial details (like sharp edge borders) to bypass the information bottleneck of deep encoders.

Run:
    python "02.unet.py" --epochs 10
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import segmentation_common as mc


class DoubleConv(nn.Module):
    """Utility block applying Conv2d -> BatchNorm2d -> ReLU twice."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Symmetric U-Net segmentation network with contracting and expanding paths."""
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        # Encoder Contracting Path
        self.enc1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2) # -> 64x64

        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2) # -> 32x32

        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(2) # -> 16x16

        # Bottleneck
        self.bottleneck = DoubleConv(64, 128)

        # Decoder Expanding Path
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # -> 32x32
        self.dec3 = DoubleConv(128, 64) # 64 (skip) + 64 (up) = 128

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) # -> 64x64
        self.dec2 = DoubleConv(64, 32) # 32 (skip) + 32 (up) = 64

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2) # -> 128x128
        self.dec1 = DoubleConv(32, 16) # 16 (skip) + 16 (up) = 32

        # Final Pixel Classifier
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # 1. Contracting Path (Encoder)
        s1 = self.enc1(x)
        p1 = self.pool1(s1)

        s2 = self.enc2(p1)
        p2 = self.pool2(s2)

        s3 = self.enc3(p2)
        p3 = self.pool3(s3)

        # 2. Bottleneck
        b = self.bottleneck(p3)

        # 3. Expanding Path with Skip Connections (Decoder)
        up3_out = self.up3(b)
        merge3 = torch.cat([up3_out, s3], dim=1) # Concatenate along channels
        d3 = self.dec3(merge3)

        up2_out = self.up2(d3)
        merge2 = torch.cat([up2_out, s2], dim=1)
        d2 = self.dec2(merge2)

        up1_out = self.up1(d2)
        merge1 = torch.cat([up1_out, s1], dim=1)
        d1 = self.dec1(merge1)

        return self.final_conv(d1)


def main():
    p = mc.build_argparser("U-Net Semantic Segmentation", epochs=10)
    args = p.parse_args()

    device = mc.get_device(args.device)

    # Generate synthetic image-mask dataset
    images, masks = mc.generate_segmentation_dataset(num_samples=args.limit or 1000)

    # Split (80/20)
    split_idx = int(len(images) * 0.8)
    train_img, train_mask = images[:split_idx], masks[:split_idx]
    test_img, test_mask = images[split_idx:], masks[split_idx:]

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # Binary segmentation: Soft Dice Loss combined with BCE Loss
    bce_criterion = nn.BCEWithLogitsLoss()
    dice_criterion = mc.SoftDiceLoss()

    print("Training U-Net...")
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
        grid_path = os.path.join(save_dir, "unet_segmentation_results.png")

        # Capture logits for the first 3 test samples
        val_img = test_img[:3].to(device)
        with torch.no_grad():
            val_logits = model(val_img)

        mc.plot_segmentation_results(test_img[:3], test_mask[:3], val_logits.cpu(), grid_path)


if __name__ == "__main__":
    main()
