"""
05. SegNet (Badrinarayanan et al., 2015)
=======================================

An encoder-decoder architecture for semantic pixel-wise segmentation that highlights the use of pooling indices for upsampling.

Architecture Diagram / Layout:
    Input [3 x 128 x 128] -> Layer 1: Conv -> BN -> ReLU -> MaxPool2d (return indices) -> [16 x 64 x 64]
                          -> Layer 2: Conv -> BN -> ReLU -> MaxPool2d (return indices) -> [32 x 32 x 32]
                          -> Layer 3: Conv -> BN -> ReLU -> MaxPool2d (return indices) -> [64 x 16 x 16]
                          -> Decoder Layer 3: MaxUnpool2d -> Conv -> BN -> ReLU -> [32 x 32 x 32]
                          -> Decoder Layer 2: MaxUnpool2d -> Conv -> BN -> ReLU -> [16 x 64 x 64]
                          -> Decoder Layer 1: MaxUnpool2d -> Conv -> BN -> Output -> [1 x 128 x 128]

Key insights / educational takeaways:
    * Upsampling with Pooling Indices: SegNet saves the locations of the maximum values during max-pooling in the encoder. The decoder uses these indices to unpool the feature maps, positioning activations back in their original spatial coordinates.
    * Resource Efficiency: Upsampling via indices removes the need for learning upsampling parameters (unlike transposed convolutions in FCN) or storing entire feature maps (unlike U-Net's skip connection concatenation).
    * Backbone Simplification: A plain max-pooling backbone with unpooling is used instead of the VGG-16 base described in the original paper to keep the network lightweight and suitable for $128 \times 128$ educational training.

Run:
    python "05.segnet.py" --epochs 10
    python "05.segnet.py" --limit 100 --epochs 1        # fast smoke test
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import segmentation_common as mc


class SegNet(nn.Module):
    """SegNet architecture optimized for 128x128 resolution."""
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        # Encoder contracting path
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # -> 64x64

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # -> 32x32

        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # -> 16x16

        # Decoder expanding path
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        size1 = x1.size()
        p1, ind1 = self.pool1(x1)

        x2 = self.enc2(p1)
        size2 = x2.size()
        p2, ind2 = self.pool2(x2)

        x3 = self.enc3(p2)
        size3 = x3.size()
        p3, ind3 = self.pool3(x3)

        # Decoder (Unpooling and convolutions)
        d3 = self.unpool3(p3, ind3, output_size=size3)
        d3 = self.dec3(d3)

        d2 = self.unpool2(d3, ind2, output_size=size2)
        d2 = self.dec2(d2)

        d1 = self.unpool1(d2, ind1, output_size=size1)
        d1 = self.dec1(d1)

        return d1


def main():
    p = mc.build_argparser("SegNet Semantic Segmentation", epochs=10)
    args = p.parse_args()

    device = mc.get_device(args.device)

    # Generate synthetic image-mask dataset
    images, masks = mc.generate_segmentation_dataset(num_samples=args.limit or 1000)

    # Split (80/20)
    split_idx = int(len(images) * 0.8)
    train_img, train_mask = images[:split_idx], masks[:split_idx]
    test_img, test_mask = images[split_idx:], masks[split_idx:]

    model = SegNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # Binary segmentation: Soft Dice Loss combined with BCE Loss
    bce_criterion = nn.BCEWithLogitsLoss()
    dice_criterion = mc.SoftDiceLoss()

    print("Training SegNet...")
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
        grid_path = os.path.join(save_dir, "segnet_segmentation_results.png")

        # Capture logits for the first 3 test samples
        val_img = test_img[:3].to(device)
        with torch.no_grad():
            val_logits = model(val_img)

        mc.plot_segmentation_results(test_img[:3], test_mask[:3], val_logits.cpu(), grid_path)


if __name__ == "__main__":
    main()
