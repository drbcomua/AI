"""
03. Pyramid Scene Parsing Network (PSPNet) (Zhao et al., 2017)
=============================================================

Implements a multi-scale context aggregation network using a Pyramid Pooling Module (PPM) at the latent feature level.

Architecture Diagram / Layout:
    Input [3 x 128 x 128] -> Backbone (Conv/BatchNorm/ReLU/MaxPool)
                          -> Feature Map [64 x 16 x 16]
                          -> Pyramid Pooling Module (PPM) at 4 scales:
                             * Scale 1x1: Pool -> 1x1 Conv -> Bilinear Upsample -> [16 x 16 x 16]
                             * Scale 2x2: Pool -> 1x1 Conv -> Bilinear Upsample -> [16 x 16 x 16]
                             * Scale 3x3: Pool -> 1x1 Conv -> Bilinear Upsample -> [16 x 16 x 16]
                             * Scale 6x6: Pool -> 1x1 Conv -> Bilinear Upsample -> [16 x 16 x 16]
                          -> Concatenate (Backbone + 4 scales) -> [128 x 16 x 16]
                          -> Bottleneck Conv (3x3) & Dropout -> [32 x 16 x 16]
                          -> Final Classifier (1x1 Conv) -> [1 x 16 x 16]
                          -> Bilinear Interpolation (8x Upsample) -> Output Mask [1 x 128 x 128]

Key insights / educational takeaways:
    * Receptive Field Enhancement: Standard CNNs suffer from limited receptive fields. PPM aggregates global and regional sub-region context across multiple scales.
    * Sub-region Pooling: Captures spatial patterns at multiple resolutions (from full image context to fine division details).
    * Downscaling Design: Adjusts backbone pooling and pooling bin sizes to match the $128 \times 128$ image dimension without collapsing features.
    * Backbone Simplification: A plain max-pooling backbone is used instead of the dilated/atrous ResNet backbone (output-stride 8) described in the original paper to keep the network lightweight and suitable for $128 \times 128$ educational training.

Run:
    python "03.pspnet.py" --epochs 10
    python "03.pspnet.py" --limit 100 --epochs 1        # fast smoke test
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import segmentation_common as mc


class PSPModule(nn.Module):
    """Pyramid Pooling Module (PPM) to gather context at multiple spatial resolutions."""
    def __init__(self, in_channels: int, bin_sizes: list[int] = [1, 2, 3, 6]):
        super().__init__()
        out_channels = in_channels // len(bin_sizes)
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(bin_size) for bin_size in bin_sizes])
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in bin_sizes
        ])

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [x]
        device = x.device
        for pool, conv in zip(self.pools, self.convs):
            # MPS backend has a known bug where adaptive_avg_pool2d fails if the input size (16)
            # is not divisible by the output bin size (e.g. 3 or 6). To work around this,
            # we temporarily perform pooling on CPU and cast the result back.
            if device.type == "mps":
                pooled = pool(x.cpu()).to(device)
            else:
                pooled = pool(x)
            conv_out = conv(pooled)
            upsampled = F.interpolate(conv_out, size=(h, w), mode="bilinear", align_corners=True)
            pyramids.append(upsampled)
        return torch.cat(pyramids, dim=1)


class PSPNet(nn.Module):
    """PSPNet architecture optimized for 128x128 resolution."""
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        # Custom lightweight backbone downsampling to 16x16
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 64x64

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 16x16
        )

        self.ppm = PSPModule(in_channels=64, bin_sizes=[1, 2, 3, 6])
        # Concatenated features: 64 (backbone) + 4 * 16 (ppm branches) = 128
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        feat = self.backbone(x)
        ppm_feat = self.ppm(feat)
        out = self.bottleneck(ppm_feat)
        out = self.final_conv(out)
        # Upsample 8x back to original resolution
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)


def main():
    p = mc.build_argparser("PSPNet Semantic Segmentation", epochs=10)
    args = p.parse_args()

    device = mc.get_device(args.device)

    # Generate synthetic image-mask dataset
    images, masks = mc.generate_segmentation_dataset(num_samples=args.limit or 1000)

    # Split (80/20)
    split_idx = int(len(images) * 0.8)
    train_img, train_mask = images[:split_idx], masks[:split_idx]
    test_img, test_mask = images[split_idx:], masks[split_idx:]

    model = PSPNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # Binary segmentation: Soft Dice Loss combined with BCE Loss
    bce_criterion = nn.BCEWithLogitsLoss()
    dice_criterion = mc.SoftDiceLoss()

    print("Training Pyramid Scene Parsing Network (PSPNet)...")
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
        grid_path = os.path.join(save_dir, "pspnet_segmentation_results.png")

        # Capture logits for the first 3 test samples
        val_img = test_img[:3].to(device)
        with torch.no_grad():
            val_logits = model(val_img)

        mc.plot_segmentation_results(test_img[:3], test_mask[:3], val_logits.cpu(), grid_path)


if __name__ == "__main__":
    main()
