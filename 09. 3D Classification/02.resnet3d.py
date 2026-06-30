"""
02. 3D ResNet
=============

Residual neural networks adapted for 3D/Volumetric image classification (Hara et al., 2017).

Architecture Diagram / Layout:
    Input [1 x 28 x 28 x 28] -> Conv3d (16)
                             -> ResNet3DBlock (16 -> 16, stride=1)
                             -> ResNet3DBlock (16 -> 32, stride=2) [32 x 14 x 14 x 14]
                             -> ResNet3DBlock (32 -> 64, stride=2) [64 x 7 x 7 x 7]
                             -> AdaptiveAvgPool3d (1) -> Linear (10)

Key insights / educational takeaways:
    * Implements 3D residual skip connections to stabilize deep volumetric gradients.
    * Explains how global average pooling reduces parameter size compared to dense flatten layers.

Run:
    python "02.resnet3d.py" --epochs 5
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import voxel_common as mc


class ResNet3DBlock(nn.Module):
    """Volumetric residual learning block with identity skip mapping."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet3D(nn.Module):
    """Volumetric residual classification network."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.block1 = ResNet3DBlock(16, 16, stride=1)
        self.block2 = ResNet3DBlock(16, 32, stride=2) # Downsample to 14x14x14
        self.block3 = ResNet3DBlock(32, 64, stride=2) # Downsample to 7x7x7
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x):
        h = self.in_conv(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = F.adaptive_avg_pool3d(h, 1)
        h = h.reshape(h.size(0), -1)
        return self.linear(h)


def main():
    p = mc.build_argparser("Volumetric ResNet3D on 3D MNIST", epochs=5)
    args = p.parse_args()

    device = mc.get_device(args.device)

    # Load 3D MNIST
    volumes, labels = mc.load_3d_mnist(limit=args.limit)

    # Train / test split (80/20)
    split_idx = int(len(volumes) * 0.8)
    train_vol, train_lbl = volumes[:split_idx], labels[:split_idx]
    test_vol, test_lbl = volumes[split_idx:], labels[split_idx:]

    model = ResNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("Training 3D ResNet...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    train_dataset = TensorDataset(train_vol, train_lbl)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Apply random 3D rotations for data augmentation
            x = mc.random_rotate_3d(x, max_angle=15.0)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_acc = correct / total
        train_loss_avg = epoch_loss / total
        print(f"Epoch {epoch:2d}/{args.epochs} | train_loss: {train_loss_avg:.4f} | train_acc: {train_acc * 100:.2f}%")

    print("-" * 64)

    # Test evaluation
    test_dataset = TensorDataset(test_vol, test_lbl)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    all_preds = []
    all_targets = []
    correct = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds.cpu() == y).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.numpy())

    test_acc = correct / len(test_dataset)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Save output visualization figures
    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))

        # Plot test confusion matrix
        cm_path = os.path.join(save_dir, "resnet3d_confusion_matrix.png")
        mc.plot_confusion_matrix(np.array(all_targets), np.array(all_preds), cm_path, "3D ResNet")


if __name__ == "__main__":
    main()
