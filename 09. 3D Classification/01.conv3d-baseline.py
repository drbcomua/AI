"""
01. Volumetric CNN Baseline
===========================

Basic 3D Convolutional Neural Network for voxel classification.

Architecture Diagram / Layout:
    Input [1 x 28 x 28 x 28] -> Conv3d (16) -> MaxPool3d [16 x 14 x 14 x 14]
                             -> Conv3d (32) -> MaxPool3d [32 x 7 x 7 x 7]
                             -> Linear (128) -> Linear (10)

Key insights / educational takeaways:
    * Demonstrates how standard 2D convolutions generalize to 3D voxel spaces.
    * Highlights the parameter and floating-point operations scaling with 3D kernels.

Run:
    python "01.conv3d-baseline.py" --epochs 5
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import voxel_common as mc


class Conv3DNet(nn.Module):
    """Basic 3D CNN baseline for volumetric grid classification."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2), # -> 14x14x14

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2), # -> 7x7x7
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        h = self.features(x)
        h = h.reshape(h.size(0), -1)
        return self.classifier(h)


def main():
    p = mc.build_argparser("Volumetric CNN Baseline on 3D MNIST", epochs=5)
    args = p.parse_args()

    device = mc.get_device(args.device)

    # Load 3D MNIST
    volumes, labels = mc.load_3d_mnist(limit=args.limit)

    # Train / test split (80/20)
    split_idx = int(len(volumes) * 0.8)
    train_vol, train_lbl = volumes[:split_idx], labels[:split_idx]
    test_vol, test_lbl = volumes[split_idx:], labels[split_idx:]

    model = Conv3DNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("Training 3D CNN Volumetric Baseline...")
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

            # Apply random 3D rotations on training set for augmentation
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

        # Plot 3D voxel representation of first test image
        voxel_path = os.path.join(save_dir, "conv3d_digit_sample.png")
        mc.plot_voxel_grid(test_vol[0], voxel_path, title=f"Voxel Representation of Target Label {test_lbl[0].item()}")

        # Plot test confusion matrix
        cm_path = os.path.join(save_dir, "conv3d_confusion_matrix.png")
        mc.plot_confusion_matrix(np.array(all_targets), np.array(all_preds), cm_path, "Volumetric CNN Baseline")


if __name__ == "__main__":
    main()
