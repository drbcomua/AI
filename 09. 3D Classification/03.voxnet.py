"""
03. VoxNet
==========

Real-time Volumetric CNN for 3D occupancy grids (Maturana & Scherer, IROS 2015).

Architecture Diagram / Layout:
    Input [1 x 28 x 28 x 28] -> Conv3d (32, k=5, stride=2) [32 x 12 x 12 x 12]
                             -> Conv3d (32, k=3, stride=1) [32 x 10 x 10 x 10]
                             -> MaxPool3d (2) [32 x 5 x 5 x 5]
                             -> Linear (128) -> Dropout (0.5) -> Linear (10)

Key insights / educational takeaways:
    * Tailored for sparse binary voxel data (like Occupancy Grids and LiDAR point clouds).
    * Uses strided convolutions and no padding to rapidly reduce volume dimensionality, saving computation.

Run:
    python "03.voxnet.py" --epochs 5
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import voxel_common as mc


class VoxNet(nn.Module):
    """VoxNet model optimized for 3D binary grids."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # No padding to shrink the sparse grid quickly
            nn.Conv3d(1, 32, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1),

            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1),

            nn.MaxPool3d(2) # -> 5x5x5
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 5 * 5 * 5, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        h = self.features(x)
        h = h.reshape(h.size(0), -1)
        return self.classifier(h)


def main():
    p = mc.build_argparser("VoxNet occupancy grid model on 3D MNIST", epochs=5)
    args = p.parse_args()

    device = mc.get_device(args.device)

    # Load 3D MNIST
    volumes, labels = mc.load_3d_mnist(limit=args.limit)

    # Train / test split (80/20)
    split_idx = int(len(volumes) * 0.8)
    train_vol, train_lbl = volumes[:split_idx], labels[:split_idx]
    test_vol, test_lbl = volumes[split_idx:], labels[split_idx:]

    model = VoxNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("Training VoxNet...")
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
        cm_path = os.path.join(save_dir, "voxnet_confusion_matrix.png")
        mc.plot_confusion_matrix(np.array(all_targets), np.array(all_preds), cm_path, "VoxNet")


if __name__ == "__main__":
    main()
