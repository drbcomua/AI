"""
04. 3D Vision Transformer (ViT3D)
=================================

Attention-based volumetric model dividing voxel space into 3D Tubelets.

Architecture Diagram / Layout:
    Input [1 x 28 x 28 x 28] -> Voxel Patch Extraction (4x4x4 tubelets) -> 343 Patches
                             -> Flatten & Project [343 x 64] -> Prepend CLS token [344 x 64]
                             -> Add 3D Positional Encoding -> 2x Transformer Encoder Layer
                             -> CLS representation [64] -> Linear (10)

Key insights / educational takeaways:
    * Replaces dense 3D convolutions with global self-attention over volumetric tokens (patches).
    * Teaches how to project 3D grids into sequential inputs for sequence-based models.

Run:
    python "04.vit3d.py" --epochs 5
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import voxel_common as mc


class ViT3D(nn.Module):
    """3D Vision Transformer (ViT3D) using volumetric patches (Tubelets)."""
    def __init__(self, volume_size: int = 28, patch_size: int = 4, in_channels: int = 1,
                 num_classes: int = 10, embed_dim: int = 64, depth: int = 2, heads: int = 4, mlp_dim: int = 128):
        super().__init__()
        assert volume_size % patch_size == 0, "Volume size must be divisible by patch size"
        self.patch_size = patch_size
        self.num_patches = (volume_size // patch_size) ** 3
        patch_dim = in_channels * (patch_size ** 3)

        # Patch projection
        self.patch_to_embedding = nn.Linear(patch_dim, embed_dim)

        # CLS token & Positional Encodings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classifier
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        P = self.patch_size

        # Volumetric patch extraction via PyTorch unfold:
        # Shape: [B, C, D/P, P, H/P, P, W/P, P]
        x_patches = x.unfold(2, P, P).unfold(3, P, P).unfold(4, P, P)
        x_patches = x_patches.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
        x_patches = x_patches.view(B, -1, C * (P ** 3))

        # Project patches
        x_embeddings = self.patch_to_embedding(x_patches)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_embeddings = torch.cat((cls_tokens, x_embeddings), dim=1)

        # Add positional encodings
        x_embeddings += self.pos_embedding

        # Encoder pass
        out = self.transformer(x_embeddings)

        # Predict based on CLS representation
        cls_rep = out[:, 0]
        return self.mlp_head(cls_rep)


def main():
    p = mc.build_argparser("3D Vision Transformer (ViT3D) on 3D MNIST", epochs=5)
    args = p.parse_args()

    device = mc.get_device(args.device)

    # Load 3D MNIST
    volumes, labels = mc.load_3d_mnist(limit=args.limit)

    # Train / test split (80/20)
    split_idx = int(len(volumes) * 0.8)
    train_vol, train_lbl = volumes[:split_idx], labels[:split_idx]
    test_vol, test_lbl = volumes[split_idx:], labels[split_idx:]

    model = ViT3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("Training 3D Vision Transformer (ViT3D)...")
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
        cm_path = os.path.join(save_dir, "vit3d_confusion_matrix.png")
        mc.plot_confusion_matrix(np.array(all_targets), np.array(all_preds), cm_path, "3D Vision Transformer")


if __name__ == "__main__":
    main()
