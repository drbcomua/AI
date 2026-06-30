"""
voxel_common.py
===============

Shared utilities for 3D/Volumetric MNIST Classification.
Handles synthetic 3D MNIST dataset creation from upscaled digits, 3D spatial transformations
using PyTorch affine grids, 3D voxel matplotlib rendering, and confusion matrix plots.
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits


# --------------------------------------------------------------------------- #
# Volumetric MNIST Dataset Generator
# --------------------------------------------------------------------------- #
def load_3d_mnist(limit: int | None = None):
    """Loads scikit-learn digits, upscales to 28x28, and extrudes to 28x28x28 3D voxel grids.

    Provides a zero-network-download, highly reliable volumetric dataset.
    """
    print("Loading digits dataset...")
    digits = load_digits()
    images = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) # [N, 1, 8, 8]
    labels = torch.tensor(digits.target, dtype=torch.long)

    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]

    # Normalize to [0, 1]
    images = images / 16.0

    print("Interpolating 8x8 digits to 28x28...")
    # Bilinear interpolation to scale up to 28x28
    images_28 = F.interpolate(images, size=(28, 28), mode="bilinear", align_corners=True)

    print("Extruding 2D images to 3D voxel volumes (28x28x28)...")
    N = images_28.size(0)

    # Vectorized 3D extrusion:
    # We construct a Z coordinate grid and set voxels active if close to the center
    # thickness is proportional to the pixel intensity
    z_coords = torch.arange(28, dtype=torch.float32).view(1, 1, 28, 1, 1) # [1, 1, Depth, 1, 1]
    dz = torch.abs(z_coords - 13.5) # distance to middle slice

    intensity = images_28.unsqueeze(2) # [N, 1, 1(Depth), H, W]

    # Voxel active threshold: dz <= intensity * 4.5
    mask = (dz <= (intensity * 4.5 + 0.5)).float()
    volumes = intensity * mask # [N, 1, Depth, Height, Width]

    print(f"3D MNIST loaded: {volumes.size(0)} samples of shape {list(volumes.shape[1:])}")
    return volumes, labels


def random_rotate_3d(volumes: torch.Tensor, max_angle: float = 15.0):
    """Applies random 3D rotations (roll, pitch, yaw) to a batch of volumetric tensors using affine grids."""
    B = volumes.size(0)
    device = volumes.device

    # Random angles in radians
    angles = (torch.rand(B, 3, device=device) * 2.0 - 1.0) * (max_angle * np.pi / 180.0)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # Roll (x-axis rotation)
    R_x = torch.zeros(B, 3, 3, device=device)
    R_x[:, 0, 0] = 1.0
    R_x[:, 1, 1] = cos[:, 0]
    R_x[:, 1, 2] = -sin[:, 0]
    R_x[:, 2, 1] = sin[:, 0]
    R_x[:, 2, 2] = cos[:, 0]

    # Pitch (y-axis rotation)
    R_y = torch.zeros(B, 3, 3, device=device)
    R_y[:, 0, 0] = cos[:, 1]
    R_y[:, 0, 2] = sin[:, 1]
    R_y[:, 1, 1] = 1.0
    R_y[:, 2, 0] = -sin[:, 1]
    R_y[:, 2, 2] = cos[:, 1]

    # Yaw (z-axis rotation)
    R_z = torch.zeros(B, 3, 3, device=device)
    R_z[:, 0, 0] = cos[:, 2]
    R_z[:, 0, 1] = -sin[:, 2]
    R_z[:, 1, 0] = sin[:, 2]
    R_z[:, 1, 1] = cos[:, 2]
    R_z[:, 2, 2] = 1.0

    # Combine rotations: R = R_z * R_y * R_x
    R = torch.bmm(R_z, torch.bmm(R_y, R_x))

    # Affine matrix [B, 3, 4]
    theta = torch.zeros(B, 3, 4, device=device)
    theta[:, :3, :3] = R

    grid = F.affine_grid(theta, size=volumes.size(), align_corners=True)
    rotated = F.grid_sample(volumes, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return rotated


# --------------------------------------------------------------------------- #
# Representation converters: voxels -> point cloud / multi-view
# --------------------------------------------------------------------------- #
def voxels_to_points(volumes: torch.Tensor, num_points: int = 256, threshold: float = 0.1):
    """Sample a fixed-size point cloud from the occupied voxels of each volume.

    volumes: [B, 1, D, H, W] -> points [B, num_points, 3] with coords in [-1, 1].
    """
    B, device, D = volumes.size(0), volumes.device, volumes.size(2)
    out = torch.zeros(B, num_points, 3, device=device)
    for b in range(B):
        occ = (volumes[b, 0] > threshold).nonzero(as_tuple=False).float()   # [M, 3]
        if occ.size(0) == 0:
            continue
        idx = torch.randint(0, occ.size(0), (num_points,), device=device)    # sample (w/ replacement)
        out[b] = (occ[idx] / (D - 1)) * 2 - 1
    return out


def index_points(points: torch.Tensor, idx: torch.Tensor):
    """Batched gather: points [B, N, C], idx [B, S, ...] -> [B, S, ..., C]."""
    B = points.size(0)
    view_shape = [B] + [1] * (idx.dim() - 1)
    batch = torch.arange(B, device=points.device).view(view_shape).expand_as(idx)
    return points[batch, idx]


def knn_indices(x: torch.Tensor, k: int, exclude_self: bool = True):
    """k-nearest-neighbour indices of each point. x: [B, N, C] -> idx [B, N, k]."""
    dist = torch.cdist(x, x)
    kk = k + 1 if exclude_self else k
    idx = dist.topk(kk, dim=-1, largest=False).indices
    return idx[:, :, 1:] if exclude_self else idx


def voxels_to_multiview(volumes: torch.Tensor, num_views: int = 6):
    """Render silhouette views by rotating around the vertical axis and max-projecting.

    volumes: [B, 1, D, H, W] -> views [B, V, 1, H, W].
    """
    B, device = volumes.size(0), volumes.device
    views = []
    for v in range(num_views):
        a = v * (2 * np.pi / num_views)
        cos, sin = float(np.cos(a)), float(np.sin(a))
        R = torch.tensor([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]], device=device)  # rotate about height
        theta = torch.zeros(B, 3, 4, device=device)
        theta[:, :3, :3] = R
        grid = F.affine_grid(theta, size=volumes.size(), align_corners=True)
        rot = F.grid_sample(volumes, grid, align_corners=True)
        views.append(rot.max(dim=2).values)                                  # max-project along depth
    return torch.stack(views, dim=1)


# --------------------------------------------------------------------------- #
# Shared training / evaluation loop
# --------------------------------------------------------------------------- #
def train_and_eval(model, train_vol, train_lbl, test_vol, test_lbl, device, args,
                   model_name: str, augment: bool = True):
    """Standard train loop (with 3D-rotation augmentation) + test accuracy + confusion matrix."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    print(f"Device: {device} | trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("-" * 64)
    train_loader = DataLoader(TensorDataset(train_vol, train_lbl), batch_size=args.batch_size, shuffle=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        el = cor = tot = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if augment:
                x = random_rotate_3d(x, max_angle=15.0)
            opt.zero_grad()
            out = model(x)
            loss = ce(out, y)
            loss.backward(); opt.step()
            el += loss.item() * x.size(0); cor += (out.argmax(1) == y).sum().item(); tot += x.size(0)
        print(f"Epoch {epoch:2d}/{args.epochs} | train_loss {el / tot:.4f} | train_acc {cor / tot * 100:.2f}%")
    print("-" * 64)

    test_loader = DataLoader(TensorDataset(test_vol, test_lbl), batch_size=args.batch_size, shuffle=False)
    model.eval()
    preds, tgts, cor = [], [], 0
    with torch.no_grad():
        for x, y in test_loader:
            p = model(x.to(device)).argmax(1).cpu()
            cor += (p == y).sum().item(); preds.extend(p.numpy()); tgts.extend(y.numpy())
    acc = cor / len(test_vol)
    print(f"{model_name} Test Accuracy: {acc * 100:.2f}%")
    if not args.no_figure:
        slug = model_name.lower().replace(" ", "_").replace("+", "plus").replace("-", "")
        save_dir = os.path.dirname(os.path.abspath(__file__))
        plot_confusion_matrix(np.array(tgts), np.array(preds),
                              os.path.join(save_dir, f"{slug}_confusion_matrix.png"), model_name)
    return acc


# --------------------------------------------------------------------------- #
# Device, Argparser, and Common Functions
# --------------------------------------------------------------------------- #
def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_argparser(description: str, epochs: int = 5, batch_size: int = 32, lr: float = 1e-3):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--batch-size", type=int, default=batch_size)
    p.add_argument("--lr", type=float, default=lr)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--no-figure", action="store_true", help="do not save figures")
    p.add_argument("--limit", type=int, default=None, help="limit dataset samples for quick local checks")
    return p


# --------------------------------------------------------------------------- #
# Visualizations: 3D Voxels and Confusion Matrix
# --------------------------------------------------------------------------- #
def plot_voxel_grid(volume: torch.Tensor, save_path: str, title: str = "3D Voxel Digit"):
    """Draws and saves a beautiful 3D voxel representation using matplotlib voxels."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping voxel rendering plot: {e})")
        return

    # volume shape: [C, D, H, W] or [D, H, W]
    if volume.ndim == 4:
        vol_np = volume[0].detach().cpu().numpy()
    else:
        vol_np = volume.detach().cpu().numpy()

    # Create binary occupancy grid based on threshold
    occupancy = vol_np > 0.25

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Draw voxels
    ax.voxels(
        occupancy,
        facecolors="royalblue",
        edgecolors="navy",
        linewidth=0.2,
        alpha=0.6
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Depth (D)")
    ax.set_ylabel("Height (H)")
    ax.set_zlabel("Width (W)")

    # Adjust view angle for depth perception
    ax.view_init(elev=20, azim=45)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved voxel grid visualization -> {save_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str, model_name: str):
    """Generates and saves a confusion matrix heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
    except Exception as e:
        print(f"(skipping confusion matrix plot: {e})")
        return

    cm = confusion_matrix(y_true, y_pred)
    classes = [str(i) for i in range(10)]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f"Confusion Matrix ({model_name})",
        ylabel="True label",
        xlabel="Predicted label"
    )

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved confusion matrix plot -> {save_path}")
