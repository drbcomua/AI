"""
segmentation_common.py
======================

Shared utilities for Image Segmentation.
Handles synthetic image-mask dataset generation, pixel-level augmentations,
Soft Dice Loss, IoU metric trackers, and side-by-side plot layout generators.
"""

from __future__ import annotations

import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Synthetic Segmentation Image-Mask Dataset Generator
# --------------------------------------------------------------------------- #
def generate_segmentation_dataset(num_samples: int = 1000):
    """Generates synthetic RGB images and corresponding binary target segmentation masks.

    Images: [num_samples, 3, 128, 128]
    Masks: [num_samples, 1, 128, 128]
    """
    random.seed(42)
    np.random.seed(42)

    images = []
    masks = []

    shapes = ["circle", "square", "triangle"]

    for _ in range(num_samples):
        # Image background: dark gray
        img = np.ones((128, 128, 3), dtype=np.float32) * 0.15
        # Mask background: 0
        mask = np.zeros((128, 128), dtype=np.float32)

        shape = random.choice(shapes)
        color = [random.uniform(0.3, 1.0) for _ in range(3)]

        cx = random.randint(40, 88)
        cy = random.randint(40, 88)
        r = random.randint(18, 35)

        Y, X = np.ogrid[:128, :128]

        if shape == "circle":
            s_mask = (X - cx)**2 + (Y - cy)**2 <= r**2
            img[s_mask] = color
            mask[s_mask] = 1.0
        elif shape == "square":
            s_mask = (np.abs(X - cx) <= r) & (np.abs(Y - cy) <= r)
            img[s_mask] = color
            mask[s_mask] = 1.0
        elif shape == "triangle":
            s_mask = (Y >= cy - r) & (Y <= cy + r) & (np.abs(X - cx) <= (cy + r - Y))
            img[s_mask] = color
            mask[s_mask] = 1.0

        # Add some random background noise (distracting small circles)
        for _ in range(3):
            ncx = random.randint(10, 118)
            ncy = random.randint(10, 118)
            # Ensure noise does not overlap with primary shape
            if (ncx - cx)**2 + (ncy - cy)**2 > (r + 15)**2:
                nr = random.randint(3, 8)
                n_mask = (X - ncx)**2 + (Y - ncy)**2 <= nr**2
                n_color = [random.uniform(0.1, 0.4) for _ in range(3)]
                img[n_mask] = n_color

        images.append(img.transpose(2, 0, 1))
        masks.append(mask[np.newaxis, :, :])

    return torch.tensor(np.array(images), dtype=torch.float32), torch.tensor(np.array(masks), dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Soft Dice Loss and mIoU Tracker
# --------------------------------------------------------------------------- #
class SoftDiceLoss(nn.Module):
    """Soft Dice Loss function to handle foreground/background imbalance."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


def compute_iou(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes Intersection over Union (IoU) metric for binary predictions."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    intersection = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item() - intersection

    if union == 0:
        return 1.0
    return intersection / union


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


def build_argparser(description: str, epochs: int = 10, batch_size: int = 32, lr: float = 2e-3):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--batch-size", type=int, default=batch_size)
    p.add_argument("--lr", type=float, default=lr)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--no-figure", action="store_true", help="do not save figures")
    p.add_argument("--limit", type=int, default=None, help="limit dataset samples for quick local checks")
    return p


# --------------------------------------------------------------------------- #
# Visualizations: Side-by-side [Image | GT Mask | Pred Mask] Grid
# --------------------------------------------------------------------------- #
def plot_segmentation_results(images: torch.Tensor, ground_truth: torch.Tensor,
                              predictions: torch.Tensor, save_path: str):
    """Plots a 3x3 grid showing side-by-side comparison of Input, Ground Truth, and Predictions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping segmentation plotting: {e})")
        return

    n_display = min(3, len(images))
    fig, axes = plt.subplots(n_display, 3, figsize=(9, 3 * n_display))

    # Handle single sample case dimension wrapping
    if n_display == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_display):
        # 1. Input Image
        img = images[i].numpy().transpose(1, 2, 0)
        img = np.clip(img, 0.0, 1.0)
        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("Input Image", fontsize=11, fontweight="bold")

        # 2. Ground Truth Mask
        gt = ground_truth[i, 0].numpy()
        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].axis("off")
        if i == 0:
            axes[i, 1].set_title("Ground Truth Mask", fontsize=11, fontweight="bold")

        # 3. Predicted Mask
        pred = (torch.sigmoid(predictions[i, 0]) > 0.5).float().numpy()
        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].axis("off")
        if i == 0:
            axes[i, 2].set_title("Predicted Mask", fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved segmentation predictions grid -> {save_path}")
