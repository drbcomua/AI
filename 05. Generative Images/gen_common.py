"""
gen_common.py
=============

Shared utilities for Generative Image modeling in this folder.
Handles raw dataset downloads (FashionMNIST & MNIST), sequence parsing,
latent space interpolation (slerp), and grid output visualizations.
"""

from __future__ import annotations

import os
import gzip
import ssl
import urllib.request
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Mirrors
FASHION_MIRROR = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"

_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# --------------------------------------------------------------------------- #
# Dataset Download and Parsing
# --------------------------------------------------------------------------- #
def _download(mirror: str, fname: str, data_dir: str) -> str:
    """Download standard dataset gzip file if not cached, with SSL validation bypass."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, fname)
    if os.path.exists(path):
        return path

    url = mirror + fname
    print(f"Downloading {url}...")
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=120) as r:
            blob = r.read()
    except Exception:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=120) as r:
            blob = r.read()

    with open(path, "wb") as f:
        f.write(blob)
    return path


def _read_idx_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        assert magic == 2051, f"bad image magic {magic} in {path}"
        n = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)


def _read_idx_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        assert magic == 2049, f"bad label magic {magic} in {path}"
        n = int.from_bytes(f.read(4), "big")
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8)


def get_dataloaders(name: str = "fashion", batch_size: int = 128, limit: int | None = None,
                    data_dir: str = _DATA_DIR, num_workers: int = 0):
    """Load and normalize images. Outputs pixel values in range [-1, 1].

    Returns (train_loader, test_loader, num_classes).
    """
    mirror = FASHION_MIRROR if name == "fashion" else MNIST_MIRROR
    folder = os.path.join(data_dir, name)

    paths = {k: _download(mirror, v, folder) for k, v in _FILES.items()}
    X_train = _read_idx_images(paths["train_images"])
    y_train = _read_idx_labels(paths["train_labels"])
    X_test = _read_idx_images(paths["test_images"])
    y_test = _read_idx_labels(paths["test_labels"])

    if limit is not None:
        X_train, y_train = X_train[:limit], y_train[:limit]
        n_test = max(1, limit // 6)
        X_test, y_test = X_test[:n_test], y_test[:n_test]

    def to_tensor(X, y):
        # Normalize from [0, 255] to [-1, 1] for stable GAN/Diffusion outputs
        X = torch.from_numpy(X.astype(np.float32) / 127.5 - 1.0).unsqueeze(1) # [N, 1, 28, 28]
        y = torch.from_numpy(y.astype(np.int64))
        return TensorDataset(X, y)

    train_ds = to_tensor(X_train, y_train)
    test_ds = to_tensor(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, 10


# --------------------------------------------------------------------------- #
# Device, Argparser, and Interpolation
# --------------------------------------------------------------------------- #
def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_argparser(description: str, epochs: int = 5, batch_size: int = 128, lr: float = 1e-3):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dataset", type=str, default="fashion", choices=["fashion", "mnist"])
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--batch-size", type=int, default=batch_size)
    p.add_argument("--lr", type=float, default=lr)
    p.add_argument("--limit", type=int, default=None, help="use first N samples for quick checks")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--latent-dim", type=int, default=64, help="dimension of latent noise vector z")
    p.add_argument("--no-figure", action="store_true", help="do not save generation plots")
    p.add_argument("--variant", type=str, default=None)
    return p


def slerp(val, low, high):
    """Spherical linear interpolation between low and high vectors."""
    low_norm = low / torch.norm(low, dim=-1, keepdim=True)
    high_norm = high / torch.norm(high, dim=-1, keepdim=True)
    omega = torch.acos(torch.clamp((low_norm * high_norm).sum(dim=-1), -1.0, 1.0))
    so = torch.sin(omega)
    if torch.isnan(so) or so < 1e-6:
        # Fallback to linear interpolation
        return (1.0 - val) * low + val * high
    return (torch.sin((1.0 - val) * omega) / so).unsqueeze(-1) * low + (torch.sin(val * omega) / so).unsqueeze(-1) * high


# --------------------------------------------------------------------------- #
# Frechet Inception Distance (FID)
# --------------------------------------------------------------------------- #
class _FIDExtractor(nn.Module):
    """Small CNN classifier; its penultimate features are used for FID.

    Real FID uses InceptionV3 (pretrained on ImageNet) features; that needs
    torchvision + a weight download. For this offline, MNIST-scale folder we instead
    briefly train this tiny classifier on the dataset and use its features — the
    standard lightweight "MNIST-FID". Trained features (unlike random ones) capture
    realism, so the metric ranks generators correctly.
    """
    def __init__(self, num_classes: int = 10, feat_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),       # -> 14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # -> 7
            nn.Conv2d(32, feat_dim, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


def _build_fid_extractor(train_loader, device, steps: int = 300):
    """Briefly train the feature extractor as a classifier on the real data."""
    net = _FIDExtractor().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    net.train()
    it = 0
    while it < steps:
        for x, y in train_loader:
            opt.zero_grad()
            ce(net(x.to(device)), y.to(device)).backward()
            opt.step()
            it += 1
            if it >= steps:
                break
    net.eval()
    return net


def compute_fid(real_images, fake_images, train_loader, device=None, steps: int = 300) -> float:
    """Frechet distance between real and generated images, in the feature space of a
    classifier trained briefly on ``train_loader`` (lower = closer to real).

    Self-contained (no InceptionV3/torchvision). Inputs are tensors [N, 1, 28, 28]
    in [-1, 1]; ``train_loader`` supplies the labeled real data for the extractor.
    """
    device = device or get_device()
    rng_state = torch.get_rng_state()
    torch.manual_seed(0)                                   # reproducible extractor
    extractor = _build_fid_extractor(train_loader, device, steps)
    torch.set_rng_state(rng_state)

    @torch.no_grad()
    def feats(images, batch: int = 256):
        out = []
        for i in range(0, len(images), batch):
            out.append(extractor.features(images[i:i + batch].to(device)).cpu().numpy())
        return np.concatenate(out, axis=0)

    fr, ff = feats(real_images), feats(fake_images)
    mu1, mu2 = fr.mean(axis=0), ff.mean(axis=0)
    s1, s2 = np.cov(fr, rowvar=False), np.cov(ff, rowvar=False)
    diff = mu1 - mu2
    # Tr(sqrtm(s1 @ s2)) = sum of sqrt of eigenvalues of (s1 @ s2)  (PSD => real, >=0)
    eig = np.linalg.eigvals(s1 @ s2).real
    cov_term = np.sqrt(np.clip(eig, 0, None)).sum()
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2.0 * cov_term)


def get_real_images(loader, n: int = 1024):
    """Collect up to n real images from a dataloader (for FID reference statistics)."""
    imgs, count = [], 0
    for x, _ in loader:
        imgs.append(x)
        count += x.size(0)
        if count >= n:
            break
    return torch.cat(imgs, dim=0)[:n]


# --------------------------------------------------------------------------- #
# Visualizations
# --------------------------------------------------------------------------- #
def save_grid_png(images_tensor, filename: str, nrows: int = 8, ncols: int = 8):
    """Helper to convert generated [-1, 1] torch tensors into standard grid files."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping grid plot: {e})")
        return

    # Convert from [-1, 1] to [0, 1] for plotting
    images = (images_tensor.detach().cpu().numpy() + 1.0) / 2.0
    images = np.clip(images, 0.0, 1.0)

    n_images = min(nrows * ncols, len(images))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.2))

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j] if nrows > 1 else axes[j]
            idx = i * ncols + j
            if idx < n_images:
                ax.imshow(images[idx, 0], cmap="gray")
            ax.axis("off")

    fig.tight_layout(pad=0.2)
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"Saved generated samples grid -> {filename}")


@torch.no_grad()
def save_latent_walk_png(generator, filename: str, latent_dim: int, device=None, num_steps: int = 10):
    """Saves a grid showing a smooth transition (slerp) between two random latent codes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping walk plot: {e})")
        return

    device = device or get_device()
    if hasattr(generator, "to"):
        generator.to(device)
    if hasattr(generator, "eval"):
        generator.eval()

    # Sample two random codes
    z_start = torch.randn(1, latent_dim, device=device)
    z_end = torch.randn(1, latent_dim, device=device)

    interpolated_zs = []
    for step in range(num_steps):
        t = step / (num_steps - 1)
        z_interp = slerp(t, z_start, z_end)
        interpolated_zs.append(z_interp)

    # Stack: [num_steps, latent_dim]
    z_batch = torch.cat(interpolated_zs, dim=0)

    # Generate images
    # Supports both standard generator(z) and cgan which needs class labels
    # CGAN will be passed a wrapper if needed, but standard generator takes z directly
    try:
        gen_imgs = generator(z_batch)
    except TypeError:
        # cgans might expect label. If so, generate class 0 for all
        labels = torch.zeros(num_steps, dtype=torch.long, device=device)
        gen_imgs = generator(z_batch, labels)

    gen_imgs = (gen_imgs.detach().cpu().numpy() + 1.0) / 2.0
    gen_imgs = np.clip(gen_imgs, 0.0, 1.0)

    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 1.2, 1.5))
    for i in range(num_steps):
        axes[i].imshow(gen_imgs[i, 0], cmap="gray")
        axes[i].axis("off")

    fig.tight_layout(pad=0.2)
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"Saved latent space walk -> {filename}")
