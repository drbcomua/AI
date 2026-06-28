"""
mnist_common.py
===============

Shared utilities for the MNIST architecture demos in this folder.

It deliberately avoids `torchvision` (which is not installed here) and instead
downloads the raw IDX files from the same public mirror torchvision uses,
parses them with NumPy, and exposes:

    * load_mnist(...)      -> raw uint8 NumPy arrays
    * get_dataloaders(...) -> ready-to-train PyTorch DataLoaders
    * get_device(...)      -> pick cuda / mps / cpu
    * train(...)           -> training loop printing per-epoch accuracy
    * evaluate(...)        -> collect y_true / y_pred / y_prob on a loader
    * report(...)          -> research-grade test metrics + confusion matrix

Every architecture script imports these helpers so each file can focus on the
*model* rather than on boilerplate.
"""

from __future__ import annotations

import gzip
import os
import ssl
import urllib.request

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# MNIST global pixel statistics (the canonical values used everywhere).
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _download(fname: str, data_dir: str) -> str:
    """Download a single IDX .gz file if not already cached. Returns its path."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, fname)
    if os.path.exists(path):
        return path

    url = _MIRROR + fname
    # macOS python.org builds frequently lack a usable CA bundle, which breaks
    # certificate verification.  Fall back to an unverified context so the demo
    # works out of the box (data integrity is not security-sensitive here).
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


def load_mnist(data_dir: str = _DATA_DIR):
    """Return (X_train, y_train, X_test, y_test) as uint8 NumPy arrays.

    Images have shape (N, 28, 28); labels have shape (N,).
    """
    paths = {k: _download(v, data_dir) for k, v in _FILES.items()}
    X_train = _read_idx_images(paths["train_images"])
    y_train = _read_idx_labels(paths["train_labels"])
    X_test = _read_idx_images(paths["test_images"])
    y_test = _read_idx_labels(paths["test_labels"])
    return X_train, y_train, X_test, y_test


def get_dataloaders(batch_size: int = 128, limit: int | None = None,
                    data_dir: str = _DATA_DIR, num_workers: int = 0):
    """Build normalized train/test DataLoaders.

    ``limit`` keeps only the first N training and N//6 test samples, which is
    handy for a quick smoke test (``--limit 2000``).
    """
    X_train, y_train, X_test, y_test = load_mnist(data_dir)

    if limit is not None:
        X_train, y_train = X_train[:limit], y_train[:limit]
        n_test = max(1, limit // 6)
        X_test, y_test = X_test[:n_test], y_test[:n_test]

    def to_tensor(X, y):
        X = torch.from_numpy(X.astype(np.float32) / 255.0).unsqueeze(1)  # (N,1,28,28)
        X = (X - MNIST_MEAN) / MNIST_STD
        y = torch.from_numpy(y.astype(np.int64))
        return TensorDataset(X, y)

    train_ds = to_tensor(X_train, y_train)
    test_ds = to_tensor(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader


# --------------------------------------------------------------------------- #
# Device / training / evaluation
# --------------------------------------------------------------------------- #
def build_argparser(description: str, epochs: int = 5, batch_size: int = 128,
                    lr: float = 1e-3):
    """A consistent CLI shared by every architecture script."""
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--batch-size", type=int, default=batch_size)
    p.add_argument("--lr", type=float, default=lr)
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N training samples (quick smoke test)")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--no-figure", action="store_true",
                   help="do not save the confusion-matrix PNG")
    return p


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def train(model, train_loader, test_loader, *, epochs: int = 5, lr: float = 1e-3,
          device=None, criterion=None, optimizer=None):
    """Standard training loop. Prints epoch / loss / train-acc / test-acc.

    ``criterion`` defaults to cross-entropy and assumes ``model(x)`` returns
    class logits (CapsNet passes its own margin loss; its "logits" are the
    capsule lengths, which argmax-decode identically).
    """
    device = device or get_device()
    model.to(device)
    criterion = criterion or nn.CrossEntropyLoss()
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        test_acc = _accuracy(model, test_loader, device)
        print(f"Epoch {epoch:2d}/{epochs} | loss {train_loss:.4f} | "
              f"train_acc {train_acc:.4f} | test_acc {test_acc:.4f}")
    print("-" * 64)
    return model


@torch.no_grad()
def evaluate(model, loader, device=None):
    """Run the model over ``loader`` and return (y_true, y_pred, y_prob)."""
    device = device or get_device()
    model.to(device)
    model.eval()
    ys, preds, probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        p = torch.softmax(out, dim=1)
        ys.append(y.numpy())
        preds.append(p.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)


# --------------------------------------------------------------------------- #
# Research-grade reporting
# --------------------------------------------------------------------------- #
def _wilson_interval(correct: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return center - half, center + half


def report(y_true, y_pred, y_prob=None, *, class_names=None, model_name="model",
           save_dir=None, chance_level: float = 0.1):
    """Print a research-oriented test report and save the confusion matrix PNG."""
    from sklearn.metrics import (classification_report, cohen_kappa_score,
                                 confusion_matrix, log_loss,
                                 matthews_corrcoef,
                                 precision_recall_fscore_support)
    from scipy.stats import binomtest

    if class_names is None:
        class_names = [str(i) for i in range(10)]

    n = len(y_true)
    correct = int((y_true == y_pred).sum())
    acc = correct / n
    lo, hi = _wilson_interval(correct, n)

    # p-value: is accuracy better than blind guessing (10% for 10 classes)?
    pval = binomtest(correct, n, p=chance_level, alternative="greater").pvalue

    pr_m, rc_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    pr_w, rc_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"\n================  TEST REPORT: {model_name}  ================")
    print(f"Samples              : {n}")
    print(f"Accuracy             : {acc:.4f}  (95% Wilson CI [{lo:.4f}, {hi:.4f}])")
    print(f"Error rate           : {1 - acc:.4f}")
    print(f"p-value vs chance     : {pval:.3e}  "
          f"(H0: acc <= {chance_level:.2f}, binomial one-sided)")
    print(f"F1  (macro / weighted): {f1_m:.4f} / {f1_w:.4f}")
    print(f"Precision (macro/wtd) : {pr_m:.4f} / {pr_w:.4f}")
    print(f"Recall    (macro/wtd) : {rc_m:.4f} / {rc_w:.4f}")
    print(f"Cohen's kappa        : {kappa:.4f}")
    print(f"Matthews corrcoef    : {mcc:.4f}")

    if y_prob is not None:
        ll = log_loss(y_true, y_prob, labels=list(range(len(class_names))))
        top2 = np.mean([yt in np.argsort(p)[-2:] for yt, p in zip(y_true, y_prob)])
        print(f"Log loss (test)      : {ll:.4f}")
        print(f"Top-2 accuracy       : {top2:.4f}")

    print("\nPer-class report:")
    # Pass explicit labels so the report stays aligned with class_names even when
    # a quick/small run doesn't predict (or contain) every class.
    print(classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                target_names=class_names, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows = true, cols = predicted):")
    header = "      " + " ".join(f"{c:>5}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>4}  " + " ".join(f"{v:>5}" for v in row))

    if save_dir is not None:
        _save_confusion_png(cm, class_names, model_name, save_dir)
    print("=" * (len(model_name) + 44) + "\n")
    return cm


def _save_confusion_png(cm, class_names, model_name, save_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping confusion-matrix figure: {e})")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = os.path.join(save_dir, f"confusion_{model_name}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved confusion matrix figure -> {out}")
