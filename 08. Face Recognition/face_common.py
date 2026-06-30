"""
face_common.py
==============

Shared utilities for the Face Recognition metric learning demos.
Handles downloading, parsing, pairing LFW images, standard training wrappers,
ROC curve plotting, and t-SNE embedding visualizations.
"""

from __future__ import annotations

import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_lfw_people

# Default save folder for figures
_FIGURE_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# LFW Dataset Loader & Splitter
# --------------------------------------------------------------------------- #
def load_lfw(min_faces_per_person: int = 5, resize: float = 0.5):
    """Downloads LFW dataset via scikit-learn. Converts to float32 channel-first format."""
    # Globally disable SSL certificate verification to prevent issues on macOS
    import ssl
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass

    print(f"Loading LFW dataset (min_faces_per_person={min_faces_per_person}, resize={resize})...")
    # sklearn fetches LFW locally under ~/scikit_learn_data/
    lfw = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize, color=True)

    # Images shape in sklearn: [N, H, W, 3] -> transpose to PyTorch: [N, 3, H, W]
    images = lfw.images.transpose(0, 3, 1, 2)
    # Normalize to [0, 1]
    images = images.astype(np.float32) / 255.0

    labels = lfw.target
    num_classes = len(np.unique(labels))

    print(f"LFW loaded: {len(images)} images, {num_classes} identities.")
    print(f"Image resolution: {images.shape[2]}x{images.shape[3]} pixels.")

    return torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def split_lfw_identities(images: torch.Tensor, labels: torch.Tensor, train_ratio: float = 0.8):
    """Splits images by unique people identities (Open-Set Evaluation).

    Ensures the test set contains completely unseen faces.
    """
    unique_labels = np.unique(labels.numpy())
    np.random.seed(42)
    np.random.shuffle(unique_labels)

    split_idx = int(len(unique_labels) * train_ratio)
    train_identities = set(unique_labels[:split_idx])
    test_identities = set(unique_labels[split_idx:])

    train_indices = []
    test_indices = []

    for idx, label in enumerate(labels.numpy()):
        if label in train_identities:
            train_indices.append(idx)
        else:
            test_indices.append(idx)

    train_indices = torch.tensor(train_indices, dtype=torch.long)
    test_indices = torch.tensor(test_indices, dtype=torch.long)

    # Remap training labels to 0..num_train_classes-1 (required for ArcFace classification)
    train_labels = labels[train_indices].numpy()
    unique_train_labels = sorted(list(set(train_labels)))
    label_remap = {old: new for new, old in enumerate(unique_train_labels)}
    remapped_train_labels = torch.tensor([label_remap[l] for l in train_labels], dtype=torch.long)

    return (
        images[train_indices], remapped_train_labels,
        images[test_indices], labels[test_indices]
    )


# --------------------------------------------------------------------------- #
# Verification Pairs Generator
# --------------------------------------------------------------------------- #
def generate_verification_pairs(images: torch.Tensor, labels: torch.Tensor, num_pairs: int = 1000):
    """Generates matching pairs (same identity) and non-matching pairs (different identities) for test evaluation."""
    labels_np = labels.numpy()
    unique_labels = np.unique(labels_np)

    # Pre-group indices by labels
    label_to_indices = {l: np.where(labels_np == l)[0] for l in unique_labels}
    # Filter identities with at least 2 photos for positive pairs
    valid_labels_for_positives = [l for l in unique_labels if len(label_to_indices[l]) >= 2]

    if not valid_labels_for_positives:
        raise ValueError("No identities have at least 2 images to generate matching pairs!")

    pairs_1 = []
    pairs_2 = []
    pair_labels = [] # 1 = Same, 0 = Different

    half_pairs = num_pairs // 2

    # 1. Matching Pairs (Positive)
    for _ in range(half_pairs):
        l = random.choice(valid_labels_for_positives)
        idx1, idx2 = np.random.choice(label_to_indices[l], size=2, replace=False)
        pairs_1.append(images[idx1])
        pairs_2.append(images[idx2])
        pair_labels.append(1)

    # 2. Non-matching Pairs (Negative)
    for _ in range(half_pairs):
        l1, l2 = np.random.choice(unique_labels, size=2, replace=False)
        idx1 = random.choice(label_to_indices[l1])
        idx2 = random.choice(label_to_indices[l2])
        pairs_1.append(images[idx1])
        pairs_2.append(images[idx2])
        pair_labels.append(0)

    # Stack to create tensors [num_pairs, 3, H, W]
    pairs_1 = torch.stack(pairs_1)
    pairs_2 = torch.stack(pairs_2)
    pair_labels = torch.tensor(pair_labels, dtype=torch.long)

    return pairs_1, pairs_2, pair_labels


# --------------------------------------------------------------------------- #
# Standard CNN Backbone for Embeddings
# --------------------------------------------------------------------------- #
class FaceEmbeddingNet(nn.Module):
    """A standard deep convolutional network to map face images to L2-normalized embeddings."""
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        # Input size: 62x47 (for resize=0.5)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 31 x 23

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 15 x 11

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 7 x 5
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        emb = self.fc(h)
        # Normalize to hypersphere: embedding distance corresponds to Cosine distance
        return nn.functional.normalize(emb, p=2, dim=1)


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


def build_argparser(description: str, epochs: int = 10, batch_size: int = 64, lr: float = 1e-3):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--batch-size", type=int, default=batch_size)
    p.add_argument("--lr", type=float, default=lr)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--no-figure", action="store_true", help="do not save figures")
    p.add_argument("--variant", type=str, default=None)
    p.add_argument("--limit", type=int, default=None, help="limit dataset samples for quick local checks")
    return p


# --------------------------------------------------------------------------- #
# ROC Curve and t-SNE Cluster Plotting
# --------------------------------------------------------------------------- #
def plot_verification_roc(y_true: np.ndarray, distances: np.ndarray, save_path: str, model_name: str):
    """Plot verification ROC curve, calculate optimal threshold, accuracy, and AUC."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
    except Exception as e:
        print(f"(skipping ROC plot: {e})")
        return

    # In metric learning, y=1 (same) should have small distance; y=0 (different) should have large distance.
    # To compute ROC, we use similarities = -distances
    similarities = -distances
    fpr, tpr, thresholds = roc_curve(y_true, similarities)
    roc_auc = auc(fpr, tpr)

    # Compute optimal accuracy and corresponding distance threshold
    best_acc = 0.0
    best_thresh = 0.0
    for t in thresholds:
        preds = (similarities >= t).astype(int)
        acc = np.mean(preds == y_true)
        if acc > best_acc:
            best_acc = acc
            best_thresh = -t

    print(f"Optimal Verification Threshold: {best_thresh:.4f}")
    print(f"Optimal Verification Accuracy: {best_acc * 100:.2f}%")
    print(f"Verification AUC: {roc_auc:.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"LFW Verification ROC ({model_name})")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved verification ROC plot -> {save_path}")


def plot_tsne_embeddings(embeddings: torch.Tensor, labels: torch.Tensor, save_path: str, title: str):
    """Plot t-SNE clustering of LFW face embeddings, colored by identity."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping embedding cluster plot: {e})")
        return

    print("Computing t-SNE face embedding projection...")
    emb_np = embeddings.detach().cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Filter to top 8 most frequent classes to make the visual chart legible
    unique, counts = np.unique(labels_np, return_counts=True)
    frequent_classes = unique[np.argsort(counts)[-8:]]

    keep_indices = np.isin(labels_np, frequent_classes)
    emb_np = emb_np[keep_indices]
    labels_np = labels_np[keep_indices]

    if len(emb_np) < 5:
        print("(too few samples to plot t-SNE embeddings, skipping)")
        return

    perplexity = min(30, max(5, len(emb_np) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    emb_2d = tsne.fit_transform(emb_np)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    scatter = ax.scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=labels_np, cmap="tab10",
        alpha=0.8, edgecolors="black", linewidths=0.3, s=35
    )
    legend = ax.legend(*scatter.legend_elements(), title="Identities", loc="upper right")
    ax.add_artist(legend)
    ax.set_title(title)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved embedding projection plot -> {save_path}")


# --------------------------------------------------------------------------- #
# Shared evaluation: verification, identification (CMC), and PK batching
# --------------------------------------------------------------------------- #
@torch.no_grad()
def embed_all(model, images, device, batch_size: int = 128):
    """Run the backbone over all images, returning a single embedding tensor."""
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        out.append(model(images[i:i + batch_size].to(device)).cpu())
    return torch.cat(out, dim=0)


def verify_and_report(model, test_img, test_lbl, device, batch_size, model_name,
                      save_dir, no_figure, num_pairs: int = 1000):
    """Verification eval shared by every metric-learning script: builds unseen pairs,
    measures embedding distances, prints AUC, and (unless --no-figure) saves ROC + t-SNE."""
    p1, p2, plbl = generate_verification_pairs(test_img, test_lbl, num_pairs)
    model.eval()
    dists = []
    with torch.no_grad():
        for i in range(0, len(p1), batch_size):
            e1 = model(p1[i:i + batch_size].to(device))
            e2 = model(p2[i:i + batch_size].to(device))
            dists.extend(torch.norm(e1 - e2, p=2, dim=1).cpu().numpy())
    dists = np.array(dists)
    slug = model_name.lower().replace(" ", "_").replace("-", "")
    if no_figure:
        from sklearn.metrics import roc_auc_score
        print(f"{model_name} Verification AUC: {roc_auc_score(plbl.numpy(), -dists):.4f}")
    else:
        plot_verification_roc(plbl.numpy(), dists, os.path.join(save_dir, f"{slug}_roc.png"), model_name)
        emb = embed_all(model, test_img, device, batch_size)
        plot_tsne_embeddings(emb, test_lbl, os.path.join(save_dir, f"{slug}_tsne.png"),
                             f"t-SNE Projection of {model_name} Face Embeddings")


def evaluate_identification(embeddings: torch.Tensor, labels: torch.Tensor, max_rank: int = 10):
    """Closed-set identification: first image per identity is the gallery, the rest are
    probes. Returns the CMC curve (cmc[k-1] = rank-k accuracy)."""
    labels_np = labels.cpu().numpy()
    seen, gallery_idx, probe_idx = set(), [], []
    for i, l in enumerate(labels_np):
        if l not in seen:
            seen.add(l); gallery_idx.append(i)
        else:
            probe_idx.append(i)
    if not probe_idx:
        raise ValueError("No probes: every identity has only one image.")
    g, g_lab = embeddings[gallery_idx], labels_np[gallery_idx]
    p, p_lab = embeddings[probe_idx], labels_np[probe_idx]
    order = torch.cdist(p, g).argsort(dim=1).numpy()             # nearest gallery first
    max_rank = min(max_rank, len(gallery_idx))
    cmc = np.zeros(max_rank)
    for i in range(len(p)):
        ranked = g_lab[order[i]]
        hit = np.where(ranked == p_lab[i])[0]
        if len(hit) and hit[0] < max_rank:
            cmc[hit[0]:] += 1
    return cmc / len(p)


def plot_cmc(cmc, save_path: str, model_name: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping CMC plot: {e})")
        return
    ranks = np.arange(1, len(cmc) + 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(ranks, cmc * 100, "o-", color="seagreen", lw=2)
    ax.set_xlabel("Rank k"); ax.set_ylabel("Identification accuracy (%)")
    ax.set_title(f"CMC Curve ({model_name})")
    ax.set_ylim([0, 101]); ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)
    print(f"Saved CMC curve -> {save_path}")


def iterate_pk_batches(images, labels, p: int, k: int, steps: int):
    """Yield 'P identities x K images' batches (for in-batch metric losses like SupCon)."""
    labels_np = labels.numpy()
    label_to_idx = {l: np.where(labels_np == l)[0] for l in np.unique(labels_np)}
    valid = [l for l in label_to_idx if len(label_to_idx[l]) >= 2]
    for _ in range(steps):
        chosen = np.random.choice(valid, size=min(p, len(valid)), replace=False)
        idxs = []
        for l in chosen:
            pool = label_to_idx[l]
            idxs.extend(np.random.choice(pool, size=k, replace=len(pool) < k))
        idxs = torch.tensor(np.array(idxs), dtype=torch.long)
        yield images[idxs], labels[idxs]
