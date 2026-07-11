"""
multimodal_common.py
====================

Shared utilities for Multi-Modal Learning (CLIP & Image Captioning).
Handles synthetic colored shapes dataset generation, tokenization vocabularies,
device configurations, similarity matrix visualization, and caption grids.
"""

from __future__ import annotations

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn

# Vocab Definition: 18 tokens (12 base caption tokens + 5 VQA question words + 1 MLM mask)
VOCAB = {
    "<pad>": 0,
    "<s>": 1,
    "</s>": 2,
    "a": 3,
    "red": 4,
    "green": 5,
    "blue": 6,
    "yellow": 7,
    "circle": 8,
    "square": 9,
    "triangle": 10,
    "cross": 11,
    "what": 12,
    "is": 13,
    "this": 14,
    "color": 15,
    "shape": 16,
    "<mask>": 17,
}
INV_VOCAB = {v: k for k, v in VOCAB.items()}

# Answer classes for the synthetic VQA task (color or shape attribute of the shape)
ANSWER_LIST = ["red", "green", "blue", "yellow", "circle", "square", "triangle", "cross"]
ANSWER_TO_IDX = {a: i for i, a in enumerate(ANSWER_LIST)}

# Colors mapping
COLORS = {
    "red": [1.0, 0.1, 0.1],
    "green": [0.1, 0.9, 0.1],
    "blue": [0.1, 0.1, 1.0],
    "yellow": [0.9, 0.9, 0.1]
}


# --------------------------------------------------------------------------- #
# Synthetic Multi-Modal Shapes Dataset Generator
# --------------------------------------------------------------------------- #
def generate_shapes_dataset(num_samples: int = 1200):
    """Generates synthetic RGB images containing colored shapes with target captions.

    Images: [num_samples, 3, 64, 64]
    Captions: list of strings (e.g. "a red circle")
    """
    random.seed(42)
    np.random.seed(42)

    shapes_list = ["circle", "square", "triangle", "cross"]
    colors_list = ["red", "green", "blue", "yellow"]

    images = []
    captions = []

    for _ in range(num_samples):
        shape = random.choice(shapes_list)
        color_name = random.choice(colors_list)

        # Create canvas: shape [64, 64, 3], background dark gray
        img = np.ones((64, 64, 3), dtype=np.float32) * 0.15

        color = COLORS[color_name]
        cx, cy = 32, 32
        r = 18

        Y, X = np.ogrid[:64, :64]

        if shape == "circle":
            mask = (X - cx)**2 + (Y - cy)**2 <= r**2
            img[mask] = color
        elif shape == "square":
            mask = (np.abs(X - cx) <= r) & (np.abs(Y - cy) <= r)
            img[mask] = color
        elif shape == "triangle":
            # Simple upward triangle
            mask = (Y >= cy - r) & (Y <= cy + r) & (np.abs(X - cx) <= (cy + r - Y) * 0.9)
            img[mask] = color
        elif shape == "cross":
            # Horizontal and vertical bars
            mask1 = (np.abs(X - cx) <= 5) & (np.abs(Y - cy) <= r)
            mask2 = (np.abs(Y - cy) <= 5) & (np.abs(X - cx) <= r)
            img[mask1 | mask2] = color

        # Transpose to PyTorch channel-first format [3, 64, 64]
        img_t = img.transpose(2, 0, 1)
        images.append(img_t)
        captions.append(f"a {color_name} {shape}")

    return torch.tensor(np.array(images), dtype=torch.float32), captions


# --------------------------------------------------------------------------- #
# Synthetic VQA Question/Answer Generator
# --------------------------------------------------------------------------- #
def generate_vqa_pairs(captions: list[str], seed: int = 42):
    """Builds a synthetic VQA question/answer pair for every "a {color} {shape}" caption.

    Randomly asks either a color or a shape question and returns the matching
    attribute as the answer, e.g. "a red circle" -> ("what color is this", "red").
    """
    rng = random.Random(seed)
    questions, answers = [], []
    for cap in captions:
        _, color, shape = cap.split()
        if rng.random() < 0.5:
            questions.append("what color is this")
            answers.append(color)
        else:
            questions.append("what shape is this")
            answers.append(shape)

    answer_idx = torch.tensor([ANSWER_TO_IDX[a] for a in answers], dtype=torch.long)
    return questions, answers, answer_idx


# --------------------------------------------------------------------------- #
# Simple Word Tokenizer
# --------------------------------------------------------------------------- #
def tokenize_captions(captions: list[str], max_len: int = 6):
    """Converts caption strings into integer token tensors with padding and start/stop tags."""
    token_tensors = []
    for cap in captions:
        words = cap.strip().lower().split()
        tokens = [VOCAB["<s>"]] + [VOCAB[w] for w in words if w in VOCAB] + [VOCAB["</s>"]]

        # Pad or truncate
        if len(tokens) < max_len:
            tokens = tokens + [VOCAB["<pad>"]] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        token_tensors.append(tokens)

    return torch.tensor(token_tensors, dtype=torch.long)


def detokenize_caption(tokens: torch.Tensor | list[int]) -> str:
    """Converts token indices back to string caption."""
    words = []
    for t in tokens:
        if isinstance(t, torch.Tensor):
            t = t.item()
        word = INV_VOCAB[t]
        if word == "</s>":
            break
        if word not in ["<s>", "<pad>"]:
            words.append(word)
    return " ".join(words)


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


# --------------------------------------------------------------------------- #
# Image-Text Matching (ITM) Helpers for Fused (Single/Dual-Stream) Encoders
# --------------------------------------------------------------------------- #
def build_itm_pairs(img: torch.Tensor, tok: torch.Tensor):
    """Builds positive + hard-negative image-text pairs for Image-Text-Matching pretraining.

    Negatives are formed by rolling the caption batch by one position, so every image
    is paired with the caption of a different sample within the same mini-batch.
    """
    B = img.size(0)
    neg_tok = torch.roll(tok, shifts=1, dims=0)
    combined_img = torch.cat([img, img], dim=0)
    combined_tok = torch.cat([tok, neg_tok], dim=0)
    labels = torch.cat([torch.ones(B), torch.zeros(B)], dim=0)
    return combined_img, combined_tok, labels


def evaluate_itm_retrieval(score_fn, test_img: torch.Tensor, test_captions: list[str], device, n_eval: int = 60):
    """Evaluates 1-of-16 zero-shot retrieval accuracy for pairwise image-text matching models.

    Fused encoders (VisualBERT, ViLBERT) couple image and text representations inside
    self/cross-attention layers, so embeddings cannot be precomputed independently like
    CLIP's dual towers -- every candidate caption requires a fresh joint forward pass,
    i.e. O(N x C) instead of O(N + C) at retrieval time.
    """
    colors_list = ["red", "green", "blue", "yellow"]
    shapes_list = ["circle", "square", "triangle", "cross"]
    unique_caps = [f"a {c} {s}" for c in colors_list for s in shapes_list]
    unique_toks = tokenize_captions(unique_caps).to(device)

    n_eval = min(n_eval, len(test_img))
    eval_img = test_img[:n_eval].to(device)
    eval_captions = test_captions[:n_eval]
    cap_to_idx = {cap: idx for idx, cap in enumerate(unique_caps)}
    targets = np.array([cap_to_idx[c] for c in eval_captions])

    all_scores = torch.zeros(n_eval, len(unique_caps))
    with torch.no_grad():
        for j in range(len(unique_caps)):
            tok_batch = unique_toks[j:j + 1].expand(n_eval, -1)
            all_scores[:, j] = score_fn(eval_img, tok_batch).cpu()

    preds = all_scores.argmax(dim=1).numpy()
    acc = float(np.mean(preds == targets))
    return acc, all_scores.numpy(), targets, unique_caps


def build_argparser(description: str, epochs: int = 10, batch_size: int = 64, lr: float = 1e-3):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--batch-size", type=int, default=batch_size)
    p.add_argument("--lr", type=float, default=lr)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--no-figure", action="store_true", help="do not save figures")
    p.add_argument("--limit", type=int, default=None, help="limit dataset samples for quick local checks")
    return p


# --------------------------------------------------------------------------- #
# Visualizations: Similarity Matrix and Caption Grid
# --------------------------------------------------------------------------- #
def plot_similarity_matrix(similarity: np.ndarray, labels: list[str], save_path: str,
                            title: str = "CLIP Vision-Language Alignment Matrix"):
    """Plots and saves an image-text similarity/matching-score heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping similarity plot: {e})")
        return

    # Pick top 8 unique samples to display clearly
    unique_indices = []
    seen = set()
    for idx, label in enumerate(labels):
        if label not in seen:
            seen.add(label)
            unique_indices.append(idx)
        if len(unique_indices) == 8:
            break

    sub_sim = similarity[unique_indices][:, unique_indices]
    sub_labels = [labels[i] for i in unique_indices]

    fig, ax = plt.subplots(figsize=(7, 6.5))
    im = ax.imshow(sub_sim, cmap="magma", aspect="equal")
    fig.colorbar(im, ax=ax, label="Cosine Similarity")

    ax.set_xticks(np.arange(len(sub_labels)))
    ax.set_yticks(np.arange(len(sub_labels)))
    ax.set_xticklabels(sub_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(sub_labels, fontsize=9)

    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved similarity alignment matrix -> {save_path}")


def plot_vqa_grid(images: torch.Tensor, questions: list[str], ground_truth: list[str],
                   predicted: list[str], save_path: str):
    """Saves a grid of test images with their VQA question, ground-truth and predicted answer."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping VQA grid plot: {e})")
        return

    n_display = min(6, len(images))
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.flatten()

    for i in range(n_display):
        img = images[i].numpy().transpose(1, 2, 0)
        img = np.clip(img, 0.0, 1.0)

        axes[i].imshow(img)
        axes[i].axis("off")
        color = "blue" if ground_truth[i] == predicted[i] else "red"
        axes[i].set_title(f"Q: {questions[i]}\nGT: {ground_truth[i]} | Pred: {predicted[i]}", fontsize=9, color=color)

    for j in range(n_display, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved VQA results grid -> {save_path}")


def plot_caption_grid(images: torch.Tensor, ground_truth: list[str], predicted: list[str], save_path: str):
    """Saves a grid of test images with predicted captions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping caption grid plot: {e})")
        return

    # Select 6 images to display
    n_display = min(6, len(images))
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.flatten()

    for i in range(n_display):
        img = images[i].numpy().transpose(1, 2, 0)
        # Clip to [0, 1] for safety
        img = np.clip(img, 0.0, 1.0)

        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"GT: {ground_truth[i]}\nPred: {predicted[i]}", fontsize=9, color="blue" if ground_truth[i] == predicted[i] else "red")

    # Hide remaining empty subplots
    for j in range(n_display, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved generated captions grid -> {save_path}")
