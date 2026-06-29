"""
tc_common.py
============

Shared utilities for the Text Classification demos in this folder.
Handles data downloading, tokenization, sequence encoding, training, and metrics.
"""

from __future__ import annotations

import os
import re
import ssl
import urllib.request
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Raw download parameters
IMDB_URL = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# --------------------------------------------------------------------------- #
# Data Download & Fallbacks
# --------------------------------------------------------------------------- #
def _download_file(url: str, dest_path: str):
    """Download a file with SSL bypass context to prevent certificate issues on macOS."""
    # Re-download if file is missing or too small (e.g. cached 404 error page)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024 * 1024:
        return dest_path
    print(f"Downloading {url}...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as r:
        blob = r.read()
    with open(dest_path, "wb") as f:
        f.write(blob)
    return dest_path


def _generate_synthetic_imdb(num_samples: int = 1000) -> list[tuple[str, int]]:
    """Generate realistic synthetic IMDB reviews with labels (1=positive, 0=negative)

    for fully offline testing and robust fallbacks.
    """
    pos_phrases = [
        "this movie was absolutely fantastic and great",
        "i loved the acting, characters, and beautiful story",
        "a masterpiece of cinematography and acting brilliance",
        "highly recommended film, wonderful and inspiring",
        "what a great experience, i would watch it again"
    ]
    neg_phrases = [
        "this was a horrible and boring movie",
        "waste of time, bad acting and terrible script",
        "the plot made no sense, worst film of the year",
        "i fell asleep, it was slow and uninteresting",
        "completely trash, do not watch it at all"
    ]

    np.random.seed(42)
    reviews = []
    for _ in range(num_samples):
        label = np.random.randint(0, 2)
        phrase_list = pos_phrases if label == 1 else neg_phrases
        # Combine random phrases
        text = " . ".join(np.random.choice(phrase_list, size=2))
        reviews.append((text, label))
    return reviews


def load_imdb(limit: int | None = None, data_dir: str = _DATA_DIR) -> list[tuple[str, int]]:
    """Load IMDB reviews. Returns list of (text_string, label_int)."""
    os.makedirs(data_dir, exist_ok=True)
    dest_path = os.path.join(data_dir, "imdb_dataset.csv")

    try:
        # Download a clean IMDB dump
        _download_file(IMDB_URL, dest_path)
        # Parse CSV simple way to avoid dependency on pandas
        reviews = []
        import csv
        with open(dest_path, "r", encoding="latin-1") as f:
            reader = csv.reader(f)
            next(reader) # skip header
            # Columns in IMDB Dataset.csv: [review, sentiment]
            for row in reader:
                if len(row) >= 2:
                    text = row[0]
                    label_str = row[1]
                    if label_str in ["positive", "pos"]:
                        reviews.append((text, 1))
                    elif label_str in ["negative", "neg"]:
                        reviews.append((text, 0))
    except Exception as e:
        print(f"IMDB download/parsing failed: {e}. Falling back to synthetic IMDB reviews...")
        reviews = _generate_synthetic_imdb(1000)

    if limit is not None:
        reviews = reviews[:limit]
    return reviews


# --------------------------------------------------------------------------- #
# Tokenizers & Encoders
# --------------------------------------------------------------------------- #
class WordTokenizer:
    """Simple space/alphanumeric word tokenizer for classification."""
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}

    def fit(self, texts: list[str]):
        word_counts = {}
        for t in texts:
            words = self._tokenize(t)
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1

        # Sort words by count
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        # Add to vocab up to vocab_size
        for word, _ in sorted_words:
            if len(self.word2idx) >= self.vocab_size:
                break
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def _tokenize(self, text: str) -> list[str]:
        # Lowercase and split by non-alphanumeric
        text = text.lower()
        return re.findall(r'\b\w+\b', text)

    def encode(self, text: str, max_len: int = 150) -> list[int]:
        words = self._tokenize(text)
        encoded = [self.word2idx.get(w, 1) for w in words] # 1 is <unk>
        # Pad or truncate
        if len(encoded) < max_len:
            encoded += [0] * (max_len - len(encoded)) # 0 is <pad>
        else:
            encoded = encoded[:max_len]
        return encoded


# --------------------------------------------------------------------------- #
# Dataloader Builders
# --------------------------------------------------------------------------- #
def get_imdb_dataloaders(batch_size: int = 64, limit: int | None = None,
                         vocab_size: int = 5000, max_len: int = 150):
    """Load IMDB, build vocabulary, encode reviews, and return dataloaders."""
    reviews = load_imdb(limit)
    texts = [r[0] for r in reviews]
    labels = [r[1] for r in reviews]

    tokenizer = WordTokenizer(vocab_size)
    tokenizer.fit(texts)

    encoded_texts = [tokenizer.encode(t, max_len) for t in texts]

    X = np.array(encoded_texts, dtype=np.int64)
    y = np.array(labels, dtype=np.float32) # float32 for BCEWithLogitsLoss

    # Train / Test split 80% / 20%
    n = len(X)
    split = int(n * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, tokenizer


class CharTokenizer:
    """Character-level tokenizer (for char-CNN). Index 0 is reserved for pad/unknown."""
    ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 .,;:!?'\"-()[]"

    def __init__(self):
        self.char2idx = {c: i + 1 for i, c in enumerate(self.ALPHABET)}
        self.vocab_size = len(self.ALPHABET) + 1            # +1 for the 0 = pad/unk slot

    def encode(self, text: str, max_len: int) -> list[int]:
        enc = [self.char2idx.get(c, 0) for c in text.lower()][:max_len]
        if len(enc) < max_len:
            enc += [0] * (max_len - len(enc))
        return enc


def get_imdb_char_dataloaders(batch_size: int = 64, limit: int | None = None, max_len: int = 400):
    """Character-level IMDB loaders for the char-CNN. Returns (train, test, char_tokenizer)."""
    reviews = load_imdb(limit)
    texts = [r[0] for r in reviews]
    labels = [r[1] for r in reviews]

    tok = CharTokenizer()
    X = np.array([tok.encode(t, max_len) for t in texts], dtype=np.int64)
    y = np.array(labels, dtype=np.float32)

    split = int(len(X) * 0.8)
    train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
    test_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, tok


def get_imdb_hierarchical_dataloaders(batch_size: int = 64, limit: int | None = None,
                                      vocab_size: int = 5000, max_sents: int = 10,
                                      max_sent_len: int = 30):
    """Hierarchical (document -> sentences -> words) IMDB loaders for HAN.

    Each document becomes a [max_sents, max_sent_len] grid of word ids.
    Returns (train, test, word_tokenizer).
    """
    reviews = load_imdb(limit)
    texts = [r[0] for r in reviews]
    labels = [r[1] for r in reviews]

    tok = WordTokenizer(vocab_size)
    tok.fit(texts)

    def encode_doc(text: str) -> list[list[int]]:
        sents = [s for s in re.split(r"[.!?]+", text) if s.strip()][:max_sents]
        doc = [tok.encode(s, max_sent_len) for s in sents]
        while len(doc) < max_sents:                        # pad with empty sentences
            doc.append([0] * max_sent_len)
        return doc

    X = np.array([encode_doc(t) for t in texts], dtype=np.int64)   # (N, max_sents, max_sent_len)
    y = np.array(labels, dtype=np.float32)

    split = int(len(X) * 0.8)
    train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
    test_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, tok


# --------------------------------------------------------------------------- #
# Device, Argparser, and Training
# --------------------------------------------------------------------------- #
def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_argparser(description: str, epochs: int = 5, batch_size: int = 64, lr: float = 1e-3):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--batch-size", type=int, default=batch_size)
    p.add_argument("--lr", type=float, default=lr)
    p.add_argument("--limit", type=int, default=None, help="limit review count for smoke tests")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--variant", type=str, default=None)
    p.add_argument("--no-figure", action="store_true", help="skip saving confusion matrix")
    return p


def train_classifier(model, train_loader, test_loader, *, epochs: int = 5, lr: float = 1e-3, device=None):
    """Standard trainer for Binary Sentiment Classification."""
    device = device or get_device()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x).squeeze(-1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            preds = (torch.sigmoid(out) >= 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Val evaluation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x).squeeze(-1)
                preds = (torch.sigmoid(out) >= 0.5).float()
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch:2d}/{epochs} | loss {train_loss:.4f} | train_acc {train_acc:.4f} | test_acc {val_acc:.4f}")

    print("-" * 64)
    return model


# --------------------------------------------------------------------------- #
# Evaluation Reports
# --------------------------------------------------------------------------- #
def report_classification(y_true, y_pred, model_name="Classifier", save_dir=None):
    """Print classification reports (Precision, Recall, F1, Accuracy) and save confusion matrix."""
    from sklearn.metrics import classification_report, confusion_matrix
    n = len(y_true)
    correct = (y_true == y_pred).sum()
    acc = correct / n

    print(f"\n================  SENTIMENT REPORT: {model_name}  ================")
    print(f"Test samples         : {n}")
    print(f"Accuracy             : {acc:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"], digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(f"Negative: [TN: {cm[0,0]:<5} FP: {cm[0,1]:<5}]")
    print(f"Positive: [FN: {cm[1,0]:<5} TP: {cm[1,1]:<5}]")

    if save_dir is not None:
        _save_confusion_png(cm, ["Negative", "Positive"], model_name, save_dir)
    print("=" * (len(model_name) + 38) + "\n")


def _save_confusion_png(cm, class_names, model_name, save_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping confusion-matrix figure: {e})")
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Purples")
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = os.path.join(save_dir, f"confusion_{model_name}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved confusion matrix figure -> {out}")
