"""
lm_common.py
============

Shared utilities for the Language Modeling and text generation demos in this folder.
Handles data downloading, tokenization, sequence encoding, training, and text generation.
"""

from __future__ import annotations

import os
import ssl
import urllib.request
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Raw download parameters
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# --------------------------------------------------------------------------- #
# Data Download & Fallbacks
# --------------------------------------------------------------------------- #
def _download_file(url: str, dest_path: str):
    """Download a file with SSL bypass context to prevent certificate issues on macOS."""
    if os.path.exists(dest_path):
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


def _generate_synthetic_shakespeare() -> str:
    """Generate basic synthetic Elizabethan text if offline."""
    return """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved, resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.
"""


def load_shakespeare(data_dir: str = _DATA_DIR) -> str:
    """Load Shakespeare text as a single string."""
    os.makedirs(data_dir, exist_ok=True)
    dest_path = os.path.join(data_dir, "shakespeare.txt")

    try:
        _download_file(SHAKESPEARE_URL, dest_path)
        with open(dest_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Shakespeare download failed: {e}. Falling back to synthetic text...")
        text = _generate_synthetic_shakespeare()
    return text


# --------------------------------------------------------------------------- #
# Tokenizers & Encoders
# --------------------------------------------------------------------------- #
class CharTokenizer:
    """Character-level tokenizer for text generation."""
    def __init__(self):
        self.chars = []
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0

    def fit(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def decode(self, indices: list[int]) -> str:
        return "".join([self.idx2char[i] for i in indices])


# --------------------------------------------------------------------------- #
# Dataloader Builders
# --------------------------------------------------------------------------- #
def get_shakespeare_dataloaders(seq_len: int = 64, batch_size: int = 64, limit: int | None = None):
    """Load Shakespeare text, build char tokenizer, and slice sequences for training loaders."""
    text = load_shakespeare()
    if limit is not None:
        text = text[:limit]

    tokenizer = CharTokenizer()
    tokenizer.fit(text)

    encoded = tokenizer.encode(text)
    # Generate slices of length seq_len + 1 (inputs of seq_len, targets shifted by 1)
    X_list, y_list = [], []
    for i in range(len(encoded) - seq_len):
        X_list.append(encoded[i : i + seq_len])
        y_list.append(encoded[i + 1 : i + seq_len + 1]) # next step targets for autoregressive loss

    X = np.array(X_list, dtype=np.int64)
    y = np.array(y_list, dtype=np.int64)

    # Split train / test 90% / 10%
    split = int(len(X) * 0.9)
    train_ds = TensorDataset(torch.from_numpy(X[:split]), torch.from_numpy(y[:split]))
    test_ds = TensorDataset(torch.from_numpy(X[split:]), torch.from_numpy(y[split:]))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, tokenizer


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
    p.add_argument("--limit", type=int, default=None, help="limit text character count for smoke tests")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--variant", type=str, default=None)
    return p


def train_language_model(model, train_loader, test_loader, *, epochs: int = 5, lr: float = 1e-3, device=None):
    """Standard trainer for Autoregressive Causal Language Modeling."""
    device = device or get_device()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_tokens = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x) # out shape: [Batch, Seq_Len, Vocab_Size]
            # Flatten for cross entropy
            loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.numel()
            total_tokens += y.numel()

        train_loss = running_loss / total_tokens
        perplexity = np.exp(train_loss)

        # Val evaluation
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
                val_loss += loss.item() * y.numel()
                val_tokens += y.numel()
        val_train_loss = val_loss / val_tokens
        val_perplexity = np.exp(val_train_loss)

        print(f"Epoch {epoch:2d}/{epochs} | loss {train_loss:.4f} | train_ppl {perplexity:.2f} | test_ppl {val_perplexity:.2f}")

    print("-" * 64)
    return model


# --------------------------------------------------------------------------- #
# Text Generation Loops
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_text(model, start_str: str, tokenizer, gen_len: int = 150, temperature: float = 0.8, device=None):
    """Generate autoregressive text characters using the trained model."""
    device = device or get_device()
    model.to(device)
    model.eval()

    context = tokenizer.encode(start_str)
    # The models are trained on sequence lengths (like 64). We keep a sliding context.
    W = 64 # window size expected by model

    print(f"\n--- GENERATING TEXT (Seed: '{start_str}') ---")
    generated = list(context)

    for _ in range(gen_len):
        # Slice the last W elements of generated sequence as input
        input_seq = generated[-W:]
        if len(input_seq) < W:
            # Pad on left if seed is shorter than W
            input_seq = [0] * (W - len(input_seq)) + input_seq

        x = torch.tensor([input_seq], dtype=torch.long, device=device) # [1, W]
        out = model(x) # [1, W, Vocab_Size]
        logits = out[0, -1, :] / temperature # Last sequence step logits
        probs = torch.softmax(logits, dim=-1)

        # Sample from distribution
        next_char_idx = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_char_idx)

    result_text = tokenizer.decode(generated)
    print(result_text)
    print("-------------------------------------------\n")
    return result_text
