"""
04. Sigmoid Loss for Language-Image Pretraining (SigLIP) (Zhai et al., Google DeepMind, 2023)
================================================================================================

A near-identical dual-encoder architecture to CLIP, but replacing the batch-wide softmax
(InfoNCE) contrastive loss with an independent pairwise sigmoid loss over every entry of
the similarity matrix.

Architecture Diagram / Layout:
    Image [B x 3 x 64 x 64] -> ImageEncoder (CNN) -> L2 Normalized Embed [B x 64]
                                                                  |
                                                                  +---> Similarity Matrix [B x B] * t + b
                                                                  |
    Text [B x SeqLen] -> TextEncoder (GRU) -> L2 Normalized Embed [B x 64]

Key insights / educational takeaways:
    * CLIP's softmax InfoNCE loss normalizes over the *entire row/column* of the similarity
      matrix, which requires synchronizing huge batches across devices for good negatives.
    * SigLIP instead treats every (image, text) cell as an independent binary classification
      problem (match vs. non-match) via a sigmoid, removing the need for a global normalizer.
    * This makes the loss trivially parallelizable and stable even at very small batch sizes
      -- compare this script's behavior with `--batch-size 8` against `01.clip.py` at the same size.
    * An extra learnable bias `b` (initialized very negative) counteracts the large number of
      negatives early in training, since unlike softmax there is no implicit normalization.

Run:
    python "04.siglip.py" --epochs 10
    python "04.siglip.py" --limit 2000        # fast smoke test
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import multimodal_common as mc


class ImageEncoder(nn.Module):
    """Simple 2D CNN extracting visual features."""
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 32x32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 8x8
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        return self.fc(h)


class TextEncoder(nn.Module):
    """Simple GRU text encoder projecting captions into shared dimensions."""
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64, embedding_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        emb = self.embedding(x)
        _, hn = self.gru(emb) # hn shape: [1, B, hidden_dim]
        return self.fc(hn[0])


class SigLIP(nn.Module):
    """Dual-encoder image-text alignment network trained with a pairwise sigmoid loss."""
    def __init__(self, vocab_size: int, embedding_dim: int = 64):
        super().__init__()
        self.image_encoder = ImageEncoder(embedding_dim)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim)
        # Learnable temperature (init log(10)) and bias (init -10), following the SigLIP paper.
        self.t_prime = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(-10.0, dtype=torch.float32))

    def forward(self, image, text):
        img_emb = self.image_encoder(image)
        text_emb = self.text_encoder(text)

        img_emb = nn.functional.normalize(img_emb, p=2, dim=1)
        text_emb = nn.functional.normalize(text_emb, p=2, dim=1)

        temperature = torch.exp(self.t_prime)
        logits = torch.matmul(img_emb, text_emb.T) * temperature + self.bias
        return logits


def siglip_loss(logits: torch.Tensor) -> torch.Tensor:
    """Pairwise sigmoid loss: every cell of the [B x B] matrix is an independent binary label."""
    B = logits.size(0)
    # +1 on the diagonal (matching pairs), -1 everywhere else (non-matching pairs)
    labels = 2 * torch.eye(B, device=logits.device) - 1
    return -torch.mean(F.logsigmoid(labels * logits))


def main():
    p = mc.build_argparser("SigLIP Vision-Language Sigmoid-Loss Alignment", epochs=10, batch_size=16)
    args = p.parse_args()

    device = mc.get_device(args.device)

    # Generate multi-modal dataset
    images, captions = mc.generate_shapes_dataset(num_samples=args.limit or 1200)
    tokens = mc.tokenize_captions(captions)

    # Train / test split (80/20)
    split_idx = int(len(images) * 0.8)
    train_img, train_tok = images[:split_idx], tokens[:split_idx]
    test_img, test_tok = images[split_idx:], tokens[split_idx:]
    test_captions = captions[split_idx:]

    model = SigLIP(vocab_size=len(mc.VOCAB)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print("Training SigLIP Model...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    train_dataset = TensorDataset(train_img, train_tok)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        total = 0

        for img, tok in train_loader:
            img, tok = img.to(device), tok.to(device)

            optimizer.zero_grad()
            logits = model(img, tok)
            loss = siglip_loss(logits)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * img.size(0)
            total += img.size(0)

        epoch_loss_avg = epoch_loss / total
        print(f"Epoch {epoch:2d}/{args.epochs} | loss: {epoch_loss_avg:.4f}")

    print("-" * 64)

    # Evaluate zero-shot classification quality on test set
    model.eval()
    with torch.no_grad():
        colors_list = ["red", "green", "blue", "yellow"]
        shapes_list = ["circle", "square", "triangle", "cross"]
        unique_caps = [f"a {c} {s}" for c in colors_list for s in shapes_list]
        unique_toks = mc.tokenize_captions(unique_caps).to(device)

        val_img = test_img[:100].to(device)
        val_captions = test_captions[:100]

        img_emb = model.image_encoder(val_img)
        text_emb = model.text_encoder(unique_toks)

        img_emb = nn.functional.normalize(img_emb, p=2, dim=1)
        text_emb = nn.functional.normalize(text_emb, p=2, dim=1)

        logits = torch.matmul(img_emb, text_emb.T)
        preds = logits.argmax(dim=1).cpu().numpy()

        cap_to_idx = {cap: idx for idx, cap in enumerate(unique_caps)}
        targets = np.array([cap_to_idx[cap] for cap in val_captions])

        acc = np.mean(preds == targets)

    print(f"Test Zero-shot Classification Accuracy (1-of-16): {acc * 100:.2f}%")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        sim_path = os.path.join(save_dir, "siglip_similarity_matrix.png")

        unique_indices = []
        seen = set()
        for idx, label in enumerate(test_captions):
            if label not in seen:
                seen.add(label)
                unique_indices.append(idx)
            if len(unique_indices) == 8:
                break

        sub_img = test_img[unique_indices].to(device)
        sub_tok = test_tok[unique_indices].to(device)
        sub_captions = [test_captions[i] for i in unique_indices]

        with torch.no_grad():
            img_emb = model.image_encoder(sub_img)
            text_emb = model.text_encoder(sub_tok)
            img_emb = nn.functional.normalize(img_emb, p=2, dim=1)
            text_emb = nn.functional.normalize(text_emb, p=2, dim=1)
            cosine_sim = torch.matmul(img_emb, text_emb.T)

        mc.plot_similarity_matrix(cosine_sim.cpu().numpy(), sub_captions, sim_path,
                                   title="SigLIP Vision-Language Alignment Matrix")


if __name__ == "__main__":
    main()
