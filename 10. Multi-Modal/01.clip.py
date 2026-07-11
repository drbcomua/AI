"""
01. Contrastive Language-Image Pretraining (CLIP)
==================================================

Symmetric cross-entropy contrastive alignment of separate text and image encoders (Radford et al., OpenAI, 2021).

Architecture Diagram / Layout:
    Image [B x 3 x 64 x 64] -> ImageEncoder (CNN) -> L2 Normalized Embed [B x 64]
                                                                  |
                                                                  +---> Cosine Similarity Matrix [B x B]
                                                                  |
    Text [B x SeqLen] -> TextEncoder (GRU) -> L2 Normalized Embed [B x 64]

Key insights / educational takeaways:
    * Map completely different visual and textual representations into a single shared metric hypersphere.
    * Use a symmetric InfoNCE loss matrix to pull matching pairs together while pushing all other pairings apart.

Run:
    python "01.clip.py" --epochs 10
"""

import os
import numpy as np
import torch
import torch.nn as nn
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
    def __init__(self, vocab_size: int = 12, embed_dim: int = 32, hidden_dim: int = 64, embedding_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        emb = self.embedding(x)
        _, hn = self.gru(emb) # hn shape: [1, B, hidden_dim]
        return self.fc(hn[0])


class CLIP(nn.Module):
    """Contrastive Language-Image Pretraining alignment network."""
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.image_encoder = ImageEncoder(embedding_dim)
        self.text_encoder = TextEncoder(vocab_size=12, embedding_dim=embedding_dim)
        # Learnable temperature parameter (initialized to log(1/0.07))
        self.t_prime = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def forward(self, image, text):
        img_emb = self.image_encoder(image)
        text_emb = self.text_encoder(text)

        # L2 normalize embeddings
        img_emb = nn.functional.normalize(img_emb, p=2, dim=1)
        text_emb = nn.functional.normalize(text_emb, p=2, dim=1)

        # Compute cosine similarity matrix
        temperature = torch.exp(self.t_prime)
        logits = torch.matmul(img_emb, text_emb.T) * temperature
        return logits


def main():
    p = mc.build_argparser("CLIP Vision-Language Contrastive Alignment", epochs=10, batch_size=16)
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

    model = CLIP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("Training CLIP Model...")
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

            # Targets are the diagonal indices [0, 1, ..., B-1]
            targets = torch.arange(img.size(0), device=device)

            # Symmetric InfoNCE loss
            loss_i = criterion(logits, targets)
            loss_t = criterion(logits.T, targets)
            loss = (loss_i + loss_t) / 2.0

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
        # Define the 16 unique possible classes
        colors_list = ["red", "green", "blue", "yellow"]
        shapes_list = ["circle", "square", "triangle", "cross"]
        unique_caps = [f"a {c} {s}" for c in colors_list for s in shapes_list]
        unique_toks = mc.tokenize_captions(unique_caps).to(device)

        # Process a validation set of 100 samples
        val_img = test_img[:100].to(device)
        val_captions = test_captions[:100]

        # Compute cosine similarity between 100 test images and 16 unique captions
        img_emb = model.image_encoder(val_img)
        text_emb = model.text_encoder(unique_toks)

        img_emb = nn.functional.normalize(img_emb, p=2, dim=1)
        text_emb = nn.functional.normalize(text_emb, p=2, dim=1)

        # Cosine similarity matrix [100, 16]
        logits = torch.matmul(img_emb, text_emb.T)

        # Prediction is the index of highest similarity description out of 16
        preds = logits.argmax(dim=1).cpu().numpy()

        # Map target captions to their indices in unique_caps
        cap_to_idx = {cap: idx for idx, cap in enumerate(unique_caps)}
        targets = np.array([cap_to_idx[cap] for cap in val_captions])

        acc = np.mean(preds == targets)

    print(f"Test Zero-shot Classification Accuracy (1-of-16): {acc * 100:.2f}%")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        sim_path = os.path.join(save_dir, "clip_similarity_matrix.png")

        # To show a beautiful square matrix with a bright diagonal representing exact matches:
        # Find 8 images with unique labels from the test set
        unique_indices = []
        seen = set()
        for idx, label in enumerate(test_captions):
            if label not in seen:
                seen.add(label)
                unique_indices.append(idx)
            if len(unique_indices) == 8:
                break

        # Slices these 8 matching images and tokens
        sub_img = test_img[unique_indices].to(device)
        sub_tok = test_tok[unique_indices].to(device)
        sub_captions = [test_captions[i] for i in unique_indices]

        with torch.no_grad():
            img_emb = model.image_encoder(sub_img)
            text_emb = model.text_encoder(sub_tok)
            img_emb = nn.functional.normalize(img_emb, p=2, dim=1)
            text_emb = nn.functional.normalize(text_emb, p=2, dim=1)
            # Compute square cosine similarity [8, 8]
            cosine_sim = torch.matmul(img_emb, text_emb.T)

        mc.plot_similarity_matrix(cosine_sim.cpu().numpy(), sub_captions, sim_path)


if __name__ == "__main__":
    main()
