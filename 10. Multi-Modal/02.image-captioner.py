"""
02. Image Captioning Transformer
=================================

An encoder-decoder model combining a CNN visual feature extractor with a causal recurrent decoder.

Architecture Diagram / Layout:
    Input Image [B x 3 x 64 x 64] -> ImageEncoder (CNN) -> Visual Embeddings [B x 64]
                                                                 |
                                                                 v
    Causal Input Tokens [B x SeqLen-1] -> Embedding -> Cat with Visual -> GRU Decoder -> Output Logits [B x SeqLen-1 x 12]

Key insights / educational takeaways:
    * Condition a causal autoregressive language decoder on continuous visual representations.
    * Use teacher forcing during training to allow fast vectorized sequence computations.

Run:
    python "02.image-captioner.py" --epochs 12
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import multimodal_common as mc


class ImageEncoder(nn.Module):
    """Simple 2D CNN mapping images to low-dimensional visual feature vectors."""
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


class ImageCaptioner(nn.Module):
    """Sequence-to-sequence model generating text from images."""
    def __init__(self, vocab_size: int = 12, img_feat_dim: int = 64, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.image_encoder = ImageEncoder(img_feat_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim + img_feat_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.img_feat_dim = img_feat_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

    def forward(self, image, captions):
        # image shape: [B, 3, 64, 64]
        # captions shape: [B, SeqLen - 1] (inputs for teacher forcing)
        B, SeqLen = captions.shape

        # 1. Extract visual features
        img_features = self.image_encoder(image) # [B, img_feat_dim]

        # 2. Get word embeddings
        word_embs = self.embedding(captions) # [B, SeqLen, embed_dim]

        # 3. Repeat image features along time steps and concatenate
        img_features_expanded = img_features.unsqueeze(1).expand(-1, SeqLen, -1) # [B, SeqLen, img_feat_dim]
        x = torch.cat([word_embs, img_features_expanded], dim=-1) # [B, SeqLen, embed_dim + img_feat_dim]

        # 4. Decoder recurrent forward pass
        out, _ = self.gru(x) # [B, SeqLen, hidden_dim]
        logits = self.fc(out) # [B, SeqLen, vocab_size]
        return logits

    def generate_caption(self, image, max_len: int = 6, device: str = "cpu"):
        """Autoregressive decoding loop for generating caption sequence from image feature prefix."""
        self.eval()
        B = image.size(0)

        # Extract vision features
        img_features = self.image_encoder(image) # [B, img_feat_dim]

        # Start with <s> token for each batch sample
        tokens = torch.ones(B, 1, dtype=torch.long, device=device) * mc.VOCAB["<s>"]
        h = None

        generated_seqs = [[] for _ in range(B)]
        finished = np.zeros(B, dtype=bool)

        for _ in range(max_len - 1):
            # Take last generated token
            current_tok = tokens[:, -1:] # [B, 1]
            word_emb = self.embedding(current_tok) # [B, 1, embed_dim]

            # Concatenate with vision embedding
            img_feat_exp = img_features.unsqueeze(1) # [B, 1, img_feat_dim]
            x = torch.cat([word_emb, img_feat_exp], dim=-1) # [B, 1, embed_dim + img_feat_dim]

            out, h = self.gru(x, h) # out shape: [B, 1, hidden_dim]
            logits = self.fc(out[:, 0]) # [B, vocab_size]
            next_toks = logits.argmax(dim=-1) # [B]

            # Append to lists
            tokens = torch.cat([tokens, next_toks.unsqueeze(1)], dim=1)

            for b in range(B):
                if not finished[b]:
                    tok_idx = next_toks[b].item()
                    generated_seqs[b].append(tok_idx)
                    if tok_idx == mc.VOCAB["</s>"]:
                        finished[b] = True

            if finished.all():
                break

        return generated_seqs


def main():
    p = mc.build_argparser("Image Captioning Transformer Decoder", epochs=12)
    args = p.parse_args()

    device = mc.get_device(args.device)

    # Load shapes dataset
    images, captions = mc.generate_shapes_dataset(num_samples=args.limit or 1200)
    tokens = mc.tokenize_captions(captions)

    # Train / test split (80/20)
    split_idx = int(len(images) * 0.8)
    train_img, train_tok = images[:split_idx], tokens[:split_idx]
    test_img, test_tok = images[split_idx:], tokens[split_idx:]
    test_captions = captions[split_idx:]

    model = ImageCaptioner().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("Training Image Captioner Model...")
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

            # Split caption sequence into input (<s> to target-1) and target (target to </s>)
            inputs = tok[:, :-1]
            targets = tok[:, 1:]

            optimizer.zero_grad()
            logits = model(img, inputs)

            # Flatten logits and targets to compute cross entropy loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * img.size(0)
            total += img.size(0)

        epoch_loss_avg = epoch_loss / total
        print(f"Epoch {epoch:2d}/{args.epochs} | loss: {epoch_loss_avg:.4f}")

    print("-" * 64)

    # Autoregressive generation on unseen test images
    print("Generating captions on unseen test images...")
    model.eval()
    val_images = test_img[:6].to(device)
    val_gt = test_captions[:6]

    with torch.no_grad():
        generated_tokens = model.generate_caption(val_images, device=device)

    predicted_captions = [mc.detokenize_caption(tokens) for tokens in generated_tokens]

    print("\nSample Test Generations:")
    for i in range(len(val_gt)):
        print(f"GT: {val_gt[i]:<18} | Pred: {predicted_captions[i]}")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        grid_path = os.path.join(save_dir, "captioner_results.png")
        mc.plot_caption_grid(val_images.cpu(), val_gt, predicted_captions, grid_path)


if __name__ == "__main__":
    main()
