"""
05. VisualBERT: Single-Stream Vision-Language Fusion Transformer (Li et al., 2019)
=====================================================================================

A single shared Transformer encoder consumes a concatenated sequence of [CLS] + visual
grid tokens + text tokens, letting image and text positions attend to each other freely
inside ordinary self-attention layers. Pretrained jointly with Image-Text Matching (ITM)
and Masked Language Modeling (MLM), the two classic BERT-style objectives.

Architecture Diagram / Layout:
    Image [B x 3 x 64 x 64] -> ConvGridBackbone -> Visual Tokens [B x 16 x 64]
                                                            |
    [CLS] (learnable) ---------------------------+         |
                                                   v         v
                                  Concat -> [B x (1+16+SeqLen) x 64] (+ segment & position embeddings)
                                                   |
                                       Shared Self-Attention Transformer (single stream)
                                                   |
                                +------------------+------------------+
                                v                                     v
                     [CLS] output -> ITM head (match logit)   Text outputs -> MLM head (vocab logits)

Key insights / educational takeaways:
    * "Single-stream": there is exactly one self-attention stack, and visual/text tokens
      are fused as early as the first layer -- contrast with the dual-tower design of
      `01.clip.py`/`04.siglip.py`, where the two modalities never interact until the final
      cosine similarity.
    * Because fusion happens inside attention, the model cannot precompute independent
      image/text embeddings: retrieval requires a fresh joint forward pass per candidate
      caption (see `mc.evaluate_itm_retrieval`), an O(N x C) cost that dual encoders avoid.
    * MLM forces the model to resolve a masked word ("a red ____") using cues from both the
      surrounding text *and* the attended visual tokens, directly exercising cross-modal fusion.

Run:
    python "05.visualbert.py" --epochs 10
    python "05.visualbert.py" --limit 2000        # fast smoke test
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import multimodal_common as mc


class ConvGridBackbone(nn.Module):
    """CNN backbone producing a spatial grid of visual tokens (instead of a single pooled vector)."""
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),   # -> 32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # -> 16x16
            nn.Conv2d(32, d_model, kernel_size=3, padding=1), nn.BatchNorm2d(d_model), nn.ReLU(), nn.MaxPool2d(2), # -> 8x8
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1), nn.BatchNorm2d(d_model), nn.ReLU(), nn.MaxPool2d(2), # -> 4x4
        )

    def forward(self, x):
        h = self.conv(x)             # [B, d_model, 4, 4]
        B, C, H, W = h.shape
        return h.reshape(B, C, H * W).permute(0, 2, 1) # [B, 16, d_model] grid visual tokens


class VisualBERT(nn.Module):
    """Single-stream fused Transformer for joint Image-Text Matching + Masked Language Modeling."""
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 n_visual_tokens: int = 16, seq_len: int = 6):
        super().__init__()
        self.n_visual_tokens = n_visual_tokens
        self.seq_len = seq_len
        total_len = 1 + n_visual_tokens + seq_len # [CLS] + visual grid + text

        self.backbone = ConvGridBackbone(d_model)
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.segment_embedding = nn.Embedding(2, d_model) # 0 = visual, 1 = text
        self.position_embedding = nn.Embedding(total_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
                                                     batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.itm_head = nn.Linear(d_model, 1)
        self.mlm_head = nn.Linear(d_model, vocab_size)

        segment_ids = torch.cat([torch.zeros(1 + n_visual_tokens, dtype=torch.long),
                                  torch.ones(seq_len, dtype=torch.long)])
        self.register_buffer("segment_ids", segment_ids)
        self.register_buffer("position_ids", torch.arange(total_len, dtype=torch.long))

    def forward(self, image, tokens, return_mlm: bool = False):
        B = image.size(0)
        visual_tokens = self.backbone(image)            # [B, 16, d_model]
        text_tokens = self.text_embedding(tokens)        # [B, SeqLen, d_model]
        cls = self.cls_token.expand(B, -1, -1)            # [B, 1, d_model]

        seq = torch.cat([cls, visual_tokens, text_tokens], dim=1)
        seq = seq + self.segment_embedding(self.segment_ids) + self.position_embedding(self.position_ids)

        out = self.encoder(seq) # [B, total_len, d_model] -- visual & text tokens attend to each other freely
        cls_out = out[:, 0]
        itm_logit = self.itm_head(cls_out).squeeze(-1)

        if return_mlm:
            text_out = out[:, 1 + self.n_visual_tokens:]
            mlm_logits = self.mlm_head(text_out)
            return itm_logit, mlm_logits
        return itm_logit


def mask_tokens(tok: torch.Tensor, mlm_prob: float = 0.35):
    """Randomly masks content tokens (not <pad>/<s>/</s>) for the MLM objective.

    Returns (masked_tok, mlm_labels) where mlm_labels is -100 at non-masked positions
    so they are ignored by CrossEntropyLoss.
    """
    special_ids = {mc.VOCAB["<pad>"], mc.VOCAB["<s>"], mc.VOCAB["</s>"]}
    masked = tok.clone()
    labels = torch.full_like(tok, -100)

    for b in range(tok.size(0)):
        eligible = [t for t in range(tok.size(1)) if tok[b, t].item() not in special_ids]
        chosen = [t for t in eligible if random.random() < mlm_prob]
        if not chosen and eligible:
            chosen = [random.choice(eligible)]
        for t in chosen:
            labels[b, t] = tok[b, t]
            masked[b, t] = mc.VOCAB["<mask>"]

    return masked, labels


def main():
    p = mc.build_argparser("VisualBERT Single-Stream Fusion Transformer", epochs=10, batch_size=32)
    args = p.parse_args()

    device = mc.get_device(args.device)

    images, captions = mc.generate_shapes_dataset(num_samples=args.limit or 1200)
    tokens = mc.tokenize_captions(captions)

    split_idx = int(len(images) * 0.8)
    train_img, train_tok = images[:split_idx], tokens[:split_idx]
    test_img, test_tok = images[split_idx:], tokens[split_idx:]
    test_captions = captions[split_idx:]

    model = VisualBERT(vocab_size=len(mc.VOCAB)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion_itm = nn.BCEWithLogitsLoss()
    criterion_mlm = nn.CrossEntropyLoss(ignore_index=-100)

    print("Training VisualBERT Model...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    train_dataset = TensorDataset(train_img, train_tok)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        itm_loss_sum, mlm_loss_sum, total = 0.0, 0.0, 0

        for img, tok in train_loader:
            optimizer.zero_grad()

            # Image-Text Matching: positive pairs + batch-rolled negatives
            combined_img, combined_tok, itm_labels = mc.build_itm_pairs(img, tok)
            combined_img, combined_tok, itm_labels = combined_img.to(device), combined_tok.to(device), itm_labels.to(device)
            itm_logits = model(combined_img, combined_tok)
            loss_itm = criterion_itm(itm_logits, itm_labels)

            # Masked Language Modeling: only on genuinely matching (positive) pairs
            masked_tok, mlm_labels = mask_tokens(tok)
            img_d, masked_tok, mlm_labels = img.to(device), masked_tok.to(device), mlm_labels.to(device)
            _, mlm_logits = model(img_d, masked_tok, return_mlm=True)
            loss_mlm = criterion_mlm(mlm_logits.reshape(-1, mlm_logits.size(-1)), mlm_labels.reshape(-1))

            loss = loss_itm + loss_mlm
            loss.backward()
            optimizer.step()

            B = img.size(0)
            itm_loss_sum += loss_itm.item() * B
            mlm_loss_sum += loss_mlm.item() * B
            total += B

        print(f"Epoch {epoch:2d}/{args.epochs} | itm_loss: {itm_loss_sum / total:.4f} | mlm_loss: {mlm_loss_sum / total:.4f}")

    print("-" * 64)

    # Zero-shot retrieval evaluation (requires a joint forward pass per candidate caption)
    model.eval()

    def score_fn(img_batch, tok_batch):
        return model(img_batch, tok_batch)

    acc, all_scores, targets, unique_caps = mc.evaluate_itm_retrieval(score_fn, test_img, test_captions, device)
    print(f"Test Zero-shot Image-Text Matching Accuracy (1-of-16): {acc * 100:.2f}%")

    # Qualitative MLM cloze predictions
    print("\nSample Masked Language Modeling predictions:")
    sample_img, sample_tok, sample_caps = test_img[:6], test_tok[:6], test_captions[:6]
    masked_sample, mlm_sample_labels = mask_tokens(sample_tok)
    with torch.no_grad():
        _, mlm_logits = model(sample_img.to(device), masked_sample.to(device), return_mlm=True)
    mlm_preds = mlm_logits.argmax(dim=-1).cpu()

    for i in range(len(sample_caps)):
        filled = masked_sample[i].clone()
        for t in range(filled.size(0)):
            if mlm_sample_labels[i, t].item() != -100:
                filled[t] = mlm_preds[i, t]
        masked_str = mc.detokenize_caption(masked_sample[i])
        filled_str = mc.detokenize_caption(filled)
        print(f"Masked: {masked_str:<18} | Filled: {filled_str:<18} | GT: {sample_caps[i]}")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        sim_path = os.path.join(save_dir, "visualbert_matching_matrix.png")

        unique_indices = []
        seen = set()
        for idx, t in enumerate(targets):
            if t not in seen:
                seen.add(t)
                unique_indices.append(idx)
            if len(unique_indices) == 8:
                break

        col_indices = [targets[i] for i in unique_indices]
        sub_scores = all_scores[np.ix_(unique_indices, col_indices)]
        sub_labels = [unique_caps[c] for c in col_indices]

        mc.plot_similarity_matrix(sub_scores, sub_labels, sim_path,
                                   title="VisualBERT Single-Stream Image-Text Matching Matrix")


if __name__ == "__main__":
    main()
