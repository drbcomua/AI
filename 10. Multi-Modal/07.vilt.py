"""
07. ViLT: Vision-and-Language Transformer Without Convolutions (Kim et al., 2021)
====================================================================================

Every fused vision-language model so far in this directory (CLIP, SigLIP, VisualBERT,
ViLBERT) uses a CNN to turn pixels into visual tokens. ViLT removes the CNN entirely:
raw image patches are flattened and linearly projected -- exactly like ViT -- and fed
into the *same* single-stream fused Transformer recipe as `05.visualbert.py`.

Architecture Diagram / Layout:
    Image [B x 3 x 64 x 64] -> split into 8x8 patches -> flatten -> Linear -> Patch Tokens [B x 64 x 64]
                                                                                     |
    [CLS] (learnable) --------------------------------------------------------------+
                                                                                     v
                                  Concat -> [B x (1+64+SeqLen) x 64] (+ segment & position embeddings)
                                                                                     |
                                                         Shared Self-Attention Transformer (single stream)
                                                                                     |
                                                +------------------------------------+------------------+
                                                v                                                       v
                                     [CLS] output -> ITM head (match logit)                 Text outputs -> MLM head

Key insights / educational takeaways:
    * Removing the convolutional (or, in the original paper, a slow pretrained object
      detector's region-feature) backbone makes the visual "tokenizer" a single linear
      layer -- most of ViLT's capacity lives in the shared Transformer, not a bespoke
      vision stem.
    * Compare trainable parameter counts against `05.visualbert.py`: the patch embedding
      here is a single `Linear(192, 64)`, versus VisualBERT's 4-layer CNN -- yet the fused
      Transformer can still learn to extract shape/color structure from raw pixel patches.
    * Same fusion trade-off as VisualBERT applies: because patches and text tokens are
      fused from layer one, retrieval still needs a joint forward pass per candidate
      caption (`mc.evaluate_itm_retrieval`), unlike CLIP's independent dual towers.
    * ViLT is genuinely harder to train than a CNN-backboned fusion model: with coarse
      8x8 patches, Image-Text Matching reliably learns *color* (an easy, spatially
      diffuse cue) but plateaus well below VisualBERT's ~100% because it cannot resolve
      *shape* (a fine-grained boundary cue) from a single linear projection per patch.
      Finer 4x4 patches help the shared self-attention pick up shape structure, but even
      then, zero-shot ITM retrieval accuracy on this tiny dataset stays noticeably
      variable run-to-run (observed roughly 30-90% across seeds/epoch counts) -- a
      small-scale echo of why the original paper needed large-scale pretraining data to
      match region-feature models at all. MLM cloze-filling, by contrast, converges
      reliably and near-perfectly in every run (even correctly filling in *both* the
      color and shape words from the image alone when both are masked) -- a per-token
      argmax over a small closed vocabulary is a far more forgiving readout of a
      still-forming visual representation than the single, precise, 16-way global
      discrimination that Image-Text Matching retrieval demands.

Run:
    python "07.vilt.py" --epochs 20
    python "07.vilt.py" --limit 2000        # fast smoke test
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import multimodal_common as mc


class PatchEmbed(nn.Module):
    """Splits an image into non-overlapping patches and linearly projects each to d_model.

    No convolution, no pretrained detector -- just a reshape and a single Linear layer.
    """
    def __init__(self, img_size: int = 64, patch_size: int = 4, in_ch: int = 3, d_model: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches_side = img_size // patch_size
        self.proj = nn.Linear(patch_size * patch_size * in_ch, d_model)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        # [B, C, H, W] -> [B, n_patches, patch_size*patch_size*C]
        x = x.unfold(2, p, p).unfold(3, p, p)               # [B, C, H/p, W/p, p, p]
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, self.n_patches_side ** 2, -1)
        return self.proj(x)                                  # [B, n_patches, d_model]


class ViLT(nn.Module):
    """Single-stream fused Transformer over raw linear patch tokens + text, trained for ITM + MLM."""
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 img_size: int = 64, patch_size: int = 4, seq_len: int = 6):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        total_len = 1 + n_patches + seq_len

        self.patch_embed = PatchEmbed(img_size, patch_size, 3, d_model)
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.segment_embedding = nn.Embedding(2, d_model) # 0 = visual, 1 = text
        self.position_embedding = nn.Embedding(total_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
                                                     batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.itm_head = nn.Linear(d_model, 1)
        self.mlm_head = nn.Linear(d_model, vocab_size)

        self.n_patches = n_patches
        segment_ids = torch.cat([torch.zeros(1 + n_patches, dtype=torch.long),
                                  torch.ones(seq_len, dtype=torch.long)])
        self.register_buffer("segment_ids", segment_ids)
        self.register_buffer("position_ids", torch.arange(total_len, dtype=torch.long))

    def forward(self, image, tokens, return_mlm: bool = False):
        B = image.size(0)
        patch_tokens = self.patch_embed(image)           # [B, n_patches, d_model]
        text_tokens = self.text_embedding(tokens)         # [B, SeqLen, d_model]
        cls = self.cls_token.expand(B, -1, -1)

        seq = torch.cat([cls, patch_tokens, text_tokens], dim=1)
        seq = seq + self.segment_embedding(self.segment_ids) + self.position_embedding(self.position_ids)

        out = self.encoder(seq)
        cls_out = out[:, 0]
        itm_logit = self.itm_head(cls_out).squeeze(-1)

        if return_mlm:
            text_out = out[:, 1 + self.n_patches:]
            mlm_logits = self.mlm_head(text_out)
            return itm_logit, mlm_logits
        return itm_logit


def mask_tokens(tok: torch.Tensor, mlm_prob: float = 0.35):
    """Randomly masks content tokens (not <pad>/<s>/</s>) for the MLM objective."""
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
    p = mc.build_argparser("ViLT: Convolution-Free Vision-Language Transformer", epochs=20, batch_size=32)
    args = p.parse_args()

    device = mc.get_device(args.device)

    images, captions = mc.generate_shapes_dataset(num_samples=args.limit or 1200)
    tokens = mc.tokenize_captions(captions)

    split_idx = int(len(images) * 0.8)
    train_img, train_tok = images[:split_idx], tokens[:split_idx]
    test_img, test_tok = images[split_idx:], tokens[split_idx:]
    test_captions = captions[split_idx:]

    model = ViLT(vocab_size=len(mc.VOCAB)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion_itm = nn.BCEWithLogitsLoss()
    criterion_mlm = nn.CrossEntropyLoss(ignore_index=-100)

    print("Training ViLT Model...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_patch_embed_params = sum(p.numel() for p in model.patch_embed.parameters())
    print(f"Device: {device} | trainable params: {n_params:,} | patch embedding params: {n_patch_embed_params:,}")
    print("-" * 64)

    train_dataset = TensorDataset(train_img, train_tok)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        itm_loss_sum, mlm_loss_sum, total = 0.0, 0.0, 0

        for img, tok in train_loader:
            optimizer.zero_grad()

            combined_img, combined_tok, itm_labels = mc.build_itm_pairs(img, tok)
            combined_img, combined_tok, itm_labels = combined_img.to(device), combined_tok.to(device), itm_labels.to(device)
            itm_logits = model(combined_img, combined_tok)
            loss_itm = criterion_itm(itm_logits, itm_labels)

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

    model.eval()

    def score_fn(img_batch, tok_batch):
        return model(img_batch, tok_batch)

    acc, all_scores, targets, unique_caps = mc.evaluate_itm_retrieval(score_fn, test_img, test_captions, device)
    print(f"Test Zero-shot Image-Text Matching Accuracy (1-of-16): {acc * 100:.2f}%")

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
        sim_path = os.path.join(save_dir, "vilt_matching_matrix.png")

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
                                   title="ViLT Convolution-Free Matching Matrix")


if __name__ == "__main__":
    main()
