"""
06. ViLBERT: Dual-Stream Vision-Language Co-Attention Transformer (Lu et al., 2019)
======================================================================================

Two separate Transformer streams -- one for visual tokens, one for text tokens -- are each
self-contextualized independently, then exchange information through co-attention layers
where each stream's queries attend to the *other* stream's keys/values.

Architecture Diagram / Layout:
    Image [B x 3 x 64 x 64] -> ConvGridBackbone -> Visual Stream [B x 17 x 64] (CLS_v + 16 grid tokens)
                                                            |  self-attn
                                                            v
                                                    CoAttentionLayer x N
                                                     (V queries <-> T keys/values
                                                      T queries <-> V keys/values)
                                                            ^
                                                            |  self-attn
    Text [B x SeqLen]       -> Embedding         -> Text Stream   [B x 6 x 64]  (<s> = CLS_t + tokens)
                                                            |
                       CLS_v, CLS_t -> projection -> L2 normalize -> cosine similarity -> Match logit

Key insights / educational takeaways:
    * "Dual-stream": unlike VisualBERT's single shared self-attention stack (`05.visualbert.py`),
      each modality keeps its own Transformer with its own self-attention and feed-forward
      weights, and only *exchanges* information through dedicated cross-attention layers.
    * This lets each stream specialize its representations (different depths/capacities are
      possible per modality) while still allowing deep, bidirectional fusion -- a middle
      ground between CLIP's "no fusion until the very end" and VisualBERT's "fully fused
      from layer one".
    * Like VisualBERT, the co-attention coupling still means embeddings cannot be precomputed
      independently per modality, so retrieval remains an O(N x C) joint-encoding problem.
    * Scoring matches pooled CLS_v/CLS_t with a *projected cosine similarity* (CLIP-style)
      rather than an MLP over the concatenated/multiplied pair: an MLP fusion head trained
      with sparse in-batch hard negatives was empirically much harder to optimize for
      fine-grained (same-color, different-shape) distinctions on this task.

Run:
    python "06.vilbert.py" --epochs 20
    python "06.vilbert.py" --limit 2000        # fast smoke test
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import multimodal_common as mc


class ConvGridBackbone(nn.Module):
    """CNN backbone producing a spatial grid of visual tokens."""
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
        return h.reshape(B, C, H * W).permute(0, 2, 1) # [B, 16, d_model]


class CoAttentionLayer(nn.Module):
    """Exchanges information between the visual and text streams via cross-attention."""
    def __init__(self, dim: int = 64, n_heads: int = 4):
        super().__init__()
        self.v2t_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True) # visual queries, text K/V
        self.t2v_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True) # text queries, visual K/V
        self.v_norm1 = nn.LayerNorm(dim)
        self.t_norm1 = nn.LayerNorm(dim)
        self.v_ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))
        self.t_ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))
        self.v_norm2 = nn.LayerNorm(dim)
        self.t_norm2 = nn.LayerNorm(dim)

    def forward(self, v, t):
        v_attn, _ = self.v2t_attn(v, t, t)
        t_attn, _ = self.t2v_attn(t, v, v)
        v = self.v_norm1(v + v_attn)
        t = self.t_norm1(t + t_attn)
        v = self.v_norm2(v + self.v_ffn(v))
        t = self.t_norm2(t + self.t_ffn(t))
        return v, t


class ViLBERT(nn.Module):
    """Dual-stream visual/text Transformers fused through co-attention, trained for Image-Text Matching."""
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_co_layers: int = 2,
                 n_visual_tokens: int = 16, seq_len: int = 6):
        super().__init__()
        self.backbone = ConvGridBackbone(d_model)
        self.text_embedding = nn.Embedding(vocab_size, d_model)

        self.visual_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.visual_pos_embedding = nn.Embedding(1 + n_visual_tokens, d_model)
        self.text_pos_embedding = nn.Embedding(seq_len, d_model)

        v_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2, batch_first=True)
        t_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2, batch_first=True)
        self.v_self_attn = nn.TransformerEncoder(v_layer, num_layers=1)
        self.t_self_attn = nn.TransformerEncoder(t_layer, num_layers=1)

        self.co_layers = nn.ModuleList([CoAttentionLayer(d_model, n_heads) for _ in range(n_co_layers)])

        # Pooled CLS_v/CLS_t are matched via a projected cosine similarity (CLIP-style head),
        # rather than an MLP over the concatenated pair -- see docstring for why.
        self.v_proj = nn.Linear(d_model, d_model)
        self.t_proj = nn.Linear(d_model, d_model)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(10.0)), dtype=torch.float32))

        self.register_buffer("visual_pos_ids", torch.arange(1 + n_visual_tokens, dtype=torch.long))
        self.register_buffer("text_pos_ids", torch.arange(seq_len, dtype=torch.long))

    def forward(self, image, tokens):
        B = image.size(0)
        visual_tokens = self.backbone(image) # [B, 16, d_model]
        v = torch.cat([self.visual_cls.expand(B, -1, -1), visual_tokens], dim=1) # [B, 17, d_model]
        v = v + self.visual_pos_embedding(self.visual_pos_ids)

        t = self.text_embedding(tokens) + self.text_pos_embedding(self.text_pos_ids) # [B, 6, d_model], <s> acts as CLS_t

        v = self.v_self_attn(v) # independent visual-stream self-attention
        t = self.t_self_attn(t) # independent text-stream self-attention

        for layer in self.co_layers:
            v, t = layer(v, t)   # bidirectional cross-stream fusion

        v_cls, t_cls = v[:, 0], t[:, 0]
        v_emb = nn.functional.normalize(self.v_proj(v_cls), p=2, dim=-1)
        t_emb = nn.functional.normalize(self.t_proj(t_cls), p=2, dim=-1)
        return (v_emb * t_emb).sum(dim=-1) * torch.exp(self.logit_scale)


def main():
    p = mc.build_argparser("ViLBERT Dual-Stream Co-Attention Transformer", epochs=22, batch_size=32)
    args = p.parse_args()

    device = mc.get_device(args.device)

    images, captions = mc.generate_shapes_dataset(num_samples=args.limit or 1200)
    tokens = mc.tokenize_captions(captions)

    split_idx = int(len(images) * 0.8)
    train_img, train_tok = images[:split_idx], tokens[:split_idx]
    test_img, test_tok = images[split_idx:], tokens[split_idx:]
    test_captions = captions[split_idx:]

    model = ViLBERT(vocab_size=len(mc.VOCAB)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    print("Training ViLBERT Model...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    train_dataset = TensorDataset(train_img, train_tok)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, total = 0.0, 0

        for img, tok in train_loader:
            optimizer.zero_grad()

            combined_img, combined_tok, itm_labels = mc.build_itm_pairs(img, tok)
            combined_img, combined_tok, itm_labels = combined_img.to(device), combined_tok.to(device), itm_labels.to(device)

            itm_logits = model(combined_img, combined_tok)
            loss = criterion(itm_logits, itm_labels)
            loss.backward()
            optimizer.step()

            B = img.size(0)
            epoch_loss += loss.item() * B
            total += B

        print(f"Epoch {epoch:2d}/{args.epochs} | itm_loss: {epoch_loss / total:.4f}")

    print("-" * 64)

    model.eval()

    def score_fn(img_batch, tok_batch):
        return model(img_batch, tok_batch)

    acc, all_scores, targets, unique_caps = mc.evaluate_itm_retrieval(score_fn, test_img, test_captions, device)
    print(f"Test Zero-shot Image-Text Matching Accuracy (1-of-16): {acc * 100:.2f}%")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        sim_path = os.path.join(save_dir, "vilbert_matching_matrix.png")

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
                                   title="ViLBERT Dual-Stream Co-Attention Matching Matrix")


if __name__ == "__main__":
    main()
