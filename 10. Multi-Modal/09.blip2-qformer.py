"""
09. BLIP-2: Bootstrapping Vision-Language via a Frozen Querying Transformer (Li et al., 2023)
=================================================================================================

Every previous script trains its entire vision encoder and text tower end to end. BLIP-2's
key idea is the opposite: FREEZE a large pretrained vision backbone and a large pretrained
language model entirely, and train only a tiny "Querying Transformer" (Q-Former) that
bridges them -- a handful of learnable query tokens cross-attend into the frozen visual
features and the resulting compact summary is fed as a soft-prompt prefix into the frozen
language model.

    Note on fidelity: this repo has no access to real pretrained vision/language weights,
    so "frozen" here means randomly initialized and never updated (requires_grad=False),
    not genuinely pretrained. The lesson this script demonstrates is the *parameter-efficiency
    pattern* -- how few parameters need to be trained to bridge two fixed models -- not the
    absolute caption quality BLIP-2 achieves with real large-scale pretrained backbones.

Architecture Diagram / Layout:
    Image [B x 3 x 64 x 64] -> FrozenImageEncoder (random, requires_grad=False) -> Visual Tokens [B x 16 x 64]
                                                                                            |
    Learnable Queries [1 x 4 x 64] --(self-attn among queries)--(cross-attn to Visual Tokens)--> Query Output [B x 4 x 64]
                                                                                            |
                                                                    trainable Linear projection
                                                                                            |
                                                                                            v
                                          Prefix [B x 4 x 64] ++ Caption Tokens -> FrozenCausalDecoder (random, requires_grad=False)
                                                                                            |
                                                                                            v
                                                                              Per-token vocab logits (LM loss)

Key insights / educational takeaways:
    * The Q-Former (query embeddings + its self/cross-attention layers) and the projection
      layer are the *only* parameters with `requires_grad=True` -- everything else is frozen.
      Compare the printed trainable-vs-frozen parameter counts against `02.image-captioner.py`,
      which trains its CNN + decoder end to end: BLIP-2 trains a tiny fraction of that.
    * Gradients still flow *through* the frozen decoder (into the Q-Former's prefix) even
      though the decoder's own weights never update -- this is what lets the Q-Former learn
      to produce a prefix the fixed decoder can act on, without touching the decoder itself.
    * The learnable queries compress a 16-token visual grid down to just 4 tokens -- a
      deliberate bottleneck forcing the Q-Former to distill only what the frozen decoder
      actually needs, rather than passing all visual information through unfiltered.
    * Convergence is genuinely slower than every other script in this directory (loss
      decreases smoothly for 50+ epochs without plateauing), and it learns *color* long
      before *shape* -- unlike real BLIP-2, whose frozen LM already understands language
      and just needs visual grounding, our frozen decoder is a random function the
      Q-Former must learn to steer from scratch, using only a 4-token bottleneck. That
      is the direct cost of not having real pretrained weights available here.
    * Exact-match caption accuracy is noticeably noisier run-to-run than the fully-trained
      scripts (observed roughly 15-50% at 40 epochs across seeds), since the achievable
      function class -- steering a *fixed random* decoder purely through a 4-token prefix
      -- is far more constrained than fine-tuning the decoder itself. Longer training
      keeps improving it further (~1.0 loss at epoch 40 is not yet a plateau).

Run:
    python "09.blip2-qformer.py" --epochs 40
    python "09.blip2-qformer.py" --limit 2000        # fast smoke test
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import multimodal_common as mc


class FrozenImageEncoder(nn.Module):
    """A randomly-initialized, frozen CNN standing in for a large pretrained vision backbone.

    No BatchNorm -- since these weights are never trained, there is no running-statistics
    train/eval mismatch to worry about (unlike a trained backbone with BatchNorm).
    """
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),               # -> 32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),               # -> 16x16
            nn.Conv2d(32, d_model, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),          # -> 8x8
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # -> 4x4
        )
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        with torch.no_grad():
            h = self.conv(x)
            B, C, H, W = h.shape
            return h.reshape(B, C, H * W).permute(0, 2, 1) # [B, 16, d_model]


class QFormerLayer(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, queries, visual_tokens):
        q, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + q)
        c, _ = self.cross_attn(queries, visual_tokens, visual_tokens)
        queries = self.norm2(queries + c)
        queries = self.norm3(queries + self.ffn(queries))
        return queries


class QFormer(nn.Module):
    """The only substantial trainable component: a handful of learnable query tokens that
    cross-attend into the frozen vision backbone to distill a compact visual summary."""
    def __init__(self, n_queries: int = 4, d_model: int = 64, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.layers = nn.ModuleList([QFormerLayer(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, visual_tokens):
        B = visual_tokens.size(0)
        q = self.query_embed.expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, visual_tokens)
        return q # [B, n_queries, d_model]


class FrozenCausalDecoder(nn.Module):
    """A randomly-initialized, frozen causal Transformer LM standing in for a large pretrained
    language model. Only the Q-Former's soft-prompt prefix can adapt -- this fixed function's
    own weights never update, so all learning happens upstream of it.
    """
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 n_queries: int = 4, seq_len: int = 6):
        super().__init__()
        self.n_queries = n_queries
        total_len = n_queries + seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(total_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2, batch_first=True)
        self.layers = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

        for param in self.parameters():
            param.requires_grad_(False)

        self.register_buffer("pos_ids_full", torch.arange(total_len, dtype=torch.long))
        # Prefix-LM mask: causal overall, but every position can always attend to the query prefix.
        causal = torch.triu(torch.full((total_len, total_len), float("-inf")), diagonal=1)
        causal[:, :n_queries] = 0.0
        self.register_buffer("prefix_causal_mask_full", causal)

    def forward(self, query_prefix, tokens):
        B, L = tokens.shape
        tok_embeds = self.embedding(tokens)
        seq = torch.cat([query_prefix, tok_embeds], dim=1)
        total_len = self.n_queries + L
        seq = seq + self.pos_embedding(self.pos_ids_full[:total_len])
        mask = self.prefix_causal_mask_full[:total_len, :total_len]
        out = self.layers(seq, mask=mask)
        cap_out = out[:, self.n_queries:]
        return self.lm_head(cap_out)


class BLIP2(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_queries: int = 4, seq_len: int = 6):
        super().__init__()
        self.image_encoder = FrozenImageEncoder(d_model)
        self.qformer = QFormer(n_queries=n_queries, d_model=d_model)
        self.proj = nn.Linear(d_model, d_model) # trainable bridge: Q-Former space -> frozen-decoder embedding space
        self.decoder = FrozenCausalDecoder(vocab_size, d_model=d_model, n_queries=n_queries, seq_len=seq_len)

    def encode_prefix(self, image):
        visual_tokens = self.image_encoder(image)
        query_out = self.qformer(visual_tokens)
        return self.proj(query_out)

    def forward(self, image, tokens):
        prefix = self.encode_prefix(image)
        return self.decoder(prefix, tokens)

    def generate_caption(self, image, max_len: int = 6, device: str = "cpu"):
        self.eval()
        with torch.no_grad():
            prefix = self.encode_prefix(image)

        B = image.size(0)
        tokens = torch.full((B, 1), mc.VOCAB["<s>"], dtype=torch.long, device=device)
        finished = np.zeros(B, dtype=bool)
        generated = [[] for _ in range(B)]

        for _ in range(max_len - 1):
            with torch.no_grad():
                logits = self.decoder(prefix, tokens)[:, -1]
            next_tok = logits.argmax(dim=-1)
            tokens = torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)
            for b in range(B):
                if not finished[b]:
                    t = next_tok[b].item()
                    generated[b].append(t)
                    if t == mc.VOCAB["</s>"]:
                        finished[b] = True
            if finished.all():
                break
        return generated


def main():
    p = mc.build_argparser("BLIP-2 Q-Former: Bridging Frozen Vision & Language Models", epochs=40, batch_size=32)
    args = p.parse_args()
    device = mc.get_device(args.device)

    images, captions = mc.generate_shapes_dataset(num_samples=args.limit or 1200)
    tokens = mc.tokenize_captions(captions)

    split_idx = int(len(images) * 0.8)
    train_img, train_tok = images[:split_idx], tokens[:split_idx]
    test_img, test_tok = images[split_idx:], tokens[split_idx:]
    test_captions = captions[split_idx:]

    model = BLIP2(vocab_size=len(mc.VOCAB)).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    n_trainable = sum(p.numel() for p in trainable_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("Training BLIP-2 Q-Former (vision + text backbones frozen)...")
    print(f"Device: {device} | trainable (Q-Former + proj): {n_trainable:,} | "
          f"frozen (vision + text backbones): {n_frozen:,} | "
          f"trainable fraction: {n_trainable / (n_trainable + n_frozen) * 100:.1f}%")
    print("-" * 64)

    train_loader = DataLoader(TensorDataset(train_img, train_tok), batch_size=args.batch_size, shuffle=True, drop_last=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, total = 0.0, 0

        for img, tok in train_loader:
            img, tok = img.to(device), tok.to(device)
            inputs, targets = tok[:, :-1], tok[:, 1:]

            optimizer.zero_grad()
            logits = model(img, inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            B = img.size(0)
            epoch_loss += loss.item() * B
            total += B

        print(f"Epoch {epoch:2d}/{args.epochs} | lm_loss: {epoch_loss / total:.4f}")

    print("-" * 64)

    print("Generating captions on unseen test images...")
    val_images = test_img[:6].to(device)
    val_gt = test_captions[:6]
    generated_tokens = model.generate_caption(val_images, device=device)
    predicted_captions = [mc.detokenize_caption(t) for t in generated_tokens]

    print("\nSample Test Generations:")
    for i in range(len(val_gt)):
        print(f"GT: {val_gt[i]:<18} | Pred: {predicted_captions[i]}")

    # Full-test-set caption accuracy (exact-match), the key readout given there is no
    # retrieval/matching head in this stage-2-style generative bridging setup.
    model.eval()
    all_gen = model.generate_caption(test_img.to(device), device=device)
    all_pred = [mc.detokenize_caption(t) for t in all_gen]
    exact_match = np.mean([p == g for p, g in zip(all_pred, test_captions)])
    print(f"\nTest Exact-Match Caption Accuracy: {exact_match * 100:.2f}%")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        mc.plot_caption_grid(val_images.cpu(), val_gt, predicted_captions,
                              os.path.join(save_dir, "blip2_qformer_results.png"))


if __name__ == "__main__":
    main()
