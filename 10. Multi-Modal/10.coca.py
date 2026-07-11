"""
10. CoCa: Contrastive Captioners (Yu et al., Google Research, 2022)
======================================================================

BLIP (`08.blip.py`) needs three separate forward-pass configurations to compute its
three losses. CoCa gets contrastive alignment *and* captioning from a SINGLE forward
pass, by splitting its text tower into two stages: causal self-attention-only
"unimodal" layers that never see the image (producing a pooled embedding for
contrastive loss), followed by causal self-attention + cross-attention "multimodal"
layers that continue from there (producing per-token logits for captioning).

Architecture Diagram / Layout:
    Image [B x 3 x 64 x 64] -> ConvGridBackbone -> Visual Tokens [B x 16 x 64] -> mean-pool -> Image Embed

    Tokens [B x SeqLen] -> Embedding -> Unimodal Layers (causal self-attn only, NO image)
                                              |
                                              +--> pooled at </s> position --> Text Embed  --\
                                              |                                              +--> Contrastive Loss
                                              v                                Image Embed  --/
                                         Multimodal Layers (causal self-attn + cross-attn to image)
                                              |
                                              v
                                    Per-token vocab logits --> Captioning (LM) Loss

Key insights / educational takeaways:
    * The pooled text embedding is read off *before* the multimodal (image cross-attention)
      layers ever run -- so encoding all 16 candidate captions for retrieval never touches
      the multimodal layers or the image at all, preserving CLIP's fast, precomputable
      dual-encoder retrieval (`encode_text_for_retrieval` / `encode_image_for_retrieval`
      below) while *also* getting a captioning decoder from the same weights.
    * Because our synthetic captions are always the fixed format "a {color} {shape}", the
      </s> token always lands at the same final position of the (teacher-forced) input --
      so a plain "take the last position" pool works as the causal analogue of a [CLS]
      token, without needing to append one explicitly (unlike the original paper).
    * No Image-Text Matching loss at all: unlike BLIP, CoCa relies purely on contrastive +
      captioning, betting that a strong enough decoder makes the fused ITM objective
      unnecessary -- a simplicity/efficiency trade against BLIP's three-objective recipe.
    * A single unimodal layer is not enough capacity to serve two masters: with only one
      shared self-attention layer before the pooling point, the fast-converging LM
      gradient dominates and retrieval accuracy plateaus around ~46%. Two unimodal layers
      give the contrastive path enough of its own representational room and retrieval
      reaches 100% -- a reminder that "single forward pass, multiple losses" isn't free:
      the shared trunk needs enough depth to actually serve every objective attached to it.

Run:
    python "10.coca.py" --epochs 15
    python "10.coca.py" --limit 2000        # fast smoke test
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
        h = self.conv(x)
        B, C, H, W = h.shape
        return h.reshape(B, C, H * W).permute(0, 2, 1) # [B, 16, d_model]


class CausalSelfAttnLayer(nn.Module):
    """Unimodal decoder layer: causal self-attention only, never sees the image."""
    def __init__(self, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, causal_mask):
        a, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ffn(x))
        return x


class CausalCrossAttnLayer(nn.Module):
    """Multimodal decoder layer: causal self-attention + cross-attention to the image."""
    def __init__(self, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, image_tokens, causal_mask):
        a, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + a)
        c, _ = self.cross_attn(x, image_tokens, image_tokens)
        x = self.norm2(x + c)
        x = self.norm3(x + self.ffn(x))
        return x


class CoCaTextTower(nn.Module):
    """Unimodal (contrastive) layers feed directly into multimodal (captioning) layers."""
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4,
                 n_unimodal: int = 2, n_multimodal: int = 1, seq_len: int = 6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        self.unimodal_layers = nn.ModuleList([CausalSelfAttnLayer(d_model, n_heads) for _ in range(n_unimodal)])
        self.multimodal_layers = nn.ModuleList([CausalCrossAttnLayer(d_model, n_heads) for _ in range(n_multimodal)])

        self.register_buffer("pos_ids_full", torch.arange(seq_len, dtype=torch.long))
        causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask_full", causal)

    def embed_causal(self, tokens):
        B, L = tokens.shape
        x = self.embedding(tokens) + self.pos_embedding(self.pos_ids_full[:L])
        mask = self.causal_mask_full[:L, :L]
        return x, mask

    def forward(self, tokens, image_tokens):
        x, mask = self.embed_causal(tokens)
        for layer in self.unimodal_layers:
            x = layer(x, mask)
        text_cls = x[:, -1]                               # </s> is always the last real token here
        for layer in self.multimodal_layers:
            x = layer(x, image_tokens, mask)
        return text_cls, x


class CoCa(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, seq_len: int = 6):
        super().__init__()
        self.image_encoder = ConvGridBackbone(d_model)
        self.img_pool_proj = nn.Linear(d_model, d_model)
        self.text_tower = CoCaTextTower(vocab_size, d_model, seq_len=seq_len)
        self.txt_proj = nn.Linear(d_model, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1 / 0.07)), dtype=torch.float32))

    def forward(self, image, tokens):
        visual_tokens = self.image_encoder(image)
        img_pooled = visual_tokens.mean(dim=1)
        img_e = nn.functional.normalize(self.img_pool_proj(img_pooled), dim=-1)

        text_cls, text_hidden = self.text_tower(tokens, visual_tokens)
        txt_e = nn.functional.normalize(self.txt_proj(text_cls), dim=-1)

        lm_logits = self.lm_head(text_hidden)
        return img_e, txt_e, lm_logits

    def encode_image_for_retrieval(self, image):
        visual_tokens = self.image_encoder(image)
        img_pooled = visual_tokens.mean(dim=1)
        return nn.functional.normalize(self.img_pool_proj(img_pooled), dim=-1)

    def encode_text_for_retrieval(self, tokens):
        """Only the unimodal layers run -- no image needed, exactly like a CLIP text tower."""
        x, mask = self.text_tower.embed_causal(tokens)
        for layer in self.text_tower.unimodal_layers:
            x = layer(x, mask)
        return nn.functional.normalize(self.txt_proj(x[:, -1]), dim=-1)

    def generate_caption(self, image, max_len: int = 6, device: str = "cpu"):
        self.eval()
        with torch.no_grad():
            visual_tokens = self.image_encoder(image)

        B = image.size(0)
        tokens = torch.full((B, 1), mc.VOCAB["<s>"], dtype=torch.long, device=device)
        finished = np.zeros(B, dtype=bool)
        generated = [[] for _ in range(B)]

        for _ in range(max_len - 1):
            with torch.no_grad():
                _, text_hidden = self.text_tower(tokens, visual_tokens)
                logits = self.lm_head(text_hidden[:, -1])
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
    p = mc.build_argparser("CoCa: Contrastive Captioner (single-pass ITC + LM)", epochs=15, batch_size=32)
    args = p.parse_args()
    device = mc.get_device(args.device)

    images, captions = mc.generate_shapes_dataset(num_samples=args.limit or 1200)
    tokens = mc.tokenize_captions(captions)

    split_idx = int(len(images) * 0.8)
    train_img, train_tok = images[:split_idx], tokens[:split_idx]
    test_img, test_tok = images[split_idx:], tokens[split_idx:]
    test_captions = captions[split_idx:]

    model = CoCa(vocab_size=len(mc.VOCAB)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("Training CoCa Model...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    train_loader = DataLoader(TensorDataset(train_img, train_tok), batch_size=args.batch_size, shuffle=True, drop_last=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        contrastive_sum, lm_sum, total = 0.0, 0.0, 0

        for img, tok in train_loader:
            img, tok = img.to(device), tok.to(device)
            inputs, targets = tok[:, :-1], tok[:, 1:]      # single forward on the teacher-forced input

            optimizer.zero_grad()
            img_e, txt_e, lm_logits = model(img, inputs)

            logits_c = torch.matmul(img_e, txt_e.T) * torch.exp(model.logit_scale)
            c_targets = torch.arange(img.size(0), device=device)
            loss_contrastive = (criterion(logits_c, c_targets) + criterion(logits_c.T, c_targets)) / 2.0

            loss_lm = criterion(lm_logits.reshape(-1, lm_logits.size(-1)), targets.reshape(-1))

            loss = loss_contrastive + loss_lm
            loss.backward()
            optimizer.step()

            B = img.size(0)
            contrastive_sum += loss_contrastive.item() * B
            lm_sum += loss_lm.item() * B
            total += B

        print(f"Epoch {epoch:2d}/{args.epochs} | contrastive_loss: {contrastive_sum / total:.4f} | lm_loss: {lm_sum / total:.4f}")

    print("-" * 64)
    model.eval()

    # Zero-shot retrieval: only the unimodal text layers + image encoder run, just like CLIP.
    with torch.no_grad():
        colors_list = ["red", "green", "blue", "yellow"]
        shapes_list = ["circle", "square", "triangle", "cross"]
        unique_caps = [f"a {c} {s}" for c in colors_list for s in shapes_list]
        unique_toks = mc.tokenize_captions(unique_caps).to(device)

        val_img = test_img[:100].to(device)
        val_captions = test_captions[:100]

        img_e = model.encode_image_for_retrieval(val_img)
        txt_e = model.encode_text_for_retrieval(unique_toks)

        logits = torch.matmul(img_e, txt_e.T)
        preds = logits.argmax(dim=1).cpu().numpy()
        cap_to_idx = {cap: idx for idx, cap in enumerate(unique_caps)}
        targets = np.array([cap_to_idx[cap] for cap in val_captions])
        acc = np.mean(preds == targets)

    print(f"Test Zero-shot Retrieval Accuracy (1-of-16): {acc * 100:.2f}%")

    print("\nGenerating captions on unseen test images...")
    val_images = test_img[:6].to(device)
    val_gt = test_captions[:6]
    generated_tokens = model.generate_caption(val_images, device=device)
    predicted_captions = [mc.detokenize_caption(t) for t in generated_tokens]

    print("\nSample Test Generations:")
    for i in range(len(val_gt)):
        print(f"GT: {val_gt[i]:<18} | Pred: {predicted_captions[i]}")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))

        unique_indices = []
        seen = set()
        for idx, label in enumerate(test_captions):
            if label not in seen:
                seen.add(label)
                unique_indices.append(idx)
            if len(unique_indices) == 8:
                break

        sub_img = test_img[unique_indices].to(device)
        sub_captions = [test_captions[i] for i in unique_indices]
        with torch.no_grad():
            sub_img_e = model.encode_image_for_retrieval(sub_img)
            sub_txt_e = model.encode_text_for_retrieval(mc.tokenize_captions(sub_captions).to(device))
            cosine_sim = torch.matmul(sub_img_e, sub_txt_e.T)

        mc.plot_similarity_matrix(cosine_sim.cpu().numpy(), sub_captions,
                                   os.path.join(save_dir, "coca_similarity_matrix.png"),
                                   title="CoCa Contrastive Retrieval Matrix")
        mc.plot_caption_grid(val_images.cpu(), val_gt, predicted_captions,
                              os.path.join(save_dir, "coca_captioner_results.png"))


if __name__ == "__main__":
    main()
