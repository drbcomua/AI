"""
08. BLIP: Bootstrapping Language-Image Pretraining (Li et al., Salesforce, 2022)
===================================================================================

A single text Transformer is reused in three different "modes" against one shared
image encoder, unifying every task family seen so far in this directory into one model:
retrieval (`01.clip.py`), captioning (`02.image-captioner.py`), and fused matching
(`05.visualbert.py`) all become one architecture with three loss terms.

Architecture Diagram / Layout:
    Image [B x 3 x 64 x 64] -> ConvGridBackbone -> Visual Tokens [B x 16 x 64]
                                                            |
      +-----------------------+-----------------------------+-----------------------------+
      v                       v                                                           v
    Unimodal Text Encoder   Image-Grounded Text Encoder                    Image-Grounded Text Decoder
    (self-attn only,        (self-attn + cross-attn to visual tokens,      (CAUSAL self-attn + cross-attn,
     bidirectional)          bidirectional)                                 teacher-forced)
      |                       |                                                           |
      v                       v                                                           v
    ITC: cosine-sim         ITM: [<s>] output -> binary match logit        LM: per-token vocab logits
    retrieval embedding      (mc.build_itm_pairs hard negatives)            (autoregressive captioning)

Key insights / educational takeaways:
    * All three modes share the *same* embedding table and Transformer layer weights
      (`BLIPTextTransformer`) -- only the attention pattern (causal vs. not, cross-attention
      to the image or not) and the final head differ per mode. This is BLIP's central idea:
      one multimodal backbone, multiple objectives, rather than three separate models.
    * ITC gives fast, precomputable retrieval (like CLIP) since the unimodal encoder never
      looks at the image. ITM gives accurate but expensive fused matching (like VisualBERT)
      since the image-grounded encoder fuses modalities every layer. Real BLIP uses ITC's
      cheap similarity scores to mine hard negatives for ITM -- here we reuse
      `mc.build_itm_pairs`'s batch-rolled negatives as a simpler stand-in.
    * The LM decoder reuses the exact same weights as the ITM encoder, just switched to a
      causal attention mask -- demonstrating how little architectural change separates
      "understanding" (matching) from "generation" (captioning) in a Transformer.

Run:
    python "08.blip.py" --epochs 15
    python "08.blip.py" --limit 2000        # fast smoke test
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


class BLIPTextLayer(nn.Module):
    """One shared layer used by all three BLIP modes: self-attn, optional cross-attn, FFN."""
    def __init__(self, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, image_tokens=None, causal_mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + attn_out)
        if image_tokens is not None:                     # cross-attention only in "image-grounded" modes
            cross_out, _ = self.cross_attn(x, image_tokens, image_tokens)
            x = self.norm2(x + cross_out)
        x = self.norm3(x + self.ffn(x))
        return x


class BLIPTextTransformer(nn.Module):
    """Shared text tower: same weights power the unimodal encoder, fused encoder, and decoder modes."""
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, seq_len: int = 6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        self.layers = nn.ModuleList([BLIPTextLayer(d_model, n_heads) for _ in range(n_layers)])
        self.register_buffer("pos_ids_full", torch.arange(seq_len, dtype=torch.long))
        causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask_full", causal)

    def forward(self, tokens, image_tokens=None, causal: bool = False):
        B, L = tokens.shape
        x = self.embedding(tokens) + self.pos_embedding(self.pos_ids_full[:L])
        mask = self.causal_mask_full[:L, :L] if causal else None
        for layer in self.layers:
            x = layer(x, image_tokens=image_tokens, causal_mask=mask)
        return x


class BLIP(nn.Module):
    """One image encoder + one shared text tower, used in three modes: ITC, ITM, LM."""
    def __init__(self, vocab_size: int, d_model: int = 64, seq_len: int = 6):
        super().__init__()
        self.image_encoder = ConvGridBackbone(d_model)
        self.text = BLIPTextTransformer(vocab_size, d_model, seq_len=seq_len)

        self.itc_img_proj = nn.Linear(d_model, d_model)
        self.itc_txt_proj = nn.Linear(d_model, d_model)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1 / 0.07)), dtype=torch.float32))

        self.itm_head = nn.Linear(d_model, 1)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def encode_image(self, image):
        visual_tokens = self.image_encoder(image)        # [B, 16, d_model]
        img_pooled = visual_tokens.mean(dim=1)
        return visual_tokens, img_pooled

    def forward_itc(self, image, tokens):
        """Unimodal mode: text never sees the image -- fast, precomputable embeddings."""
        _, img_pooled = self.encode_image(image)
        text_hidden = self.text(tokens, image_tokens=None, causal=False)
        txt_pooled = text_hidden[:, 0]                    # <s> position
        img_e = nn.functional.normalize(self.itc_img_proj(img_pooled), dim=-1)
        txt_e = nn.functional.normalize(self.itc_txt_proj(txt_pooled), dim=-1)
        return img_e, txt_e

    def forward_itm(self, image, tokens):
        """Image-grounded encoder mode: bidirectional self-attn + cross-attn to the image, every layer."""
        visual_tokens, _ = self.encode_image(image)
        fused = self.text(tokens, image_tokens=visual_tokens, causal=False)
        return self.itm_head(fused[:, 0]).squeeze(-1)

    def forward_lm(self, image, tokens):
        """Image-grounded decoder mode: same weights as ITM, but with a causal attention mask."""
        visual_tokens, _ = self.encode_image(image)
        dec = self.text(tokens, image_tokens=visual_tokens, causal=True)
        return self.lm_head(dec)

    def generate_caption(self, image, max_len: int = 6, device: str = "cpu"):
        """Autoregressive decoding loop using the LM mode."""
        self.eval()
        visual_tokens, _ = self.encode_image(image)
        B = image.size(0)
        tokens = torch.full((B, 1), mc.VOCAB["<s>"], dtype=torch.long, device=device)
        finished = np.zeros(B, dtype=bool)
        generated = [[] for _ in range(B)]

        for _ in range(max_len - 1):
            dec = self.text(tokens, image_tokens=visual_tokens, causal=True)
            logits = self.lm_head(dec[:, -1])
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
    p = mc.build_argparser("BLIP: Bootstrapping Language-Image Pretraining (ITC + ITM + LM)", epochs=15, batch_size=32)
    args = p.parse_args()
    device = mc.get_device(args.device)

    images, captions = mc.generate_shapes_dataset(num_samples=args.limit or 1200)
    tokens = mc.tokenize_captions(captions)

    split_idx = int(len(images) * 0.8)
    train_img, train_tok = images[:split_idx], tokens[:split_idx]
    test_img, test_tok = images[split_idx:], tokens[split_idx:]
    test_captions = captions[split_idx:]

    model = BLIP(vocab_size=len(mc.VOCAB)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion_itc = nn.CrossEntropyLoss()
    criterion_itm = nn.BCEWithLogitsLoss()
    criterion_lm = nn.CrossEntropyLoss()

    print("Training BLIP Model...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    train_loader = DataLoader(TensorDataset(train_img, train_tok), batch_size=args.batch_size, shuffle=True, drop_last=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        itc_sum, itm_sum, lm_sum, total = 0.0, 0.0, 0.0, 0

        for img, tok in train_loader:
            img, tok = img.to(device), tok.to(device)
            optimizer.zero_grad()

            # 1. Image-Text Contrastive (unimodal encoder, symmetric softmax like CLIP)
            img_e, txt_e = model.forward_itc(img, tok)
            logits_itc = torch.matmul(img_e, txt_e.T) * torch.exp(model.logit_scale)
            itc_targets = torch.arange(img.size(0), device=device)
            loss_itc = (criterion_itc(logits_itc, itc_targets) + criterion_itc(logits_itc.T, itc_targets)) / 2.0

            # 2. Image-Text Matching (image-grounded encoder, positive + hard-negative pairs)
            combined_img, combined_tok, itm_labels = mc.build_itm_pairs(img, tok)
            itm_labels = itm_labels.to(device)
            itm_logits = model.forward_itm(combined_img, combined_tok)
            loss_itm = criterion_itm(itm_logits, itm_labels)

            # 3. Language Modeling (image-grounded decoder, teacher forcing)
            lm_inputs, lm_targets = tok[:, :-1], tok[:, 1:]
            lm_logits = model.forward_lm(img, lm_inputs)
            loss_lm = criterion_lm(lm_logits.reshape(-1, lm_logits.size(-1)), lm_targets.reshape(-1))

            loss = loss_itc + loss_itm + loss_lm
            loss.backward()
            optimizer.step()

            B = img.size(0)
            itc_sum += loss_itc.item() * B
            itm_sum += loss_itm.item() * B
            lm_sum += loss_lm.item() * B
            total += B

        print(f"Epoch {epoch:2d}/{args.epochs} | itc_loss: {itc_sum / total:.4f} | "
              f"itm_loss: {itm_sum / total:.4f} | lm_loss: {lm_sum / total:.4f}")

    print("-" * 64)
    model.eval()

    # Zero-shot retrieval via the cheap ITC embeddings (same protocol as 01.clip.py)
    with torch.no_grad():
        colors_list = ["red", "green", "blue", "yellow"]
        shapes_list = ["circle", "square", "triangle", "cross"]
        unique_caps = [f"a {c} {s}" for c in colors_list for s in shapes_list]
        unique_toks = mc.tokenize_captions(unique_caps).to(device)

        val_img = test_img[:100].to(device)
        val_captions = test_captions[:100]

        _, img_pooled = model.encode_image(val_img)
        text_hidden = model.text(unique_toks, image_tokens=None, causal=False)
        img_e = nn.functional.normalize(model.itc_img_proj(img_pooled), dim=-1)
        txt_e = nn.functional.normalize(model.itc_txt_proj(text_hidden[:, 0]), dim=-1)

        logits = torch.matmul(img_e, txt_e.T)
        preds = logits.argmax(dim=1).cpu().numpy()
        cap_to_idx = {cap: idx for idx, cap in enumerate(unique_caps)}
        targets = np.array([cap_to_idx[cap] for cap in val_captions])
        acc = np.mean(preds == targets)

    print(f"Test Zero-shot ITC Retrieval Accuracy (1-of-16): {acc * 100:.2f}%")

    # Captioning generation via the LM decoder mode (same protocol as 02.image-captioner.py)
    print("\nGenerating captions on unseen test images...")
    val_images = test_img[:6].to(device)
    val_gt = test_captions[:6]
    with torch.no_grad():
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
            _, sub_img_pooled = model.encode_image(sub_img)
            sub_text_hidden = model.text(mc.tokenize_captions(sub_captions).to(device), image_tokens=None, causal=False)
            sub_img_e = nn.functional.normalize(model.itc_img_proj(sub_img_pooled), dim=-1)
            sub_txt_e = nn.functional.normalize(model.itc_txt_proj(sub_text_hidden[:, 0]), dim=-1)
            cosine_sim = torch.matmul(sub_img_e, sub_txt_e.T)

        mc.plot_similarity_matrix(cosine_sim.cpu().numpy(), sub_captions,
                                   os.path.join(save_dir, "blip_similarity_matrix.png"),
                                   title="BLIP Image-Text Contrastive (ITC) Matrix")
        mc.plot_caption_grid(val_images.cpu(), val_gt, predicted_captions,
                              os.path.join(save_dir, "blip_captioner_results.png"))


if __name__ == "__main__":
    main()
