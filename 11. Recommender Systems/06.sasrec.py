"""
06. SASRec (Self-Attentive Sequential Recommendation)
=====================================================

Unidirectional self-attention over a user's interaction history (Kang & McAuley, 2018).

Instead of a static user embedding, SASRec represents a user by the *sequence*
of items they have consumed, and uses a causal (left-to-right) Transformer to
predict the next item from all previously attended items. It captures both
long-range dependencies (like RNNs) and the adaptive focus of attention, while
training in parallel across sequence positions.

This is the first *sequential* model in the folder, so evaluation switches from
rating MSE to top-K ranking under a leave-one-out protocol: each user's final
interaction is held out as the test target, the second-to-last as validation,
and the rest form the training sequence.

Architecture Diagram / Layout:
    seq [B x L] (item ids, 0 = left-pad)
        -> Item Embedding + Positional Embedding [B x L x d]
        -> N x { Causal Multi-Head Self-Attention -> Point-wise FFN }  (pre-LN, residual)
        -> LayerNorm -> hidden [B x L x d]
        -> logits[t] = hidden[t] . ItemEmbedding^T   (predict item at t+1)

Key insights / educational takeaways:
    * A causal attention mask makes the model autoregressive: position t may only
      attend to positions <= t, so every position is a valid next-item predictor.
    * Sharing the item embedding table between input and output layer ties
      representation and prediction, cutting parameters and improving ranking.
    * Deviation from the paper: we optimize a full-softmax next-item cross-entropy
      (clean at MovieLens' ~1.6k item scale) instead of sampled binary
      cross-entropy with one negative per position.

Run:
    python "06.sasrec.py" --epochs 30
    python "06.sasrec.py" --limit 5000 --epochs 3   # fast smoke test
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import rec_common as mc


class SASRec(nn.Module):
    """Self-attentive sequential recommender with a shared item embedding table.

    Item ids are 1-based here (0 is reserved for left-padding); the model
    vocabulary therefore has ``num_items + 1`` rows.
    """
    def __init__(self, num_items: int, max_len: int = 50, embed_dim: int = 64,
                 num_blocks: int = 2, num_heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len

        self.item_embed = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.attn_norms = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_blocks):
            self.attn_norms.append(nn.LayerNorm(embed_dim))
            self.attns.append(nn.MultiheadAttention(embed_dim, num_heads,
                                                    dropout=dropout, batch_first=True))
            self.ffn_norms.append(nn.LayerNorm(embed_dim))
            self.ffns.append(nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
                nn.Dropout(dropout),
            ))
        self.final_norm = nn.LayerNorm(embed_dim)

        nn.init.normal_(self.item_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        with torch.no_grad():
            self.item_embed.weight[0].zero_()  # keep padding row at zero

    def encode(self, seq):
        """seq: [B, L] of 1-based item ids (0 = pad). Returns hidden [B, L, d]."""
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_embed(seq) + self.pos_embed(positions)
        x = self.emb_dropout(x)

        pad_mask = seq == 0  # [B, L] True where padded
        # Causal mask: position i cannot attend to j > i.
        causal = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1)

        for attn_norm, attn, ffn_norm, ffn in zip(
                self.attn_norms, self.attns, self.ffn_norms, self.ffns):
            q = attn_norm(x)
            attn_out, _ = attn(q, q, q, attn_mask=causal,
                               key_padding_mask=pad_mask, need_weights=False)
            x = x + attn_out
            x = x + ffn(ffn_norm(x))
            # Zero-out padded positions so they never leak into later steps.
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        return self.final_norm(x)

    def forward(self, seq):
        """Training logits over the full vocabulary for every position."""
        hidden = self.encode(seq)                       # [B, L, d]
        logits = hidden @ self.item_embed.weight.t()    # [B, L, num_items+1]
        return logits

    def predict_last(self, seq):
        """Scores over real items (1..num_items) from the last position."""
        hidden = self.encode(seq)[:, -1, :]             # [B, d]
        logits = hidden @ self.item_embed.weight.t()    # [B, num_items+1]
        return logits[:, 1:]                            # drop padding column -> 0-based items


def build_sequences(sequences, max_len):
    """Leave-one-out split producing left-padded training tensors.

    For each user with >= 3 interactions:
      * test target  = last item
      * train sequence = all but the last item
      * inside the train sequence, predict item t+1 from items <= t.
    Returns training (input, target) tensors plus per-user eval structures.
    """
    train_inputs, train_targets = [], []
    eval_inputs, eval_train_items, eval_test_items = [], [], []

    for seq in sequences:
        if len(seq) < 3:
            continue
        # 1-based ids for the model (0 = pad).
        shifted = [i + 1 for i in seq]
        train_seq = shifted[:-1]        # hold out the final interaction
        test_item = seq[-1]             # 0-based ground-truth

        # --- training pair: inputs predict the next item at each step ---
        inp = train_seq[:-1][-max_len:]
        tgt = train_seq[1:][-max_len:]
        pad = max_len - len(inp)
        train_inputs.append([0] * pad + inp)
        train_targets.append([0] * pad + tgt)

        # --- eval: feed the whole train sequence, predict the held-out item ---
        ev = train_seq[-max_len:]
        pad = max_len - len(ev)
        eval_inputs.append([0] * pad + ev)
        eval_train_items.append(seq[:-1])   # 0-based items to exclude from ranking
        eval_test_items.append([test_item])

    return (
        torch.tensor(train_inputs, dtype=torch.long),
        torch.tensor(train_targets, dtype=torch.long),
        torch.tensor(eval_inputs, dtype=torch.long),
        eval_train_items,
        eval_test_items,
    )


def main():
    p = mc.build_argparser("MovieLens SASRec sequential recommender", epochs=30)
    p.add_argument("--max-len", type=int, default=50)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=2)
    args = p.parse_args()

    device = mc.get_device(args.device)

    sequences, num_users, num_items = mc.load_movielens_sequences(limit=args.limit)
    train_in, train_tgt, eval_in, eval_train_items, eval_test_items = build_sequences(
        sequences, args.max_len)
    print(f"Usable users (>=3 interactions): {train_in.size(0)}")

    model = SASRec(num_items, max_len=args.max_len, embed_dim=args.embed_dim,
                   num_blocks=args.num_blocks, num_heads=args.num_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padded targets

    print("Training SASRec...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    loader = DataLoader(TensorDataset(train_in, train_tgt),
                        batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, total = 0.0, 0
        for seq, tgt in loader:
            seq, tgt = seq.to(device), tgt.to(device)
            optimizer.zero_grad()
            logits = model(seq)                         # [B, L, V]
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * seq.size(0)
            total += seq.size(0)
        print(f"Epoch {epoch:2d}/{args.epochs} | train_ce: {epoch_loss / total:.4f}")

    print("-" * 64)

    # --- Evaluation: build a full [users, items] score matrix, then rank ---
    model.eval()
    all_scores = []
    eval_loader = DataLoader(eval_in, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for seq in eval_loader:
            seq = seq.to(device)
            all_scores.append(model.predict_last(seq).cpu())
    scores = torch.cat(all_scores, dim=0)              # [num_eval_users, num_items]

    metrics = mc.ranking_metrics_at_k(scores, eval_train_items, eval_test_items, ks=(10, 20))
    mc.print_ranking_metrics(metrics, ks=(10, 20))

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        cluster_path = os.path.join(save_dir, "sasrec_movie_embeddings.png")
        # Drop the padding row (index 0) before plotting item embeddings.
        mc.plot_movie_clusters(model.item_embed.weight[1:], cluster_path,
                               "SASRec Movie Embeddings")


if __name__ == "__main__":
    main()
