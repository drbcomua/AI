"""
07. Graph Transformer — neighborhood attention + Laplacian positional encoding
==============================================================================

Brings the Transformer to graphs (Dwivedi & Bresson, 2020). Each node uses
scaled dot-product multi-head attention over its **graph neighborhood** (the
attention is masked to edges + self), inside the full Transformer block (FFN,
residuals, LayerNorm). Because a graph has no natural node ordering, structure is
injected through a **Laplacian positional encoding**: the lowest non-trivial
eigenvectors of the graph Laplacian act like sinusoidal positions, giving each
node a structural "coordinate."

Architecture Diagram / Layout:
    node features X  (+ Laplacian PE)  -> Linear -> [N, d]
       -> N x Transformer Encoder layers (edge-masked multi-head attention + FFN)
       -> Linear classifier -> [N, num_classes]

Key insights / educational takeaways:
    * Dot-product attention generalizes GAT's additive attention; stacking layers
      grows the receptive field hop by hop.
    * Laplacian eigenvectors are the graph analogue of positional encodings — they
      are smooth, structure-aware coordinates (sign-flips are a known ambiguity).
    * Masking attention to edges injects the graph prior; unrestricted global
      attention badly overfits Cora's 140 training labels.

Run:
    python "07.graph-transformer.py" --epochs 100
"""

import os
import numpy as np
import torch
import torch.nn as nn
import gnn_common as mc

PE_DIM = 8


def laplacian_pe(adj_raw, k=PE_DIM):
    """k lowest non-trivial eigenvectors of the symmetric normalized Laplacian."""
    a = adj_raw.cpu().numpy()
    n = a.shape[0]
    deg = a.sum(1)
    d_inv_sqrt = np.zeros_like(deg); np.power(deg, -0.5, where=deg > 0, out=d_inv_sqrt)
    L = np.eye(n) - (d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :])
    vals, vecs = np.linalg.eigh(L)                         # ascending eigenvalues
    pe = vecs[:, 1:k + 1]                                  # skip the trivial constant eigenvector
    return torch.tensor(pe, dtype=torch.float32)


class GraphTransformer(nn.Module):
    def __init__(self, in_dim, pe_dim, d_model, num_classes, n_layers=3, n_heads=4, dropout=0.5):
        super().__init__()
        self.embed = nn.Linear(in_dim + pe_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, pe, attn_mask):
        h = self.embed(torch.cat([x, pe], dim=-1)).unsqueeze(0)   # [1, N, d]
        h = self.transformer(h, mask=attn_mask).squeeze(0)        # attention masked to neighborhoods
        return self.head(h), h


def main():
    args = mc.build_argparser("Graph Transformer", epochs=100, lr=5e-3).parse_args()
    device = mc.get_device(args.device)

    features, adj, labels, splits = mc.load_cora(limit=args.limit)
    _, adj_raw, _ = mc.load_cora_raw()
    if args.limit:
        adj_raw = adj_raw[:args.limit, :args.limit]
    pe = laplacian_pe(adj_raw).to(device)
    # attention mask: True where attention is FORBIDDEN (i.e. non-neighbours). Self-loops
    # are added so every node attends to itself (no fully-masked rows).
    n = adj_raw.size(0)
    connected = (adj_raw + torch.eye(n)) > 0
    attn_mask = (~connected).to(device)
    features, labels = features.to(device), labels.to(device)
    train_mask = splits["train_mask"].to(device)
    val_mask = splits["val_mask"].to(device)
    test_mask = splits["test_mask"].to(device)

    model = GraphTransformer(features.size(1), PE_DIM, 64, int(labels.max().item() + 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    print(f"Training Graph Transformer on Cora | params: {sum(q.numel() for q in model.parameters()):,}")
    print("-" * 64)
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad()
        out, _ = model(features, pe, attn_mask)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward(); optimizer.step()
        if epoch % max(1, args.epochs // 10) == 0:
            model.eval()
            with torch.no_grad():
                logits, _ = model(features, pe, attn_mask)
                val_acc = (logits.argmax(1)[val_mask] == labels[val_mask]).float().mean().item()
            print(f"Epoch {epoch:3d}/{args.epochs} | train_loss {loss.item():.4f} | val_acc {val_acc:.4f}")
    print("-" * 64)

    model.eval()
    with torch.no_grad():
        logits, emb = model(features, pe, attn_mask)
        test_acc = (logits.argmax(1)[test_mask] == labels[test_mask]).float().mean().item()
    print(f"Graph Transformer Test Accuracy: {test_acc:.4f}")

    save_dir = os.path.dirname(os.path.abspath(__file__))
    mc.plot_tsne_embeddings(emb, labels, os.path.join(save_dir, "graphtransformer_cora_tsne.png"))


if __name__ == "__main__":
    main()
