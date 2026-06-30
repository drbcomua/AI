"""
09. SGC & APPNP — decoupling propagation from transformation
============================================================

Two influential "is all that machinery necessary?" models, selectable via
--variant:

    --variant sgc     Simple Graph Convolution (Wu et al., 2019). Collapse a
                      K-layer GCN by removing every nonlinearity: precompute
                      S = A_norm^K, then a single linear classifier on S·X.
                      Shows a GCN is essentially a low-pass graph filter followed
                      by logistic regression — and it nearly matches GCN on Cora.

    --variant appnp   Approximate Personalized Propagation of Neural Predictions
                      (Klicpera et al., 2019). First predict with an MLP on each
                      node's own features, THEN propagate those predictions with
                      personalized PageRank: Z <- (1-a)·A_norm·Z + a·H. Decoupling
                      lets you propagate many hops without oversmoothing or extra
                      trainable layers.

Run:
    python "09.sgc-appnp.py" --variant sgc --epochs 100
    python "09.sgc-appnp.py" --variant appnp --epochs 200
"""

import os
import torch
import torch.nn as nn
import gnn_common as mc


class SGC(nn.Module):
    def __init__(self, in_dim, num_classes, K=2):
        super().__init__()
        self.K = K
        self.linear = nn.Linear(in_dim, num_classes)

    def precompute(self, x, adj):
        for _ in range(self.K):                      # S = A_norm^K · X (no nonlinearity)
            x = adj @ x
        return x

    def forward(self, sx):
        return self.linear(sx), sx


class APPNP(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, K=10, alpha=0.1, dropout=0.5):
        super().__init__()
        self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, num_classes))
        self.K, self.alpha = K, alpha

    def forward(self, x, adj):
        h = self.mlp(x)                              # per-node prediction first
        z = h
        for _ in range(self.K):                      # personalized PageRank propagation
            z = (1 - self.alpha) * (adj @ z) + self.alpha * h
        return z, z


def main():
    p = mc.build_argparser("SGC / APPNP", epochs=100)
    args = p.parse_args()
    variant = args.variant or "sgc"
    device = mc.get_device(args.device)

    features, adj, labels, splits = mc.load_cora(limit=args.limit)
    features, adj, labels = features.to(device), adj.to(device), labels.to(device)
    train_mask = splits["train_mask"].to(device)
    val_mask = splits["val_mask"].to(device)
    test_mask = splits["test_mask"].to(device)
    in_dim, num_classes = features.size(1), int(labels.max().item() + 1)

    if variant == "sgc":
        model = SGC(in_dim, num_classes).to(device)
        sx = model.precompute(features, adj)         # one-time propagation
        forward = lambda: model(sx)
    elif variant == "appnp":
        model = APPNP(in_dim, 64, num_classes).to(device)
        forward = lambda: model(features, adj)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    print(f"Training {variant.upper()} on Cora | params: {sum(q.numel() for q in model.parameters()):,}")
    print("-" * 64)
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad()
        out, _ = forward()
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward(); optimizer.step()
        if epoch % max(1, args.epochs // 10) == 0:
            model.eval()
            with torch.no_grad():
                logits, _ = forward()
                val_acc = (logits.argmax(1)[val_mask] == labels[val_mask]).float().mean().item()
            print(f"Epoch {epoch:3d}/{args.epochs} | train_loss {loss.item():.4f} | val_acc {val_acc:.4f}")
    print("-" * 64)

    model.eval()
    with torch.no_grad():
        logits, emb = forward()
        test_acc = (logits.argmax(1)[test_mask] == labels[test_mask]).float().mean().item()
    print(f"{variant.upper()} Test Accuracy: {test_acc:.4f}")

    save_dir = os.path.dirname(os.path.abspath(__file__))
    mc.plot_tsne_embeddings(emb, labels, os.path.join(save_dir, f"{variant}_cora_tsne.png"))


if __name__ == "__main__":
    main()
