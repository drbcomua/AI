"""
11. ChebNet — Spectral Graph Convolution (Defferrard et al., 2016)
=================================================================

The spectral predecessor to GCN. A graph convolution is a filter on the graph
Laplacian's eigenbasis; computing eigenvectors is expensive, so ChebNet
approximates the filter with a truncated **Chebyshev polynomial** of the scaled
Laplacian, which only needs sparse matrix-vector products and is exactly
K-localized (touches K-hop neighborhoods):

    g_theta(L) x  ~=  sum_{k=0}^{K-1} theta_k T_k(L_hat) x
    T_0 = I,  T_1 = L_hat,  T_k = 2 L_hat T_{k-1} - T_{k-2}
    L_hat = (2 / lambda_max) L - I    (here L = I - D^-1/2 A D^-1/2, lambda_max ~ 2)

GCN is the special case K=2 with further simplifications — so this script is the
more general filter that GCN descends from.

Run:
    python "11.chebnet.py" --epochs 200
"""

import os
import torch
import torch.nn as nn
import gnn_common as mc


class ChebConv(nn.Module):
    def __init__(self, in_dim, out_dim, K=3):
        super().__init__()
        self.K = K
        self.weight = nn.Linear(in_dim * K, out_dim)     # one filter weight per Chebyshev order

    def forward(self, x, l_hat):
        tx = [x, l_hat @ x] if self.K > 1 else [x]
        for _ in range(2, self.K):
            tx.append(2 * (l_hat @ tx[-1]) - tx[-2])     # Chebyshev recurrence
        return self.weight(torch.cat(tx[:self.K], dim=-1))


class ChebNet(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, K=3, dropout=0.5):
        super().__init__()
        self.conv1 = ChebConv(in_dim, hidden, K)
        self.conv2 = ChebConv(hidden, num_classes, K)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, l_hat):
        h = torch.relu(self.conv1(x, l_hat))
        return self.conv2(self.dropout(h), l_hat), h


def main():
    args = mc.build_argparser("ChebNet (spectral GCN)", epochs=200).parse_args()
    device = mc.get_device(args.device)

    features, adj_raw, labels = mc.load_cora_raw()
    if args.limit:
        features, adj_raw, labels = features[:args.limit], adj_raw[:args.limit, :args.limit], labels[:args.limit]
    features, adj_raw, labels = features.to(device), adj_raw.to(device), labels.to(device)
    _, _, _, splits = mc.load_cora(limit=args.limit)
    train_mask = splits["train_mask"].to(device)
    val_mask = splits["val_mask"].to(device)
    test_mask = splits["test_mask"].to(device)

    # Scaled normalized Laplacian: L = I - D^-1/2 A D^-1/2 ; lambda_max ~ 2 => L_hat = L - I
    a_norm = mc.normalize_adj(adj_raw, self_loops=False)
    n = a_norm.size(0)
    L = torch.eye(n, device=device) - a_norm
    l_hat = L - torch.eye(n, device=device)

    in_dim, num_classes = features.size(1), int(labels.max().item() + 1)
    model = ChebNet(in_dim, 32, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    print(f"Training ChebNet (K=3) on Cora | params: {sum(q.numel() for q in model.parameters()):,}")
    print("-" * 64)
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad()
        out, _ = model(features, l_hat)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward(); optimizer.step()
        if epoch % max(1, args.epochs // 10) == 0:
            model.eval()
            with torch.no_grad():
                logits, _ = model(features, l_hat)
                val_acc = (logits.argmax(1)[val_mask] == labels[val_mask]).float().mean().item()
            print(f"Epoch {epoch:3d}/{args.epochs} | train_loss {loss.item():.4f} | val_acc {val_acc:.4f}")
    print("-" * 64)

    model.eval()
    with torch.no_grad():
        logits, emb = model(features, l_hat)
        test_acc = (logits.argmax(1)[test_mask] == labels[test_mask]).float().mean().item()
    print(f"ChebNet Test Accuracy: {test_acc:.4f}")

    save_dir = os.path.dirname(os.path.abspath(__file__))
    mc.plot_tsne_embeddings(emb, labels, os.path.join(save_dir, "chebnet_cora_tsne.png"))


if __name__ == "__main__":
    main()
