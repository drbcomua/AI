"""
02. Graph Convolutional Network (GCN)
=====================================

Spatially localized spectral graph convolutions (Kipf & Welling, 2016).

Mathematical Formulation:
    H^(l+1) = ReLU( D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2) * H^(l) * W^(l) )
    where A_tilde = A + I (Adjacency with self-loops) and D_tilde is the degree matrix of A_tilde.

Key insights / educational takeaways:
    * Adding self-loops prevents a node's own features from being excluded during neighborhood aggregation.
    * Symmetric degree normalization prevents features from exploding for highly-connected hub nodes.

Run:
    python "02.gcn.py" --epochs 200
"""

import os
import torch
import torch.nn as nn
import gnn_common as mc


class GCNConv(nn.Module):
    """A standard Graph Convolutional layer in pure PyTorch."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x, adj):
        # x shape: [N, in_features], adj shape: [N, N]
        h = self.linear(x) # [N, out_features]
        # Aggregate neighbor features: D^-1/2 * A * D^-1/2 * H
        out = torch.matmul(adj, h)
        if self.bias is not None:
            out = out + self.bias
        return out


class GCN(nn.Module):
    """2-layer Graph Convolutional Network."""
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        h = torch.relu(self.conv1(x, adj))
        h_drop = self.dropout(h)
        logits = self.conv2(h_drop, adj)
        return logits, h


def main():
    p = mc.build_argparser("Graph Convolutional Network (GCN)", epochs=200)
    args = p.parse_args()

    device = mc.get_device(args.device)

    # Load Cora
    features, adj, labels, splits = mc.load_cora(limit=args.limit)

    in_dim = features.size(1)
    num_classes = int(labels.max().item() + 1)

    model = GCN(in_dim, 16, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Move tensors to device
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    train_mask = splits["train_mask"].to(device)
    val_mask = splits["val_mask"].to(device)
    test_mask = splits["test_mask"].to(device)

    print("Training Graph Convolutional Network (GCN)...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        out, _ = model(features, adj)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            logits, _ = model(features, adj)
            val_loss = criterion(logits[val_mask], labels[val_mask])
            preds = logits.argmax(dim=-1)
            val_acc = (preds[val_mask] == labels[val_mask]).float().mean().item()

        if epoch % max(1, args.epochs // 10) == 0:
            print(f"Epoch {epoch:3d}/{args.epochs} | train_loss {loss.item():.4f} | val_loss {val_loss.item():.4f} | val_acc {val_acc:.4f}")

    print("-" * 64)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        logits, h_emb = model(features, adj)
        preds = logits.argmax(dim=-1)
        test_acc = (preds[test_mask] == labels[test_mask]).float().mean().item()
    print(f"GCN Test Accuracy: {test_acc:.4f}")

    # Plot and save t-SNE of hidden node embeddings
    save_dir = os.path.dirname(os.path.abspath(__file__))
    tsne_path = os.path.join(save_dir, "gcn_cora_tsne.png")
    mc.plot_tsne_embeddings(h_emb, labels, tsne_path)


if __name__ == "__main__":
    main()
