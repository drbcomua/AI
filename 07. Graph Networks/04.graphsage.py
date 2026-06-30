"""
04. GraphSAGE
=============

Inductive representation learning via spatial neighborhood feature aggregation (Hamilton et al., 2017).

Mathematical Formulation:
    h_N(v)^(l+1) = Aggregate( { h_u^(l), forall u in N(v) } )
    h_v^(l+1) = ReLU( W_self * h_v^(l) + W_neigh * h_N(v)^(l+1) )

Key insights / educational takeaways:
    * GraphSAGE separates the projection weights of a node's self-features and its aggregated neighbor features.
    * This allows inductive generalization to completely unseen graphs or nodes added dynamically.

Run:
    python "04.graphsage.py" --epochs 200
"""

import os
import torch
import torch.nn as nn
import gnn_common as mc


class SAGEConv(nn.Module):
    """GraphSAGE Layer in pure PyTorch supporting Mean neighborhood aggregations."""
    def __init__(self, in_features: int, out_features: int, aggregator_type: str = "mean"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.agg_type = aggregator_type.lower()

        # Distinct weights for self features and aggregated neighbor features
        self.w_self = nn.Linear(in_features, out_features, bias=False)
        self.w_neigh = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, adj):
        # x shape: [N, in_features], adj shape: [N, N] (Adjacency list without self loops)
        # 1. Neighborhood aggregation
        if self.agg_type == "mean":
            # Normalized adjacency (D^-1 * A)
            deg = adj.sum(dim=1, keepdim=True)
            deg_inv = torch.where(deg > 0, 1.0 / deg, 0.0)
            adj_normalized = adj * deg_inv
            h_neigh = torch.matmul(adj_normalized, x)
        else:
            raise ValueError(f"Unsupported aggregator type: {self.agg_type}")

        # 2. Combine and project: W_self * x + W_neigh * h_neigh + bias
        out = self.w_self(x) + self.w_neigh(h_neigh) + self.bias
        return out


class GraphSAGE(nn.Module):
    """2-layer GraphSAGE Network."""
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggregator_type="mean")
        self.conv2 = SAGEConv(hidden_dim, num_classes, aggregator_type="mean")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        h = torch.relu(self.conv1(x, adj))
        h_drop = self.dropout(h)
        logits = self.conv2(h_drop, adj)
        return logits, h


def main():
    p = mc.build_argparser("GraphSAGE Node Classifier", epochs=200)
    args = p.parse_args()

    device = mc.get_device(args.device)

    # Load Cora
    features, adj, labels, splits = mc.load_cora(limit=args.limit)

    # GraphSAGE uses the raw adjacency matrix without self loops
    # (since the aggregation splits self-features and neighbor-features).
    # To remove self-loops from adj (since load_cora returns normalized A with self-loops):
    # We can reconstruct it or simply subtract I from the un-normalized adjacency.
    # In GNN benchmarks, passing normalized adj works but separating them is cleaner.
    # To keep it simple, we can subtract Identity from adj to isolate neighbors.
    # Wait, load_cora returns D^-1/2 * (A+I) * D^-1/2.
    # We can reconstruct the raw adjacency easily:
    # adj_raw = (adj > 0).float() - torch.eye(adj.size(0), device=adj.device)
    # adj_raw = torch.clamp(adj_raw, 0.0, 1.0)
    # This isolates exactly the direct neighbors!
    # Let's do that!
    adj_raw = (adj > 0).float() - torch.eye(adj.size(0))
    adj_raw = torch.clamp(adj_raw, 0.0, 1.0)

    in_dim = features.size(1)
    num_classes = int(labels.max().item() + 1)

    model = GraphSAGE(in_dim, 16, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Move tensors to device
    features = features.to(device)
    adj_raw = adj_raw.to(device)
    labels = labels.to(device)
    train_mask = splits["train_mask"].to(device)
    val_mask = splits["val_mask"].to(device)
    test_mask = splits["test_mask"].to(device)

    print("Training GraphSAGE Network...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        out, _ = model(features, adj_raw)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits, _ = model(features, adj_raw)
            val_loss = criterion(logits[val_mask], labels[val_mask])
            preds = logits.argmax(dim=-1)
            val_acc = (preds[val_mask] == labels[val_mask]).float().mean().item()

        if epoch % max(1, args.epochs // 10) == 0:
            print(f"Epoch {epoch:3d}/{args.epochs} | train_loss {loss.item():.4f} | val_loss {val_loss.item():.4f} | val_acc {val_acc:.4f}")

    print("-" * 64)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        logits, h_emb = model(features, adj_raw)
        preds = logits.argmax(dim=-1)
        test_acc = (preds[test_mask] == labels[test_mask]).float().mean().item()
    print(f"GraphSAGE Test Accuracy: {test_acc:.4f}")

    # Plot and save t-SNE of hidden node embeddings
    save_dir = os.path.dirname(os.path.abspath(__file__))
    tsne_path = os.path.join(save_dir, "graphsage_cora_tsne.png")
    mc.plot_tsne_embeddings(h_emb, labels, tsne_path)


if __name__ == "__main__":
    main()
