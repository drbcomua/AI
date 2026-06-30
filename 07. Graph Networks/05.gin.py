"""
05. Graph Isomorphism Network (GIN)
===================================

Expressive Graph Neural Network matching the Weisfeiler-Lehman (WL) graph isomorphism test (Xu et al., 2018).
Evaluated on the MUTAG Molecular Dataset for Supervised Graph Classification.

Architecture:
    Node Embeddings: Input Node Feature [N, 7] -> GINConv -> h1 -> GINConv -> h2 -> GINConv -> h3
    Pooling: Global Sum Pooling (Jumping Knowledge concat: [g1 || g2 || g3])
    Graph Classifier: MLP [Hidden_Dim * 3, Hidden_Dim] -> ReLU -> Dropout -> Linear [Hidden_Dim, 2]

Key insights / educational takeaways:
    * SUM neighborhood aggregation preserves multiset structures, unlike MEAN or MAX which wash out cardinality info.
    * This allows GIN to distinguish basic graph topologies (like rings vs. disjoint triangles) that standard GCNs cannot.

Run:
    python "05.gin.py" --epochs 100
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import gnn_common as mc


class GINConv(nn.Module):
    """GIN Convolution Layer utilizing a multi-layer perceptron (MLP) for updates."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.ReLU()
        )
        # Learnable epsilon parameter
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj):
        # x shape: [N, in_features], adj shape: [N, N] (un-normalized adjacency)
        # Sum neighborhood aggregation: A * X
        h_neigh = torch.matmul(adj, x)
        h_self = (1.0 + self.eps) * x
        return self.mlp(h_self + h_neigh)


class GIN(nn.Module):
    """Graph Isomorphism Network (GIN) Graph Classifier."""
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.conv1 = GINConv(in_dim, hidden_dim)
        self.conv2 = GINConv(hidden_dim, hidden_dim)
        self.conv3 = GINConv(hidden_dim, hidden_dim)

        # Jumping Knowledge concatenation (width: hidden_dim * 3 layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, adj):
        h1 = self.conv1(x, adj)
        h2 = self.conv2(h1, adj)
        h3 = self.conv3(h2, adj)

        # Global Sum Pooling over nodes
        g1 = h1.sum(dim=0)
        g2 = h2.sum(dim=0)
        g3 = h3.sum(dim=0)

        # Concatenate multi-scale representations
        g = torch.cat([g1, g2, g3], dim=-1).unsqueeze(0) # [1, hidden_dim * 3]
        logits = self.classifier(g)
        return logits, g


def main():
    p = mc.build_argparser("Graph Isomorphism Network (GIN) MUTAG Classifier", epochs=100, lr=0.001)
    args = p.parse_args()

    device = mc.get_device(args.device)

    # Load MUTAG dataset
    graphs = mc.load_mutag()
    num_graphs = len(graphs)

    # Set seed for reproducible split
    random.seed(42)
    random.shuffle(graphs)

    # Train/test split (approx 80/20)
    train_graphs = graphs[:150]
    test_graphs = graphs[150:]

    model = GIN(in_dim=7, hidden_dim=32, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("Training Graph Isomorphism Network (GIN) on MUTAG...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0

        # Since MUTAG graphs have variable sizes, we iterate graph-by-graph (batch_size=1)
        for g in train_graphs:
            x = g["x"].to(device)
            # GIN uses raw un-normalized adjacency to preserve sums
            # Reconstruct un-normalized A from normalized A:
            adj = (g["adj"] > 0).float().to(device)
            # Remove self-loops to count only neighbors in sum
            adj = adj - torch.eye(adj.size(0), device=device)
            adj = torch.clamp(adj, 0.0, 1.0)

            y = torch.tensor([g["y"]], dtype=torch.long, device=device)

            optimizer.zero_grad()
            out, _ = model(x, adj)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = out.argmax(dim=-1).item()
            if pred == g["y"]:
                train_correct += 1

        train_acc = train_correct / len(train_graphs)
        train_loss_avg = train_loss / len(train_graphs)

        # Periodic Validation
        if epoch % max(1, args.epochs // 10) == 0:
            model.eval()
            test_correct = 0
            with torch.no_grad():
                for g in test_graphs:
                    x = g["x"].to(device)
                    adj = (g["adj"] > 0).float().to(device)
                    adj = adj - torch.eye(adj.size(0), device=device)
                    adj = torch.clamp(adj, 0.0, 1.0)

                    out, _ = model(x, adj)
                    pred = out.argmax(dim=-1).item()
                    if pred == g["y"]:
                        test_correct += 1
            test_acc = test_correct / len(test_graphs)

            print(f"Epoch {epoch:3d}/{args.epochs} | train_loss {train_loss_avg:.4f} | train_acc {train_acc:.4f} | test_acc {test_acc:.4f}")

    print("-" * 64)

    # Final Test evaluation
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for g in test_graphs:
            x = g["x"].to(device)
            adj = (g["adj"] > 0).float().to(device)
            adj = adj - torch.eye(adj.size(0), device=device)
            adj = torch.clamp(adj, 0.0, 1.0)

            out, _ = model(x, adj)
            pred = out.argmax(dim=-1).item()
            if pred == g["y"]:
                test_correct += 1
    test_acc = test_correct / len(test_graphs)
    print(f"GIN Graph Classification Test Accuracy (MUTAG): {test_acc:.4f}")

    # Gather graph-level embeddings for all graphs to perform t-SNE visualization
    print("Gathering graph-level molecular embeddings...")
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for g in graphs:
            x = g["x"].to(device)
            adj = (g["adj"] > 0).float().to(device)
            adj = adj - torch.eye(adj.size(0), device=device)
            adj = torch.clamp(adj, 0.0, 1.0)
            _, g_emb = model(x, adj)
            all_embeddings.append(g_emb.cpu())
            all_labels.append(g["y"])

    all_embeddings = torch.cat(all_embeddings, dim=0) # [188, hidden_dim * 3]
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    # Plot and save t-SNE of graph molecular representations
    save_dir = os.path.dirname(os.path.abspath(__file__))
    tsne_path = os.path.join(save_dir, "gin_mutag_tsne.png")
    mc.plot_tsne_embeddings(
        all_embeddings, all_labels, tsne_path,
        title="t-SNE Projection of GIN Graph-Level Molecular Embeddings (MUTAG)",
        legend_title="Mutagenicity (0=No, 1=Yes)"
    )

if __name__ == "__main__":
    main()
