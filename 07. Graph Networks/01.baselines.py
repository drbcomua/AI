"""
01. MLP & Label Propagation Baselines
=====================================

Baseline models for Cora Node Classification to benchmark isolated features vs. isolated structure.

Models:
    * MLP: Predicts node class using only local word bag features, ignoring citation links.
    * Label Propagation: Propagates known labels across neighbors using graph structure, ignoring word features.

Run:
    python "01.baselines.py" --variant mlp --epochs 150
    python "01.baselines.py" --variant label_prop
"""

import os
import torch
import torch.nn as nn
import gnn_common as mc


class NodeMLP(nn.Module):
    """Simple Multi-Layer Perceptron node classifier ignoring graph structure."""
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h_drop = self.dropout(h)
        logits = self.fc2(h_drop)
        return logits, h


def run_mlp(features, labels, splits, device, epochs, lr):
    """Train and evaluate the MLP feature baseline."""
    in_dim = features.size(1)
    num_classes = int(labels.max().item() + 1)

    model = NodeMLP(in_dim, 64, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    features = features.to(device)
    labels = labels.to(device)
    train_mask = splits["train_mask"].to(device)
    val_mask = splits["val_mask"].to(device)
    test_mask = splits["test_mask"].to(device)

    print("Training Node MLP Baseline...")
    print("-" * 64)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out, _ = model(features)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            logits, _ = model(features)
            val_loss = criterion(logits[val_mask], labels[val_mask])
            preds = logits.argmax(dim=-1)
            val_acc = (preds[val_mask] == labels[val_mask]).float().mean().item()

        if epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch:3d}/{epochs} | train_loss {loss.item():.4f} | val_loss {val_loss.item():.4f} | val_acc {val_acc:.4f}")

    print("-" * 64)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        logits, h_emb = model(features)
        preds = logits.argmax(dim=-1)
        test_acc = (preds[test_mask] == labels[test_mask]).float().mean().item()
    print(f"Node MLP Test Accuracy: {test_acc:.4f}")

    # Plot and save t-SNE of hidden node embeddings
    save_dir = os.path.dirname(os.path.abspath(__file__))
    tsne_path = os.path.join(save_dir, "mlp_cora_tsne.png")
    mc.plot_tsne_embeddings(
        h_emb, labels, tsne_path,
        title="t-SNE Projection of Cora MLP Node Embeddings (No Structure)",
        legend_title="Subject Categories"
    )


def run_label_propagation(adj, labels, splits, iterations=50):
    """Evaluate structure-only Label Propagation baseline."""
    num_nodes = adj.size(0)
    num_classes = int(labels.max().item() + 1)

    # Initialize soft labels matrix [N, C]
    Y = torch.zeros(num_nodes, num_classes)
    train_mask = splits["train_mask"]
    test_mask = splits["test_mask"]

    # Fill in one-hot labels for training nodes
    train_labels = labels[train_mask]
    Y[train_mask, train_labels] = 1.0

    # Propagate labels iteratively: Y = A * Y
    for _ in range(iterations):
        Y = torch.matmul(adj, Y)
        # Clamp training labels back to their true values (boundary condition)
        Y[train_mask] = 0.0
        Y[train_mask, train_labels] = 1.0

    # Predict
    preds = Y.argmax(dim=-1)
    test_acc = (preds[test_mask] == labels[test_mask]).float().mean().item()
    print(f"Label Propagation Test Accuracy: {test_acc:.4f}")


def main():
    p = mc.build_argparser("MLP and Label Propagation Cora Baselines", epochs=150)
    args = p.parse_args()

    variant = args.variant or "mlp"
    if variant not in ["mlp", "label_prop"]:
        variant = "mlp"

    device = mc.get_device(args.device)

    # Load Cora
    features, adj, labels, splits = mc.load_cora(limit=args.limit)

    if variant == "mlp":
        run_mlp(features, labels, splits, device, args.epochs, args.lr)
    else:
        run_label_propagation(adj, labels, splits)


if __name__ == "__main__":
    main()
