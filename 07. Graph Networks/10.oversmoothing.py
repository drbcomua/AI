"""
10. Oversmoothing — why deep GCNs fail, and how GCNII fixes it
=============================================================

The signature pathology of GNNs. Each GCN layer averages a node with its
neighbors; stack too many and *every* node's representation converges to the same
vector (the graph's stationary distribution), erasing the information that
distinguishes classes. So plain GCN accuracy *peaks at 2-3 layers and then
collapses* with depth.

**GCNII** (Chen et al., 2020) fixes this with two ideas per layer:
    * Initial residual:  mix in the layer-0 features H0 every layer
                         P = (1-alpha)·A·H + alpha·H0
    * Identity mapping:   keep the weight close to identity
                         H' = sigma( (1-beta)·P + beta·W·P ),  beta = log(lambda/l + 1)
These let GCNII go 16-64 layers deep while *improving* with depth.

This script trains both models at depths {2,4,8,16,32} and plots test accuracy vs
depth — the classic oversmoothing curve.

Run:
    python "10.oversmoothing.py" --epochs 100
"""

import os
import math
import torch
import torch.nn as nn
import gnn_common as mc


class DeepGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, depth, dropout=0.5):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth)])
        self.out = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        h = torch.relu(self.inp(x))
        for layer in self.layers:
            h = torch.relu(layer(adj @ h))            # plain message passing -> oversmooths
            h = self.dropout(h)
        return self.out(h)


class GCNII(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, depth, alpha=0.1, lamb=0.5, dropout=0.5):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.weights = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(depth)])
        self.out = nn.Linear(hidden, num_classes)
        self.alpha, self.lamb = alpha, lamb
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        h0 = torch.relu(self.inp(x))
        h = h0
        for l, w in enumerate(self.weights, start=1):
            beta = math.log(self.lamb / l + 1)
            p = (1 - self.alpha) * (adj @ h) + self.alpha * h0     # initial residual
            h = torch.relu((1 - beta) * p + beta * w(p))           # identity mapping
            h = self.dropout(h)
        return self.out(h)


def train_eval(model, features, adj, labels, masks, epochs, lr):
    train_mask, val_mask, test_mask = masks
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = model(features, adj)
        ce(out[train_mask], labels[train_mask]).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(features, adj).argmax(1)
        return (pred[test_mask] == labels[test_mask]).float().mean().item()


def main():
    args = mc.build_argparser("Oversmoothing: deep GCN vs GCNII", epochs=100).parse_args()
    device = mc.get_device(args.device)
    features, adj, labels, splits = mc.load_cora(limit=args.limit)
    features, adj, labels = features.to(device), adj.to(device), labels.to(device)
    masks = (splits["train_mask"].to(device), splits["val_mask"].to(device), splits["test_mask"].to(device))
    in_dim, num_classes = features.size(1), int(labels.max().item() + 1)

    depths = [2, 4, 8, 16, 32]
    results = {"GCN": [], "GCNII": []}
    print("Sweeping depth for plain GCN vs GCNII on Cora...")
    print("-" * 64)
    for depth in depths:
        gcn = DeepGCN(in_dim, 64, num_classes, depth).to(device)
        gcnii = GCNII(in_dim, 64, num_classes, depth).to(device)
        acc_gcn = train_eval(gcn, features, adj, labels, masks, args.epochs, args.lr)
        acc_gcnii = train_eval(gcnii, features, adj, labels, masks, args.epochs, args.lr)
        results["GCN"].append(acc_gcn)
        results["GCNII"].append(acc_gcnii)
        print(f"depth {depth:2d} | GCN test_acc {acc_gcn:.4f} | GCNII test_acc {acc_gcnii:.4f}")
    print("-" * 64)
    print("Note how plain GCN collapses with depth while GCNII keeps improving.")

    # Plot the oversmoothing curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(depths, results["GCN"], "o-", label="Plain GCN (oversmooths)")
        ax.plot(depths, results["GCNII"], "s-", label="GCNII (residual + identity)")
        ax.set_xlabel("Number of layers (depth)")
        ax.set_ylabel("Cora test accuracy")
        ax.set_title("Oversmoothing: accuracy vs GNN depth")
        ax.set_xscale("log", base=2); ax.set_xticks(depths); ax.set_xticklabels(depths)
        ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oversmoothing_cora_depth.png")
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"Saved oversmoothing curve -> {out}")
    except Exception as e:
        print(f"(skipping plot: {e})")


if __name__ == "__main__":
    main()
