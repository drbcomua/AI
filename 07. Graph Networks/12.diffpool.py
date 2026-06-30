"""
12. DiffPool — Differentiable Hierarchical Graph Pooling (Ying et al., 2018)
===========================================================================

GIN reads out a whole graph by a single flat sum over all nodes. DiffPool instead
*coarsens* the graph hierarchically, the way a CNN downsamples an image. At each
pooling step a GNN produces a soft **assignment matrix** S (nodes -> clusters),
and the graph is collapsed:

    Z  = GNN_embed(X, A)            (node embeddings)
    S  = softmax(GNN_pool(X, A))    (N x C soft cluster assignments)
    X' = S^T Z                      (coarsened cluster features)
    A' = S^T A S                    (coarsened cluster adjacency)

Stacking these learns a hierarchy of ever-coarser graphs ending in a single
super-node for classification. Two auxiliary losses keep the assignments sane: a
link-prediction loss (clusters should respect connectivity, ||A - S S^T||) and an
entropy loss (each node should commit to a cluster).

Evaluated on MUTAG graph classification.

Run:
    python "12.diffpool.py" --epochs 100
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import gnn_common as mc


class GNN(nn.Module):
    """Two-layer GCN block operating on a dense (soft) adjacency."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, out_dim)
        self.l2 = nn.Linear(out_dim, out_dim)

    def forward(self, x, adj):
        h = torch.relu(adj @ self.l1(x))
        return torch.relu(adj @ self.l2(h))


class DiffPool(nn.Module):
    def __init__(self, in_dim, hidden=32, n_clusters=4, num_classes=2):
        super().__init__()
        self.embed = GNN(in_dim, hidden)
        self.assign = GNN(in_dim, n_clusters)              # produces cluster assignment logits
        self.embed2 = GNN(hidden, hidden)
        self.classifier = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                        nn.Dropout(0.5), nn.Linear(hidden, num_classes))

    def forward(self, x, adj):
        z = self.embed(x, adj)                             # [N, hidden]
        s = torch.softmax(self.assign(x, adj), dim=1)      # [N, C]
        x_pool = s.t() @ z                                 # [C, hidden]
        adj_pool = s.t() @ adj @ s                         # [C, C]
        # auxiliary losses
        link_loss = torch.norm(adj - s @ s.t(), p="fro") / adj.numel()
        ent_loss = (-s * torch.log(s + 1e-12)).sum(1).mean()
        # second GNN on the coarsened graph, then global mean over clusters
        h = self.embed2(x_pool, adj_pool)
        g = h.mean(dim=0, keepdim=True)                    # [1, hidden]
        return self.classifier(g), link_loss + ent_loss, g


def run_split(model, graphs, device, train=False, optimizer=None, criterion=None):
    correct = 0
    for g in graphs:
        x, adj, y = g["x"].to(device), g["adj"].to(device), torch.tensor([g["y"]], device=device)
        out, aux, _ = model(x, adj)
        if train:
            loss = criterion(out, y) + 0.1 * aux
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        correct += int(out.argmax(1).item() == g["y"])
    return correct / len(graphs)


def main():
    args = mc.build_argparser("DiffPool (hierarchical pooling) on MUTAG", epochs=100, lr=1e-3).parse_args()
    device = mc.get_device(args.device)
    graphs = mc.load_mutag()
    random.seed(42); random.shuffle(graphs)
    train_graphs, test_graphs = graphs[:150], graphs[150:]

    model = DiffPool(in_dim=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    print(f"Training DiffPool on MUTAG | params: {sum(q.numel() for q in model.parameters()):,}")
    print("-" * 64)
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_graphs)
        train_acc = run_split(model, train_graphs, device, train=True, optimizer=optimizer, criterion=criterion)
        if epoch % max(1, args.epochs // 10) == 0:
            model.eval()
            with torch.no_grad():
                test_acc = run_split(model, test_graphs, device)
            print(f"Epoch {epoch:3d}/{args.epochs} | train_acc {train_acc:.4f} | test_acc {test_acc:.4f}")
    print("-" * 64)

    model.eval()
    with torch.no_grad():
        test_acc = run_split(model, test_graphs, device)
        embs, labels = [], []
        for g in graphs:
            _, _, emb = model(g["x"].to(device), g["adj"].to(device))
            embs.append(emb.cpu()); labels.append(g["y"])
    print(f"DiffPool Graph Classification Test Accuracy (MUTAG): {test_acc:.4f}")

    save_dir = os.path.dirname(os.path.abspath(__file__))
    mc.plot_tsne_embeddings(torch.cat(embs, 0), torch.tensor(labels), os.path.join(save_dir, "diffpool_mutag_tsne.png"),
                            title="t-SNE of DiffPool graph embeddings (MUTAG)",
                            legend_title="Mutagenicity (0=No, 1=Yes)")


if __name__ == "__main__":
    main()
