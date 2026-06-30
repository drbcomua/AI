"""
06. Graph Autoencoders — GAE & VGAE for Link Prediction (Kipf & Welling, 2016)
==============================================================================

The third core graph task — **link prediction** — plus the unsupervised /
generative paradigm. A GCN encoder maps each node to an embedding z; the decoder
reconstructs the adjacency from inner products, A_hat = sigmoid(Z Z^T). Trained to
reproduce the observed edges, the embeddings place connected nodes nearby, so
their inner product *scores* whether a held-out (or future) link exists.

    --variant gae    Deterministic autoencoder (Z = GCN(X, A)).
    --variant vgae   Variational: GCN outputs mu and log-sigma; z is sampled and a
                     KL term regularizes the latent space toward N(0, I).

Setup: 10% of edges are removed and held out; we report ROC-AUC and Average
Precision at distinguishing those true edges from sampled non-edges (the standard
link-prediction protocol). The encoder only ever sees the remaining 90%.

Run:
    python "06.gae-vgae.py" --variant gae --epochs 200
    python "06.gae-vgae.py" --variant vgae --epochs 200
"""

import os
import numpy as np
import torch
import torch.nn as nn
import gnn_common as mc


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, adj):
        return adj @ self.lin(x)


class GAE(nn.Module):
    def __init__(self, in_dim, hidden, latent, variational=False):
        super().__init__()
        self.variational = variational
        self.gc1 = GCNLayer(in_dim, hidden)
        self.gc_mu = GCNLayer(hidden, latent)
        if variational:
            self.gc_logstd = GCNLayer(hidden, latent)

    def encode(self, x, adj):
        h = torch.relu(self.gc1(x, adj))
        mu = self.gc_mu(h, adj)
        if not self.variational:
            return mu, None, None
        logstd = self.gc_logstd(h, adj)
        z = mu + torch.randn_like(mu) * torch.exp(logstd) if self.training else mu
        return z, mu, logstd

    def forward(self, x, adj):
        z, mu, logstd = self.encode(x, adj)
        return z, mu, logstd


def score_edges(z, edges):
    s = (z[edges[:, 0]] * z[edges[:, 1]]).sum(dim=1)
    return torch.sigmoid(s)


def main():
    p = mc.build_argparser("Graph Autoencoder (GAE / VGAE) link prediction", epochs=200, lr=1e-2)
    args = p.parse_args()
    variant = args.variant or "gae"
    device = mc.get_device(args.device)

    features, adj_raw, labels = mc.load_cora_raw()
    train_adj, test_pos, test_neg = mc.split_link_prediction_edges(adj_raw, test_frac=0.1)
    adj_norm = mc.normalize_adj(train_adj, self_loops=True).to(device)
    features = features.to(device)
    target = train_adj.to(device)                          # reconstruction target (no self-loops)
    test_pos = torch.tensor(test_pos, dtype=torch.long, device=device)
    test_neg = torch.tensor(test_neg, dtype=torch.long, device=device)

    n = features.size(0)
    edges = target.sum()
    pos_weight = (n * n - edges) / edges                   # handle edge sparsity in BCE
    norm = n * n / float((n * n - edges) * 2)

    model = GAE(features.size(1), 32, 16, variational=(variant == "vgae")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Training {variant.upper()} on Cora link prediction | "
          f"held-out edges: {len(test_pos)} | params: {sum(q.numel() for q in model.parameters()):,}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad()
        z, mu, logstd = model(features, adj_norm)
        logits = z @ z.t()
        recon = norm * nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight)
        loss = recon
        if variant == "vgae":
            kl = -0.5 / n * torch.mean(torch.sum(1 + 2 * logstd - mu ** 2 - torch.exp(logstd) ** 2, dim=1))
            loss = recon + kl
        loss.backward(); optimizer.step()
        if epoch % max(1, args.epochs // 10) == 0:
            print(f"Epoch {epoch:3d}/{args.epochs} | loss {loss.item():.4f}")
    print("-" * 64)

    # Link-prediction metrics
    model.eval()
    with torch.no_grad():
        z, _, _ = model(features, adj_norm)
        pos_score = score_edges(z, test_pos).cpu().numpy()
        neg_score = score_edges(z, test_neg).cpu().numpy()
    y_true = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
    y_score = np.concatenate([pos_score, neg_score])
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        print(f"{variant.upper()} Link Prediction | ROC-AUC: {roc_auc_score(y_true, y_score):.4f} | "
              f"Average Precision: {average_precision_score(y_true, y_score):.4f}")
    except Exception as e:
        print(f"(skipping metrics: {e})")

    save_dir = os.path.dirname(os.path.abspath(__file__))
    mc.plot_tsne_embeddings(z, labels.to(device), os.path.join(save_dir, f"{variant}_cora_tsne.png"),
                            title=f"t-SNE of {variant.upper()} node embeddings (Cora)")


if __name__ == "__main__":
    main()
