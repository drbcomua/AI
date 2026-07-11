"""
07. LightGCN (Light Graph Convolutional Network)
================================================

Graph collaborative filtering stripped to its essentials (He et al., 2020).

Standard GCNs for recommendation inherited feature transformations and
non-linear activations from node-classification GNNs. LightGCN shows these hurt
collaborative filtering: the only useful operation is **neighborhood
aggregation** on the user-item bipartite graph. Each layer simply propagates
embeddings across the symmetrically-normalized adjacency; the final embedding is
the mean over all layers, smoothing each node with its multi-hop neighbors.

Trained with the BPR pairwise ranking loss on implicit feedback and evaluated by
full top-K ranking (Recall@K / NDCG@K), same protocol as `08.mult-vae.py`.

Architecture Diagram / Layout:
    E^(0) = [user_emb ; item_emb]            (only learnable parameters)
    E^(k+1) = D^{-1/2} A D^{-1/2} E^(k)       (no weights, no activation)
    E_final = mean(E^(0), E^(1), ..., E^(K))
    score(u, i) = e_u . e_i

Key insights / educational takeaways:
    * Removing feature transforms and non-linearities both simplifies the model
      and improves accuracy — the graph smoothing is what matters for CF.
    * Layer-combination (averaging across depths) mitigates over-smoothing that
      would otherwise wash out embeddings after too many propagation hops.
    * BPR optimizes the *ranking* of a positive item above a sampled negative,
      the natural objective for implicit feedback.

Run:
    python "07.lightgcn.py" --epochs 200
    python "07.lightgcn.py" --limit 5000 --epochs 20   # fast smoke test
"""

import os
import numpy as np
import torch
import torch.nn as nn
import rec_common as mc


class LightGCN(nn.Module):
    """LightGCN with layer-combination over a dense normalized adjacency."""
    def __init__(self, num_users: int, num_items: int, embed_dim: int = 64,
                 num_layers: int = 3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers

        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        nn.init.normal_(self.user_embed.weight, std=0.1)
        nn.init.normal_(self.item_embed.weight, std=0.1)

    def propagate(self, norm_adj):
        """Runs K graph-convolution layers, returns final user/item embeddings."""
        all_emb = torch.cat([self.user_embed.weight, self.item_embed.weight], dim=0)
        embs = [all_emb]
        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb) if norm_adj.is_sparse \
                else norm_adj @ all_emb
            embs.append(all_emb)
        final = torch.stack(embs, dim=0).mean(dim=0)     # layer combination
        users, items = torch.split(final, [self.num_users, self.num_items])
        return users, items

    def bpr_loss(self, norm_adj, u, pos, neg):
        users, items = self.propagate(norm_adj)
        u_e, pos_e, neg_e = users[u], items[pos], items[neg]
        pos_scores = (u_e * pos_e).sum(dim=1)
        neg_scores = (u_e * neg_e).sum(dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

        # L2 regularization on the *base* (layer-0) embeddings, as in the paper.
        reg = (self.user_embed(u).pow(2).sum()
               + self.item_embed(pos).pow(2).sum()
               + self.item_embed(neg).pow(2).sum()) / u.size(0)
        return loss, reg


def build_norm_adj(train_user_items, num_users, num_items, device):
    """Dense symmetrically-normalized bipartite adjacency  D^{-1/2} A D^{-1/2}.

    A dense matrix is used for portability (sparse mm is patchy on MPS); at
    MovieLens scale the (num_users + num_items)^2 matrix is only a few MB.
    """
    n = num_users + num_items
    adj = np.zeros((n, n), dtype=np.float32)
    for u, items in enumerate(train_user_items):
        for i in items:
            adj[u, num_users + i] = 1.0
            adj[num_users + i, u] = 1.0

    deg = adj.sum(axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    d_inv_sqrt[nz] = np.power(deg[nz], -0.5)
    norm = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
    return torch.tensor(norm, dtype=torch.float32, device=device)


def sample_bpr(train_user_items, num_items, rng):
    """Samples one (user, positive, negative) triple per training interaction."""
    users, pos_items, neg_items = [], [], []
    for u, items in enumerate(train_user_items):
        if not items:
            continue
        seen = set(items)
        for i in items:
            j = rng.integers(num_items)
            while j in seen:
                j = rng.integers(num_items)
            users.append(u)
            pos_items.append(i)
            neg_items.append(j)
    return (torch.tensor(users, dtype=torch.long),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long))


def main():
    p = mc.build_argparser("MovieLens LightGCN graph recommender", epochs=200, lr=1e-3)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--reg", type=float, default=1e-4, help="BPR L2 regularization weight")
    args = p.parse_args()

    device = mc.get_device(args.device)

    train_user_items, test_user_items, num_users, num_items = mc.load_movielens_implicit(
        limit=args.limit)

    norm_adj = build_norm_adj(train_user_items, num_users, num_items, device)
    model = LightGCN(num_users, num_items, embed_dim=args.embed_dim,
                     num_layers=args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(42)

    print("Training LightGCN...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        model.train()
        u, pos, neg = sample_bpr(train_user_items, num_items, rng)
        perm = torch.randperm(u.size(0))
        u, pos, neg = u[perm], pos[perm], neg[perm]

        epoch_loss, total = 0.0, 0
        for start in range(0, u.size(0), args.batch_size):
            bu = u[start:start + args.batch_size].to(device)
            bpos = pos[start:start + args.batch_size].to(device)
            bneg = neg[start:start + args.batch_size].to(device)

            optimizer.zero_grad()
            loss, reg = model.bpr_loss(norm_adj, bu, bpos, bneg)
            (loss + args.reg * reg).backward()
            optimizer.step()
            epoch_loss += loss.item() * bu.size(0)
            total += bu.size(0)

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | bpr_loss: {epoch_loss / total:.4f}")

    print("-" * 64)

    # --- Evaluation: full score matrix from propagated embeddings ---
    model.eval()
    with torch.no_grad():
        users, items = model.propagate(norm_adj)
        scores = (users @ items.t()).cpu()             # [num_users, num_items]

    metrics = mc.ranking_metrics_at_k(scores, train_user_items, test_user_items, ks=(10, 20))
    mc.print_ranking_metrics(metrics, ks=(10, 20))

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        cluster_path = os.path.join(save_dir, "lightgcn_movie_embeddings.png")
        mc.plot_movie_clusters(model.item_embed.weight, cluster_path,
                               "LightGCN Movie Embeddings")


if __name__ == "__main__":
    main()
