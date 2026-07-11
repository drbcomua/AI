"""
05. Wide & Deep
===============

Google's Wide & Deep learning framework for recommender systems (Cheng et al., 2016).

Wide & Deep jointly trains two complementary halves:
  * a WIDE linear model over cross-product features for *memorization* of
    frequent, specific co-occurrences (this user liked this exact movie), and
  * a DEEP embedding MLP for *generalization* to unseen user-item combinations.
Because raw user x item cross-products are enormous, the wide side hashes each
pair into a fixed bucket table -- the standard hashing trick that keeps the
memorization component tractable at scale.

Architecture Diagram / Layout:
                        WIDE (memorization)
    User ID -> linear w_u [1] ------------------------+
    Item ID -> linear w_i [1] ------------------------+
    hash(User,Item) -> cross bucket weight [1] -------+
                                                      +--> y_hat (1)
                        DEEP (generalization)         |
    User ID -> emb [1 x d] --+                        |
                             |-- concat -> MLP -> [1] -+
    Item ID -> emb [1 x d] --+

Key insights / educational takeaways:
    * Memorization (wide) and generalization (deep) are complementary; jointly
      training both beats either alone on sparse interaction data.
    * The hashing trick approximates a full user x item cross-feature table with
      a bounded number of buckets, trading a little collision noise for scale.
    * DeepFM later replaces the hand-designed wide crosses with an FM component
      that learns the same low-order interactions automatically.

Run:
    python "05.wide-and-deep.py" --epochs 10
    python "05.wide-and-deep.py" --epochs 10 --features   # add user/movie side features
    python "05.wide-and-deep.py" --limit 2000 --epochs 2  # fast smoke test

With `--features`, the DEEP tower embeds the categorical side features (gender,
age, occupation, release-decade, genres) to *generalize* across demographics,
while the WIDE side keeps the hashed user x item cross for *memorization* -- the
memorization/generalization split the paper was built around.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import rec_common as mc


class WideAndDeep(nn.Module):
    """Wide (memorization) + Deep (generalization) rating predictor."""
    def __init__(self, num_users: int, num_items: int, embed_dim: int = 32,
                 mlp_hidden_dims: list[int] = [64, 32], cross_buckets: int = 100_000):
        super().__init__()
        self.num_items = num_items
        self.cross_buckets = cross_buckets

        # --- Wide component: linear memorization + hashed user x item cross ---
        self.user_linear = nn.Embedding(num_users, 1)
        self.item_linear = nn.Embedding(num_items, 1)
        self.cross_embed = nn.Embedding(cross_buckets, 1)
        self.bias = nn.Parameter(torch.zeros(1))

        # --- Deep component: dense embeddings -> MLP ---
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, mlp_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden_dims[1], 1),
        )

        nn.init.normal_(self.user_embed.weight, std=0.02)
        nn.init.normal_(self.item_embed.weight, std=0.02)
        nn.init.zeros_(self.user_linear.weight)
        nn.init.zeros_(self.item_linear.weight)
        nn.init.zeros_(self.cross_embed.weight)

    def forward(self, user, item):
        # Wide: individual linear terms + hashed cross-product feature
        cross_idx = (user.long() * self.num_items + item.long()) % self.cross_buckets
        wide = (
            self.user_linear(user).squeeze(1)
            + self.item_linear(item).squeeze(1)
            + self.cross_embed(cross_idx).squeeze(1)
            + self.bias
        )

        # Deep: concat embeddings through the MLP tower
        concat_emb = torch.cat([self.user_embed(user), self.item_embed(item)], dim=-1)
        deep = self.mlp(concat_emb).squeeze(1)

        return wide + deep


class WideAndDeepFeatures(nn.Module):
    """Wide & Deep over side features.

    Wide  = first-order (per-feature) memorization + hashed user x item cross.
    Deep  = MLP over all field embeddings (generalization from demographics/genre).
    """
    def __init__(self, meta: dict, embed_dim: int = 32,
                 mlp_hidden_dims: list[int] = [64, 32], cross_buckets: int = 100_000):
        super().__init__()
        self.feat = mc.FeatureEmbedding(meta, embed_dim)
        self.num_items = meta["num_items"]
        self.item_offset = meta["item_offset"]
        self.cross_buckets = cross_buckets
        self.cross_embed = nn.Embedding(cross_buckets, 1)
        nn.init.zeros_(self.cross_embed.weight)

        num_fields = meta["num_single_fields"] + 1
        self.mlp = nn.Sequential(
            nn.Linear(num_fields * embed_dim, mlp_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden_dims[1], 1),
        )

    def forward(self, single_fields, genre_multihot):
        field_vecs, linear = self.feat(single_fields, genre_multihot)  # [B, F+1, k], [B]

        # Recover 0-based user/item ids from their offset-encoded columns.
        user_local = single_fields[:, 0]                       # user offset is 0
        item_local = single_fields[:, 1] - self.item_offset
        cross_idx = (user_local * self.num_items + item_local) % self.cross_buckets
        wide = linear + self.cross_embed(cross_idx).squeeze(1)

        deep = self.mlp(field_vecs.flatten(start_dim=1)).squeeze(1)
        return wide + deep


def run_features(args, device):
    """Multi-field Wide & Deep path (--features): joins user/movie metadata onto ratings."""
    single, genre, ratings, meta = mc.load_movielens_features(limit=args.limit)

    np.random.seed(42)
    perm = np.random.permutation(len(ratings))
    split = int(len(ratings) * 0.8)
    tr, te = perm[:split], perm[split:]

    model = WideAndDeepFeatures(meta, embed_dim=32).to(device)
    mc.train_feature_regression(model, single[tr], genre[tr], ratings[tr],
                                args, device, "Wide & Deep")
    mse, mae = mc.evaluate_feature_regression(model, single[te], genre[te], ratings[te],
                                              args, device)
    print(f"Test Rating MSE: {mse:.4f}")
    print(f"Test Rating MAE: {mae:.4f}")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        cluster_path = os.path.join(save_dir, "wd_features_movie_embeddings.png")
        mc.plot_movie_clusters(model.feat.item_embeddings(), cluster_path,
                               "Wide & Deep (+features) Movie Embeddings")


def main():
    p = mc.build_argparser("MovieLens Wide & Deep network", epochs=10)
    p.add_argument("--features", action="store_true",
                   help="use user/movie side features (multi-field Wide & Deep)")
    args = p.parse_args()

    device = mc.get_device(args.device)

    if args.features:
        run_features(args, device)
        return

    # Load MovieLens ratings
    users, items, ratings, num_users, num_items = mc.load_movielens(limit=args.limit)

    # Train / test split (80/20)
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(ratings))
    split_idx = int(len(ratings) * 0.8)

    train_idx = shuffled_indices[:split_idx]
    test_idx = shuffled_indices[split_idx:]

    train_users, train_items, train_ratings = users[train_idx], items[train_idx], ratings[train_idx]
    test_users, test_items, test_ratings = users[test_idx], items[test_idx], ratings[test_idx]

    model = WideAndDeep(num_users, num_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print("Training Wide & Deep...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    train_dataset = TensorDataset(train_users, train_items, train_ratings)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        total = 0

        for u, i, r in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)

            optimizer.zero_grad()
            preds = model(u, i)
            loss = criterion(preds, r)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * u.size(0)
            total += u.size(0)

        train_mse = epoch_loss / total
        print(f"Epoch {epoch:2d}/{args.epochs} | train_mse: {train_mse:.4f}")

    print("-" * 64)

    # Evaluation
    test_dataset = TensorDataset(test_users, test_items, test_ratings)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    epoch_loss = 0.0
    total = 0
    with torch.no_grad():
        for u, i, r in test_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            preds = model(u, i)
            loss = criterion(preds, r)
            epoch_loss += loss.item() * u.size(0)
            total += u.size(0)

    test_mse = epoch_loss / total
    test_mae = 0.0
    with torch.no_grad():
        for u, i, r in test_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            preds = model(u, i)
            test_mae += torch.sum(torch.abs(preds - r)).item()

    test_mae = test_mae / total

    print(f"Test Rating MSE: {test_mse:.4f}")
    print(f"Test Rating MAE: {test_mae:.4f}")

    # Plot latent movie clusters using the deep item embeddings
    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        cluster_path = os.path.join(save_dir, "wd_movie_embeddings.png")
        mc.plot_movie_clusters(model.item_embed.weight, cluster_path, "Wide & Deep Movie Embeddings")


if __name__ == "__main__":
    main()
