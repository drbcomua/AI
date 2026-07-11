"""
04. Factorization Machines (FM)
===============================

The general second-order Factorization Machine rating predictor (Rendle, 2010).

FM generalizes Matrix Factorization: instead of a single user-item dot product,
it models every pairwise interaction between input feature fields through shared
latent vectors, computed in linear time via the classic sum-of-squares trick.
With one-hot User and Item fields on MovieLens, the pairwise term reduces to the
user-item interaction, so FM cleanly reveals that "MF + biases" is a special
case of the broader FM family (and the shallow half of `03.deepfm.py`).

Architecture Diagram / Layout:
    User ID -> linear w_u [1] --------------------------+
    Item ID -> linear w_i [1] --------------------------+---> y_hat (1)
                                                        |
    User ID -> latent v_u [1 x k] --+                   |
                                     |-- FM 2nd order ---+
    Item ID -> latent v_i [1 x k] --+   0.5 * (sum^2 - sum_of_sq)

    FM(x) = w0 + sum_i w_i x_i + 0.5 * sum_f [ (sum_i v_{i,f} x_i)^2
                                               - sum_i (v_{i,f} x_i)^2 ]

Key insights / educational takeaways:
    * The sum-of-squares identity turns an O(k n^2) pairwise sum into O(k n),
      the trick that makes FM scale to high-cardinality sparse features.
    * Stacking fields as rows lets the same code handle any number of features;
      MovieLens simply uses two fields (User, Item).
    * Matrix Factorization with biases is FM restricted to a single field pair.

Run:
    python "04.fm.py" --epochs 10
    python "04.fm.py" --epochs 10 --features   # add user/movie side features
    python "04.fm.py" --limit 2000 --epochs 2  # fast smoke test

Side-feature mode (`--features`) is where FM truly earns its name: instead of
just User and Item fields it ingests gender, age group, occupation, movie
release-decade and (multi-hot) genres, and learns the pairwise interaction
between *every* pair of fields -- the reason FM generalizes far better than
Matrix Factorization on sparse, cold-start-prone data.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import rec_common as mc


class FactorizationMachine(nn.Module):
    """General second-order FM over stacked User and Item fields."""
    def __init__(self, num_users: int, num_items: int, embed_dim: int = 16):
        super().__init__()
        # 1-order linear weights (one per field value)
        self.user_linear = nn.Embedding(num_users, 1)
        self.item_linear = nn.Embedding(num_items, 1)
        self.bias = nn.Parameter(torch.zeros(1))

        # 2-order latent factors (one k-dim vector per field value)
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)

        nn.init.normal_(self.user_embed.weight, std=0.02)
        nn.init.normal_(self.item_embed.weight, std=0.02)
        nn.init.zeros_(self.user_linear.weight)
        nn.init.zeros_(self.item_linear.weight)

    def forward(self, user, item):
        # 1-order linear term
        linear_part = self.user_linear(user).squeeze(1) + self.item_linear(item).squeeze(1) + self.bias

        # 2-order FM term via the sum-of-squares trick.
        # Stack the active field vectors: [B, num_fields=2, k]
        u_emb = self.user_embed(user)
        i_emb = self.item_embed(item)
        stacked = torch.stack([u_emb, i_emb], dim=1)

        summed = stacked.sum(dim=1)              # (sum_i v_i)   -> [B, k]
        sum_of_square = (stacked ** 2).sum(dim=1)  # sum_i v_i^2 -> [B, k]
        square_of_sum = summed ** 2                # (sum_i v_i)^2
        fm_part = 0.5 * (square_of_sum - sum_of_square).sum(dim=1)  # [B]

        return linear_part + fm_part


class FactorizationMachineFeatures(nn.Module):
    """FM over all side-feature fields, reusing the shared multi-field encoder.

    The same sum-of-squares 2nd-order term as the ID-only model, now applied over
    ``num_single_fields + 1`` fields (the extra field is mean-pooled genres).
    """
    def __init__(self, meta: dict, embed_dim: int = 16):
        super().__init__()
        self.feat = mc.FeatureEmbedding(meta, embed_dim)

    def forward(self, single_fields, genre_multihot):
        field_vecs, linear = self.feat(single_fields, genre_multihot)  # [B, F+1, k], [B]
        summed = field_vecs.sum(dim=1)
        sum_of_square = (field_vecs ** 2).sum(dim=1)
        square_of_sum = summed ** 2
        fm_part = 0.5 * (square_of_sum - sum_of_square).sum(dim=1)
        return linear + fm_part


def run_features(args, device):
    """Multi-field FM path (--features): joins user/movie metadata onto ratings."""
    single, genre, ratings, meta = mc.load_movielens_features(limit=args.limit)

    np.random.seed(42)
    perm = np.random.permutation(len(ratings))
    split = int(len(ratings) * 0.8)
    tr, te = perm[:split], perm[split:]

    model = FactorizationMachineFeatures(meta, embed_dim=16).to(device)
    mc.train_feature_regression(model, single[tr], genre[tr], ratings[tr],
                                args, device, "Factorization Machine")
    mse, mae = mc.evaluate_feature_regression(model, single[te], genre[te], ratings[te],
                                              args, device)
    print(f"Test Rating MSE: {mse:.4f}")
    print(f"Test Rating MAE: {mae:.4f}")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        cluster_path = os.path.join(save_dir, "fm_features_movie_embeddings.png")
        mc.plot_movie_clusters(model.feat.item_embeddings(), cluster_path,
                               "FM (+features) Movie Embeddings")


def main():
    p = mc.build_argparser("MovieLens Factorization Machine", epochs=10)
    p.add_argument("--features", action="store_true",
                   help="use user/movie side features (multi-field FM)")
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

    model = FactorizationMachine(num_users, num_items, embed_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print("Training Factorization Machine...")
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

    # Plot latent movie clusters using FM item latent factors
    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        cluster_path = os.path.join(save_dir, "fm_movie_embeddings.png")
        mc.plot_movie_clusters(model.item_embed.weight, cluster_path, "Factorization Machine Movie Embeddings")


if __name__ == "__main__":
    main()
