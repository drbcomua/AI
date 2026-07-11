"""
03. DeepFM
==========

A Deep Factorization Machine combining low-order linear crossings with high-order deep MLP crossings (Guo et al., 2017).

Architecture Diagram / Layout:
    User & Item IDs
        -> 1-order Linear (User/Item linear embedding weights) [1] ---------------+
        -> 2-order FM (Dot product of 16-dim latent embeddings) [1] --------------+---> Predict (1)
        -> Deep MLP (Concat Embeddings -> MLP layers -> Linear) [1] --------------/

Key insights / educational takeaways:
    * Eliminates the need for manual feature crosses by jointly learning shallow linear and deep non-linear interactions.
    * Shows how factorization machines act as a regularizer inside deep recommendation networks.

Run:
    python "03.deepfm.py" --epochs 10
    python "03.deepfm.py" --epochs 10 --features   # add user/movie side features

With `--features`, DeepFM ingests the full multi-field input (user, item, gender,
age group, occupation, release-decade, genres): the FM component captures every
low-order feature pair while the deep tower models high-order genre/demographic
combinations -- exactly the CTR setting the paper targets.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import rec_common as mc


class DeepFM(nn.Module):
    """DeepFM collaborative predictor combining Linear, FM, and MLP components."""
    def __init__(self, num_users: int, num_items: int, embed_dim: int = 16, mlp_hidden_dims: list[int] = [32, 16]):
        super().__init__()
        # 1-order Linear weights
        self.user_linear = nn.Embedding(num_users, 1)
        self.item_linear = nn.Embedding(num_items, 1)
        self.bias = nn.Parameter(torch.zeros(1))

        # 2-order FM & Deep embeddings
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)

        # MLP network layers
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, mlp_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden_dims[1], 1)
        )

        # Init weights
        nn.init.normal_(self.user_embed.weight, std=0.02)
        nn.init.normal_(self.item_embed.weight, std=0.02)
        nn.init.zeros_(self.user_linear.weight)
        nn.init.zeros_(self.item_linear.weight)

    def forward(self, user, item):
        # 1. Linear 1-order component
        linear_part = self.user_linear(user).squeeze(1) + self.item_linear(item).squeeze(1) + self.bias

        # 2. FM 2-order component (pairwise inner product)
        u_emb = self.user_embed(user)
        i_emb = self.item_embed(item)
        fm_part = torch.sum(u_emb * i_emb, dim=1)

        # 3. High-order Deep component
        concat_emb = torch.cat([u_emb, i_emb], dim=-1)
        deep_part = self.mlp(concat_emb).squeeze(1)

        # Combine
        return linear_part + fm_part + deep_part


class DeepFMFeatures(nn.Module):
    """DeepFM over all side-feature fields, reusing the shared multi-field encoder.

    FM component (sum-of-squares) captures low-order pairs; the deep MLP over the
    flattened field embeddings captures high-order crossings.
    """
    def __init__(self, meta: dict, embed_dim: int = 16, mlp_hidden_dims: list[int] = [64, 32]):
        super().__init__()
        self.feat = mc.FeatureEmbedding(meta, embed_dim)
        num_fields = meta["num_single_fields"] + 1  # + mean-pooled genre field
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

        summed = field_vecs.sum(dim=1)
        sum_of_square = (field_vecs ** 2).sum(dim=1)
        fm_part = 0.5 * (summed ** 2 - sum_of_square).sum(dim=1)

        deep_part = self.mlp(field_vecs.flatten(start_dim=1)).squeeze(1)
        return linear + fm_part + deep_part


def run_features(args, device):
    """Multi-field DeepFM path (--features): joins user/movie metadata onto ratings."""
    single, genre, ratings, meta = mc.load_movielens_features(limit=args.limit)

    np.random.seed(42)
    perm = np.random.permutation(len(ratings))
    split = int(len(ratings) * 0.8)
    tr, te = perm[:split], perm[split:]

    model = DeepFMFeatures(meta, embed_dim=16).to(device)
    mc.train_feature_regression(model, single[tr], genre[tr], ratings[tr],
                                args, device, "DeepFM")
    mse, mae = mc.evaluate_feature_regression(model, single[te], genre[te], ratings[te],
                                              args, device)
    print(f"Test Rating MSE: {mse:.4f}")
    print(f"Test Rating MAE: {mae:.4f}")

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        cluster_path = os.path.join(save_dir, "deepfm_features_movie_embeddings.png")
        mc.plot_movie_clusters(model.feat.item_embeddings(), cluster_path,
                               "DeepFM (+features) Movie Embeddings")


def main():
    p = mc.build_argparser("DeepFM user-item rating network", epochs=10)
    p.add_argument("--features", action="store_true",
                   help="use user/movie side features (multi-field DeepFM)")
    args = p.parse_args()

    device = mc.get_device(args.device)

    if args.features:
        run_features(args, device)
        return

    # Load ratings
    users, items, ratings, num_users, num_items = mc.load_movielens(limit=args.limit)

    # Split (80/20)
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(ratings))
    split_idx = int(len(ratings) * 0.8)

    train_idx = shuffled_indices[:split_idx]
    test_idx = shuffled_indices[split_idx:]

    train_users, train_items, train_ratings = users[train_idx], items[train_idx], ratings[train_idx]
    test_users, test_items, test_ratings = users[test_idx], items[test_idx], ratings[test_idx]

    model = DeepFM(num_users, num_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print("Training DeepFM...")
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

    # Plot latent movie clusters using DeepFM item embeddings
    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        cluster_path = os.path.join(save_dir, "deepfm_movie_embeddings.png")
        mc.plot_movie_clusters(model.item_embed.weight, cluster_path, "DeepFM Movie Embeddings")


if __name__ == "__main__":
    main()
