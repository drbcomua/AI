"""
01. Matrix Factorization
=========================

Classic latent factor collaborative filtering rating predictor.

Architecture Diagram / Layout:
    User ID -> User Embedding [1 x 32] ----+
                                           |---> (Dot Product) + User Bias + Item Bias
    Item ID -> Item Embedding [1 x 32] ----+

Key insights / educational takeaways:
    * Represents collaborative filtering mathematically as latent alignment in vector spaces.
    * Explains how the dot product models linear user-item preferences.

Run:
    python "01.matrix-factorization.py" --epochs 10
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import rec_common as mc


class MatrixFactorization(nn.Module):
    """Matrix Factorization rating predictor with user and item biases."""
    def __init__(self, num_users: int, num_items: int, embed_dim: int = 32):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # Initialize weights
        nn.init.normal_(self.user_embed.weight, std=0.02)
        nn.init.normal_(self.item_embed.weight, std=0.02)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user, item):
        u_emb = self.user_embed(user)
        i_emb = self.item_embed(item)
        u_b = self.user_bias(user).squeeze(1)
        i_b = self.item_bias(item).squeeze(1)

        dot = torch.sum(u_emb * i_emb, dim=1)
        return dot + u_b + i_b


def main():
    p = mc.build_argparser("MovieLens Matrix Factorization", epochs=10)
    args = p.parse_args()

    device = mc.get_device(args.device)

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

    model = MatrixFactorization(num_users, num_items, embed_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print("Training Matrix Factorization...")
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
    # Also compute MAE
    with torch.no_grad():
        for u, i, r in test_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            preds = model(u, i)
            test_mae += torch.sum(torch.abs(preds - r)).item()

    test_mae = test_mae / total

    print(f"Test Rating MSE: {test_mse:.4f}")
    print(f"Test Rating MAE: {test_mae:.4f}")

    # Plot latent movie clusters
    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        cluster_path = os.path.join(save_dir, "mf_movie_embeddings.png")
        mc.plot_movie_clusters(model.item_embed.weight, cluster_path, "Matrix Factorization Movie Embeddings")


if __name__ == "__main__":
    main()
