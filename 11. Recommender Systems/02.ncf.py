r"""
02. Neural Collaborative Filtering (NCF)
========================================

Neural Collaborative Filtering mapping concatenated user/item embeddings through fully connected MLP layers (He et al., 2017).

Architecture Diagram / Layout:
    User GMF Embed [1 x 16] -\
    Item GMF Embed [1 x 16] ---> GMF Output (element-wise product) [16] -\
                                                                         +---> Fusion [24] -> Prediction (1)
    User MLP Embed [1 x 16] -\                                           |
    Item MLP Embed [1 x 16] ---> MLP Layers (Linear Layers) [8] ---------/

Key insights / educational takeaways:
    * Combines a linear GMF component with non-linear deep layers to learn complex collaborative interactions.
    * Explains why fusing two different representations (GMF and MLP) yields a more expressive metric space.

Run:
    python "02.ncf.py" --epochs 10
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import rec_common as mc


class NeuMF(nn.Module):
    """Neural Collaborative Filtering (NeuMF) model fusing GMF and MLP networks."""
    def __init__(self, num_users: int, num_items: int, latent_dim_gmf: int = 16,
                 latent_dim_mlp: int = 16, mlp_hidden_dims: list[int] = [32, 16, 8]):
        super().__init__()
        # GMF embeddings
        self.user_embed_gmf = nn.Embedding(num_users, latent_dim_gmf)
        self.item_embed_gmf = nn.Embedding(num_items, latent_dim_gmf)

        # MLP embeddings
        self.user_embed_mlp = nn.Embedding(num_users, latent_dim_mlp)
        self.item_embed_mlp = nn.Embedding(num_items, latent_dim_mlp)

        # MLP layers
        mlp_layers = []
        in_dim = latent_dim_mlp * 2
        for out_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_layers)

        # Final prediction fusion
        self.prediction_layer = nn.Linear(latent_dim_gmf + mlp_hidden_dims[-1], 1)

        # Init
        nn.init.normal_(self.user_embed_gmf.weight, std=0.02)
        nn.init.normal_(self.item_embed_gmf.weight, std=0.02)
        nn.init.normal_(self.user_embed_mlp.weight, std=0.02)
        nn.init.normal_(self.item_embed_mlp.weight, std=0.02)

    def forward(self, user, item):
        # GMF Branch
        u_gmf = self.user_embed_gmf(user)
        i_gmf = self.item_embed_gmf(item)
        phi_gmf = u_gmf * i_gmf

        # MLP Branch
        u_mlp = self.user_embed_mlp(user)
        i_mlp = self.item_embed_mlp(item)
        phi_mlp = torch.cat([u_mlp, i_mlp], dim=-1)
        phi_mlp = self.mlp(phi_mlp)

        # Fusion & Prediction
        fusion = torch.cat([phi_gmf, phi_mlp], dim=-1)
        return self.prediction_layer(fusion).squeeze(1)


def main():
    p = mc.build_argparser("Neural Collaborative Filtering (NCF)", epochs=10)
    args = p.parse_args()

    device = mc.get_device(args.device)

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

    model = NeuMF(num_users, num_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    print("Training Neural Collaborative Filtering (NCF)...")
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

    # Plot latent movie clusters using fusion GMF item embeddings
    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        cluster_path = os.path.join(save_dir, "ncf_movie_embeddings.png")
        mc.plot_movie_clusters(model.item_embed_gmf.weight, cluster_path, "NCF GMF Movie Embeddings")


if __name__ == "__main__":
    main()
