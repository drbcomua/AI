"""
08. Mult-VAE (Variational Autoencoder with Multinomial Likelihood)
==================================================================

A denoising variational autoencoder for collaborative filtering (Liang et al., 2018).

Where the other models embed individual (user, item) pairs, Mult-VAE treats each
user's entire interaction history as one bag-of-items vector and learns to
*reconstruct* it. A user is encoded into a latent Gaussian, sampled, and decoded
back into a distribution over the whole catalog via a **multinomial** likelihood
— a much better fit for implicit-feedback click data than Gaussian/logistic
losses. Input dropout turns it into a denoising autoencoder, forcing the model
to predict held-out items from a corrupted history.

Architecture Diagram / Layout:
    x  (user interaction vector, [num_items])  -- L2-normalize + dropout -->
        Encoder MLP [I -> 600] -> (mu [200], logvar [200])
        z = mu + eps * exp(0.5*logvar)                 (reparameterization)
        Decoder MLP [200 -> 600 -> I]  -> logits
    loss = -E[ sum_i x_i * log softmax(logits)_i ]  +  beta * KL(q(z|x) || N(0,1))

Key insights / educational takeaways:
    * The multinomial likelihood ranks the whole catalog jointly, rewarding the
      model for placing probability mass on the items a user actually consumed.
    * KL annealing (slowly raising beta from 0) prevents posterior collapse; the
      paper caps beta well below 1, deliberately under-regularizing the latent.
    * A single forward pass scores every item for a user, so the same trained
      encoder-decoder serves both prediction and top-K ranking.

Run:
    python "08.mult-vae.py" --epochs 100
    python "08.mult-vae.py" --limit 5000 --epochs 20   # fast smoke test
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rec_common as mc


class MultVAE(nn.Module):
    """Multinomial VAE: symmetric encoder/decoder over the item vocabulary."""
    def __init__(self, num_items: int, hidden_dim: int = 600, latent_dim: int = 200,
                 dropout: float = 0.5):
        super().__init__()
        self.num_items = num_items
        self.input_dropout = nn.Dropout(dropout)

        self.encoder = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.Tanh(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_items),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x):
        x = F.normalize(x, p=2, dim=1)      # L2-normalize the interaction vector
        x = self.input_dropout(x)           # denoising corruption
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu                           # deterministic at eval

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        return logits, mu, logvar


def vae_loss(logits, x, mu, logvar, beta):
    """Multinomial reconstruction NLL + beta-weighted KL divergence."""
    log_softmax = F.log_softmax(logits, dim=1)
    neg_ll = -(log_softmax * x).sum(dim=1).mean()
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    return neg_ll + beta * kld, neg_ll, kld


def build_train_matrix(train_user_items, num_users, num_items):
    """Dense [num_users, num_items] binary interaction matrix."""
    mat = np.zeros((num_users, num_items), dtype=np.float32)
    for u, items in enumerate(train_user_items):
        for i in items:
            mat[u, i] = 1.0
    return torch.tensor(mat)


def main():
    p = mc.build_argparser("MovieLens Mult-VAE recommender", epochs=100, batch_size=256, lr=1e-3)
    p.add_argument("--hidden-dim", type=int, default=600)
    p.add_argument("--latent-dim", type=int, default=200)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--beta-cap", type=float, default=0.2, help="max KL weight after annealing")
    p.add_argument("--anneal-epochs", type=int, default=50, help="epochs to ramp beta to cap")
    args = p.parse_args()

    device = mc.get_device(args.device)

    train_user_items, test_user_items, num_users, num_items = mc.load_movielens_implicit(
        limit=args.limit)
    train_matrix = build_train_matrix(train_user_items, num_users, num_items)

    model = MultVAE(num_items, hidden_dim=args.hidden_dim,
                    latent_dim=args.latent_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    print("Training Mult-VAE...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    user_order = np.arange(num_users)
    for epoch in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(user_order)
        beta = args.beta_cap * min(1.0, epoch / max(1, args.anneal_epochs))

        epoch_loss, total = 0.0, 0
        for start in range(0, num_users, args.batch_size):
            idx = user_order[start:start + args.batch_size]
            x = train_matrix[idx].to(device)
            optimizer.zero_grad()
            logits, mu, logvar = model(x)
            loss, _, _ = vae_loss(logits, x, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
            total += x.size(0)

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | loss: {epoch_loss / total:.4f} | beta: {beta:.3f}")

    print("-" * 64)

    # --- Evaluation: reconstruct every user, rank held-out items ---
    model.eval()
    all_scores = []
    with torch.no_grad():
        for start in range(0, num_users, args.batch_size):
            x = train_matrix[start:start + args.batch_size].to(device)
            logits, _, _ = model(x)
            all_scores.append(logits.cpu())
    scores = torch.cat(all_scores, dim=0)              # [num_users, num_items]

    metrics = mc.ranking_metrics_at_k(scores, train_user_items, test_user_items, ks=(10, 20))
    mc.print_ranking_metrics(metrics, ks=(10, 20))

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        cluster_path = os.path.join(save_dir, "multvae_movie_embeddings.png")
        # Each output-layer weight column is the decoder's representation of an item.
        item_embeddings = model.decoder[-1].weight.detach()   # [num_items, hidden_dim]
        mc.plot_movie_clusters(item_embeddings, cluster_path, "Mult-VAE Movie Embeddings")


if __name__ == "__main__":
    main()
