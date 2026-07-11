"""
rec_common.py
=============

Shared utilities for MovieLens-100k Recommender Systems.
Handles automated MovieLens dataset downloading, caching, user-item mapping,
device configurations, and t-SNE movie embedding cluster plotting.
"""

from __future__ import annotations

import os
import zipfile
import urllib.request
import ssl
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Default MovieLens directory
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# --------------------------------------------------------------------------- #
# MovieLens-100k Automated Downloader & Loader
# --------------------------------------------------------------------------- #
def download_and_extract_movielens(data_dir: str = _DATA_DIR):
    """Downloads GroupLens MovieLens-100k dataset and extracts it.

    Bypasses HTTPS SSL context validation to ensure clean runs on macOS.
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ml-100k.zip")
    extracted_dir = os.path.join(data_dir, "ml-100k")
    u_data_path = os.path.join(extracted_dir, "u.data")

    # If already extracted, return u.data path
    if os.path.exists(u_data_path):
        return u_data_path

    # Globally disable SSL context verification fallback for urllib open
    try:
        ssl_context = ssl._create_unverified_context()
    except AttributeError:
        ssl_context = None

    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    print(f"Downloading MovieLens-100k from {url}...")

    try:
        if ssl_context:
            with urllib.request.urlopen(url, context=ssl_context) as response, open(zip_path, "wb") as out_file:
                out_file.write(response.read())
        else:
            urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download MovieLens zip file: {e}")

    print("Extracting ml-100k.zip...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to extract zip file: {e}")

    print("MovieLens-100k dataset successfully extracted.")
    return u_data_path


def load_movielens(limit: int | None = None):
    """Loads u.data ratings. Maps sparse user and item IDs to continuous indices.

    Returns:
        users (tensor): shape [N]
        items (tensor): shape [N]
        ratings (tensor): shape [N]
        num_users (int): user vocabulary count
        num_items (int): movie vocabulary count
    """
    u_data_path = download_and_extract_movielens()

    ratings_list = []
    with open(u_data_path, "r", encoding="latin-1") as f:
        for line in f:
            uid, iid, rating, _ = line.strip().split("\t")
            ratings_list.append([int(uid), int(iid), float(rating)])

    ratings_arr = np.array(ratings_list)

    if limit is not None:
        ratings_arr = ratings_arr[:limit]

    # Parse IDs
    raw_user_ids = ratings_arr[:, 0].astype(np.int32)
    raw_item_ids = ratings_arr[:, 1].astype(np.int32)
    stars = ratings_arr[:, 2].astype(np.float32)

    unique_users = np.unique(raw_user_ids)
    unique_items = np.unique(raw_item_ids)

    # Build index mappings
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}

    # Map to continuous indices [0..V-1]
    mapped_users = np.array([user_to_idx[uid] for uid in raw_user_ids])
    mapped_items = np.array([item_to_idx[iid] for iid in raw_item_ids])

    num_users = len(unique_users)
    num_items = len(unique_items)

    print(f"Loaded MovieLens: {len(stars)} ratings | Users: {num_users} | Movies: {num_items}")
    return (
        torch.tensor(mapped_users, dtype=torch.long),
        torch.tensor(mapped_items, dtype=torch.long),
        torch.tensor(stars, dtype=torch.float32),
        num_users,
        num_items
    )


# --------------------------------------------------------------------------- #
# Side-feature loader & multi-field encoder (for FM / DeepFM / Wide&Deep)
# --------------------------------------------------------------------------- #
# The core loader above exposes only (user, item, rating). Factorization-Machine
# style models, however, are designed to exploit *side features*. This section
# joins the MovieLens metadata files (u.user demographics, u.item genres/year)
# onto each rating so those models can demonstrate multi-field feature crossing.
_AGE_BUCKET_BOUNDS = [18, 25, 35, 45, 50, 56]  # standard MovieLens age groups


def _age_bucket(age: int) -> int:
    """Maps an age to one of 7 canonical MovieLens age groups."""
    b = 0
    for bound in _AGE_BUCKET_BOUNDS:
        if age < bound:
            return b
        b += 1
    return b  # 56+


def load_movielens_features(limit: int | None = None):
    """Joins u.user and u.item metadata onto ratings for multi-field models.

    Single-value categorical fields (user, item, gender, age-group, occupation,
    release-decade) are offset-encoded into one shared index space so a single
    embedding table can serve them all; movie genres are returned separately as a
    multi-hot matrix (a movie may belong to several genres).

    Returns:
        single_fields (LongTensor [N, 6]): offset-encoded categorical indices
        genre_multihot (FloatTensor [N, 19]): multi-hot genre membership
        ratings (FloatTensor [N])
        meta (dict): field_dims, offsets, total_single, num_single_fields,
                     num_genres, num_users, num_items, item_offset, field_names
    """
    u_data_path = download_and_extract_movielens()
    base = os.path.dirname(u_data_path)

    # --- user demographics: id -> (age, gender, occupation) ---
    user_info = {}
    with open(os.path.join(base, "u.user"), encoding="latin-1") as f:
        for line in f:
            uid, age, gender, occ, _zip = line.strip().split("|")
            user_info[int(uid)] = (int(age), gender, occ)

    # --- item metadata: id -> (release_decade, genre_flags[19]) ---
    item_info = {}
    with open(os.path.join(base, "u.item"), encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            iid = int(parts[0])
            release = parts[2]
            year = 0
            if release and "-" in release:
                try:
                    year = int(release.split("-")[-1])
                except ValueError:
                    year = 0
            genres = [int(x) for x in parts[5:24]]  # 19 genre flags
            item_info[iid] = (year // 10, genres)   # decade bucket

    rows = _read_ratings_rows(limit)

    field_order = ["user", "item", "gender", "age", "occ", "year"]
    raw = {k: [] for k in field_order}
    genre_rows, ratings = [], []
    for u, i, r, _ in rows:
        age, gender, occ = user_info[u]
        decade, genres = item_info[i]
        raw["user"].append(u)
        raw["item"].append(i)
        raw["gender"].append(gender)
        raw["age"].append(_age_bucket(age))
        raw["occ"].append(occ)
        raw["year"].append(decade)
        genre_rows.append(genres)
        ratings.append(r)

    # Remap each field to a contiguous range, then apply per-field offsets.
    dims, maps = [], {}
    for fld in field_order:
        uniq = sorted(set(raw[fld]), key=lambda x: (str(type(x)), x))
        maps[fld] = {v: k for k, v in enumerate(uniq)}
        dims.append(len(uniq))
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)

    n = len(ratings)
    single = np.zeros((n, len(field_order)), dtype=np.int64)
    for fi, fld in enumerate(field_order):
        m, off = maps[fld], offsets[fi]
        single[:, fi] = [m[v] + off for v in raw[fld]]

    genre_mh = np.asarray(genre_rows, dtype=np.float32)

    meta = {
        "field_dims": dims,
        "offsets": offsets.tolist(),
        "total_single": int(sum(dims)),
        "num_single_fields": len(field_order),
        "num_genres": int(genre_mh.shape[1]),
        "num_users": dims[0],
        "num_items": dims[1],
        "item_offset": int(offsets[1]),
        "field_names": field_order + ["genres"],
    }
    print(f"Loaded MovieLens (features): {n} ratings | "
          f"fields {field_order}+genres | dims {dims}+[{meta['num_genres']}]")
    return (
        torch.tensor(single, dtype=torch.long),
        torch.tensor(genre_mh, dtype=torch.float32),
        torch.tensor(ratings, dtype=torch.float32),
        meta,
    )


class FeatureEmbedding(nn.Module):
    """Shared multi-field encoder for FM-family models.

    Produces per-field latent vectors ``[B, F+1, k]`` (single fields plus one
    mean-pooled genre field) and a scalar first-order/linear term ``[B]``. The
    FM 2nd-order term, deep MLP, and wide crosses are left to each model.
    """
    def __init__(self, meta: dict, embed_dim: int = 16):
        super().__init__()
        total = meta["total_single"]
        num_genres = meta["num_genres"]
        self.embed = nn.Embedding(total, embed_dim)
        self.linear = nn.Embedding(total, 1)
        self.genre_embed = nn.Parameter(torch.empty(num_genres, embed_dim))
        self.genre_linear = nn.Parameter(torch.zeros(num_genres, 1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.item_offset = meta["item_offset"]
        self.num_items = meta["num_items"]

        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.genre_embed, std=0.02)

    def forward(self, single_fields, genre_multihot):
        # Single-field latent vectors.
        field_vecs = self.embed(single_fields)                     # [B, F, k]
        # Genre field: mean-pool the embeddings of active genres.
        counts = genre_multihot.sum(dim=1, keepdim=True).clamp(min=1.0)
        genre_vec = (genre_multihot @ self.genre_embed) / counts   # [B, k]
        field_vecs = torch.cat([field_vecs, genre_vec.unsqueeze(1)], dim=1)  # [B, F+1, k]

        # First-order (linear) term.
        linear = self.linear(single_fields).sum(dim=1).squeeze(-1)
        genre_lin = ((genre_multihot @ self.genre_linear).squeeze(-1)
                     / counts.squeeze(-1))
        linear = linear + genre_lin + self.bias
        return field_vecs, linear

    def item_embeddings(self) -> torch.Tensor:
        """Item-field embedding slice, for t-SNE cluster plots."""
        return self.embed.weight[self.item_offset: self.item_offset + self.num_items]


def train_feature_regression(model, single, genre_mh, ratings, args, device, name):
    """Standard MSE training loop for multi-field regression models."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training {name} (+side features)...")
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    loader = DataLoader(TensorDataset(single, genre_mh, ratings),
                        batch_size=args.batch_size, shuffle=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, total = 0.0, 0
        for sf, gm, r in loader:
            sf, gm, r = sf.to(device), gm.to(device), r.to(device)
            optimizer.zero_grad()
            preds = model(sf, gm)
            loss = criterion(preds, r)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * sf.size(0)
            total += sf.size(0)
        print(f"Epoch {epoch:2d}/{args.epochs} | train_mse: {epoch_loss / total:.4f}")
    print("-" * 64)


def evaluate_feature_regression(model, single, genre_mh, ratings, args, device):
    """Returns (MSE, MAE) for a trained multi-field regression model."""
    loader = DataLoader(TensorDataset(single, genre_mh, ratings),
                        batch_size=args.batch_size, shuffle=False)
    model.eval()
    se, ae, total = 0.0, 0.0, 0
    with torch.no_grad():
        for sf, gm, r in loader:
            sf, gm, r = sf.to(device), gm.to(device), r.to(device)
            preds = model(sf, gm)
            se += torch.sum((preds - r) ** 2).item()
            ae += torch.sum(torch.abs(preds - r)).item()
            total += sf.size(0)
    return se / total, ae / total


# --------------------------------------------------------------------------- #
# Implicit-feedback loaders & ranking evaluation
# --------------------------------------------------------------------------- #
# The scripts above (MF, FM, NCF, Wide&Deep, DeepFM) treat recommendation as
# explicit rating regression (MSE/MAE). The models below (SASRec, LightGCN,
# Mult-VAE) instead operate on *implicit* feedback and are judged by top-K
# ranking quality. These helpers provide the shared data prep and metrics for
# that setup, keeping the model scripts thin.
def _read_ratings_rows(limit: int | None = None):
    """Reads raw (user, item, rating, timestamp) rows from u.data."""
    u_data_path = download_and_extract_movielens()
    rows = []
    with open(u_data_path, "r", encoding="latin-1") as f:
        for line in f:
            uid, iid, rating, ts = line.strip().split("\t")
            rows.append((int(uid), int(iid), float(rating), int(ts)))
    if limit is not None:
        rows = rows[:limit]
    return rows


def load_movielens_implicit(limit: int | None = None, min_rating: float = 4.0,
                            test_ratio: float = 0.2, seed: int = 42):
    """Binarizes ratings to implicit positives and builds a per-user split.

    A rating >= ``min_rating`` counts as a positive interaction. User and item
    IDs are remapped to contiguous indices over the positives only. Each user's
    positives are randomly split into train / test.

    Returns:
        train_user_items (list[list[int]]): per-user train item indices
        test_user_items (list[list[int]]):  per-user held-out item indices
        num_users (int), num_items (int)
    """
    rows = _read_ratings_rows(limit)
    pos = [(u, i) for (u, i, r, _) in rows if r >= min_rating]

    unique_users = sorted({u for u, _ in pos})
    unique_items = sorted({i for _, i in pos})
    u2idx = {u: k for k, u in enumerate(unique_users)}
    i2idx = {i: k for k, i in enumerate(unique_items)}
    num_users, num_items = len(unique_users), len(unique_items)

    by_user = [[] for _ in range(num_users)]
    for u, i in pos:
        by_user[u2idx[u]].append(i2idx[i])

    rng = np.random.default_rng(seed)
    train_user_items = [[] for _ in range(num_users)]
    test_user_items = [[] for _ in range(num_users)]
    for u in range(num_users):
        items = by_user[u]
        if len(items) < 2:
            train_user_items[u] = list(items)
            continue
        items = list(items)
        rng.shuffle(items)
        n_test = max(1, int(len(items) * test_ratio))
        test_user_items[u] = items[:n_test]
        train_user_items[u] = items[n_test:]

    n_train = sum(len(x) for x in train_user_items)
    n_test = sum(len(x) for x in test_user_items)
    print(f"Loaded MovieLens (implicit, rating>={min_rating}): "
          f"{n_train} train / {n_test} test interactions | "
          f"Users: {num_users} | Movies: {num_items}")
    return train_user_items, test_user_items, num_users, num_items


def load_movielens_sequences(limit: int | None = None, min_rating: float = 1.0):
    """Builds chronological per-user interaction sequences for sequential models.

    Interactions are ordered by timestamp. IDs are remapped to contiguous
    0-based indices over the retained interactions.

    Returns:
        sequences (list[list[int]]): per-user item indices in time order
        num_users (int), num_items (int)
    """
    rows = [(u, i, t) for (u, i, r, t) in _read_ratings_rows(limit) if r >= min_rating]

    unique_users = sorted({u for u, _, _ in rows})
    unique_items = sorted({i for _, i, _ in rows})
    u2idx = {u: k for k, u in enumerate(unique_users)}
    i2idx = {i: k for k, i in enumerate(unique_items)}
    num_users, num_items = len(unique_users), len(unique_items)

    rows.sort(key=lambda x: x[2])  # global timestamp order (stable per user)
    sequences = [[] for _ in range(num_users)]
    for u, i, _ in rows:
        sequences[u2idx[u]].append(i2idx[i])

    avg_len = np.mean([len(s) for s in sequences]) if num_users else 0.0
    print(f"Loaded MovieLens (sequences, rating>={min_rating}): "
          f"{num_users} users | {num_items} movies | avg length {avg_len:.1f}")
    return sequences, num_users, num_items


def ranking_metrics_at_k(scores: torch.Tensor,
                         train_user_items: list,
                         test_user_items: list,
                         ks=(10, 20)) -> dict:
    """Computes Recall@K and NDCG@K from a full user-item score matrix.

    Items already seen in training are masked out before ranking. Users with no
    held-out test items are skipped. Recall@K is normalized by min(K, #relevant)
    following the Mult-VAE convention.

    Args:
        scores: FloatTensor [num_users, num_items] of predicted scores.
        train_user_items: per-user item indices to exclude from ranking.
        test_user_items:  per-user held-out ground-truth item indices.
        ks: cutoffs to report.
    """
    scores = scores.clone()
    num_users = scores.size(0)

    # Mask training items so they cannot occupy top-K slots.
    for u in range(num_users):
        seen = train_user_items[u]
        if len(seen) > 0:
            scores[u, seen] = float("-inf")

    max_k = max(ks)
    topk = torch.topk(scores, max_k, dim=1).indices.cpu().numpy()

    metrics = {f"Recall@{k}": 0.0 for k in ks}
    metrics.update({f"NDCG@{k}": 0.0 for k in ks})
    # Precompute ideal DCG discounts.
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2))

    n_eval = 0
    for u in range(num_users):
        gt = test_user_items[u]
        if len(gt) == 0:
            continue
        n_eval += 1
        gt_set = set(gt)
        ranked = topk[u]
        hits = np.array([1.0 if it in gt_set else 0.0 for it in ranked])
        for k in ks:
            n_rel = min(len(gt_set), k)
            metrics[f"Recall@{k}"] += hits[:k].sum() / n_rel
            dcg = (hits[:k] * discounts[:k]).sum()
            idcg = discounts[:n_rel].sum()
            metrics[f"NDCG@{k}"] += dcg / idcg if idcg > 0 else 0.0

    for key in metrics:
        metrics[key] /= max(n_eval, 1)
    metrics["_num_eval_users"] = n_eval
    return metrics


def print_ranking_metrics(metrics: dict, ks=(10, 20)):
    """Pretty-prints the dict returned by ``ranking_metrics_at_k``."""
    print(f"Evaluated on {metrics.get('_num_eval_users', 0)} users with held-out items")
    for k in ks:
        print(f"Recall@{k}: {metrics[f'Recall@{k}']:.4f} | NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}")


# --------------------------------------------------------------------------- #
# Device, Argparser, and Common Functions
# --------------------------------------------------------------------------- #
def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_argparser(description: str, epochs: int = 10, batch_size: int = 256, lr: float = 1e-3):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--batch-size", type=int, default=batch_size)
    p.add_argument("--lr", type=float, default=lr)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--no-figure", action="store_true", help="do not save figures")
    p.add_argument("--limit", type=int, default=None, help="limit dataset samples for quick local checks")
    return p


# --------------------------------------------------------------------------- #
# Visualizations: t-SNE Latent Movie Clustering
# --------------------------------------------------------------------------- #
def plot_movie_clusters(embeddings: torch.Tensor, save_path: str, title: str):
    """Projects high-dimensional movie embeddings into 2D using t-SNE and saves a scatter plot."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping movie cluster plot: {e})")
        return

    print("Computing t-SNE projection of movie embeddings...")
    emb_np = embeddings.detach().cpu().numpy()

    # Filter to top 300 movies to keep plot legible
    emb_np = emb_np[:300]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(emb_np)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c="teal", alpha=0.7, edgecolors="black", linewidths=0.3, s=25
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved movie embedding cluster plot -> {save_path}")
