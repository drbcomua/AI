"""
gnn_common.py
=============

Shared utilities for Graph Neural Networks (GNN) demos in this folder.
Handles Cora and MUTAG dataset downloads, parsing into features and adjacencies,
and common training loops. Does not require PyG or DGL.
"""

from __future__ import annotations

import os
import ssl
import urllib.request
import argparse
import numpy as np
import torch

# Cora raw URLs
CORA_CONTENT_URL = "https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/cora.content"
CORA_CITES_URL = "https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/cora.cites"

# MUTAG raw URLs
MUTAG_BASE_URL = "https://raw.githubusercontent.com/BorgwardtLab/Graph-Datasets/master/MUTAG/"
MUTAG_FILES = [
    "MUTAG_A.txt",
    "MUTAG_graph_indicator.txt",
    "MUTAG_graph_labels.txt",
    "MUTAG_node_labels.txt"
]

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# --------------------------------------------------------------------------- #
# Data Download & SSL bypass
# --------------------------------------------------------------------------- #
def _download_file(url: str, dest_path: str):
    """Download a file with SSL bypass context to prevent certificate issues on macOS."""
    if os.path.exists(dest_path):
        return dest_path
    print(f"Downloading {url}...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as r:
        blob = r.read()
    with open(dest_path, "wb") as f:
        f.write(blob)
    return dest_path


# --------------------------------------------------------------------------- #
# Cora Dataset Loader & Parser
# --------------------------------------------------------------------------- #
def load_cora(limit: int | None = None, data_dir: str = _DATA_DIR):
    """Load Cora citation network. Returns (features, adj_normalized, labels, splits).

    Node Classification Task (Semi-supervised).
    """
    cora_dir = os.path.join(data_dir, "cora")
    os.makedirs(cora_dir, exist_ok=True)

    content_path = os.path.join(cora_dir, "cora.content")
    cites_path = os.path.join(cora_dir, "cora.cites")

    _download_file(CORA_CONTENT_URL, content_path)
    _download_file(CORA_CITES_URL, cites_path)

    # 1. Parse content
    features = []
    labels_str = []
    node_ids = []
    idx_map = {}

    with open(content_path, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue
            paper_id = int(parts[0])
            node_ids.append(paper_id)
            idx_map[paper_id] = i
            # 1433 attributes
            feat = [float(x) for x in parts[1:-1]]
            features.append(feat)
            labels_str.append(parts[-1])

    features = np.array(features, dtype=np.float32)

    # Map labels to integers
    unique_labels = sorted(list(set(labels_str)))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels_str], dtype=np.int64)

    # 2. Parse cites (edges)
    num_nodes = len(node_ids)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    with open(cites_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                p1 = int(parts[0])
                p2 = int(parts[1])
                # Graph is typically treated as undirected for message passing
                if p1 in idx_map and p2 in idx_map:
                    idx1 = idx_map[p1]
                    idx2 = idx_map[p2]
                    adj[idx1, idx2] = 1.0
                    adj[idx2, idx1] = 1.0
            except ValueError:
                continue

    # Add self loops A_tilde = A + I
    adj_tilde = adj + np.eye(num_nodes, dtype=np.float32)

    # Compute symmetric degree normalization: D^-1/2 * A_tilde * D^-1/2
    row_sum = np.sum(adj_tilde, axis=1)
    d_inv_sqrt = np.zeros_like(row_sum)
    np.power(row_sum, -0.5, where=row_sum > 0, out=d_inv_sqrt)
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)

    adj_normalized = d_mat_inv_sqrt @ adj_tilde @ d_mat_inv_sqrt

    # Convert to torch tensors
    features_t = torch.tensor(features, dtype=torch.float32)
    adj_t = torch.tensor(adj_normalized, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    # Define standard training, val, and test splits
    # Cora has 2708 nodes. We use standard split:
    # Train: 140 nodes (20 nodes per class)
    # Val: 500 nodes
    # Test: 1000 nodes
    # Rest are ignored during optimization to preserve semi-supervised setup
    indices = np.arange(num_nodes)
    np.random.seed(42)
    np.random.shuffle(indices)

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    # Pick 20 nodes for each of the 7 classes
    class_counts = {c: 0 for c in range(7)}
    train_idx = []
    remaining_idx = []

    for idx in indices:
        c = labels[idx]
        if class_counts[c] < 20:
            train_idx.append(idx)
            class_counts[c] += 1
        else:
            remaining_idx.append(idx)

    train_mask[train_idx] = True
    val_mask[remaining_idx[:500]] = True
    test_mask[remaining_idx[500:1500]] = True

    splits = {
        "train_mask": torch.tensor(train_mask, dtype=torch.bool),
        "val_mask": torch.tensor(val_mask, dtype=torch.bool),
        "test_mask": torch.tensor(test_mask, dtype=torch.bool),
    }

    if limit is not None:
        # For quick limit tests, reduce dimensions
        features_t = features_t[:limit]
        adj_t = adj_t[:limit, :limit]
        labels_t = labels_t[:limit]
        splits["train_mask"] = splits["train_mask"][:limit]
        splits["val_mask"] = splits["val_mask"][:limit]
        splits["test_mask"] = splits["test_mask"][:limit]

    return features_t, adj_t, labels_t, splits


# --------------------------------------------------------------------------- #
# MUTAG Dataset Loader & Parser
# --------------------------------------------------------------------------- #
def load_mutag(data_dir: str = _DATA_DIR) -> list[dict]:
    """Load MUTAG dataset. Returns a list of dictionaries, where each dict has:

        - 'x': node features tensor [N, 7] (one-hot atom labels)
        - 'adj': adjacency matrix tensor [N, N] (normalized with self-loops)
        - 'y': graph class label (0 or 1)

    Graph Classification Task.
    """
    import zipfile
    mutag_dir = os.path.join(data_dir, "mutag")
    os.makedirs(mutag_dir, exist_ok=True)

    zip_path = os.path.join(mutag_dir, "MUTAG.zip")
    _download_file("https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip", zip_path)

    extracted_folder = os.path.join(mutag_dir, "MUTAG")
    if not os.path.exists(extracted_folder):
        print("Extracting MUTAG.zip...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(mutag_dir)

    paths = {name: os.path.join(extracted_folder, name) for name in MUTAG_FILES}

    # 1. Load Graph Indicator (maps 1-based node IDs to 1-based graph IDs)
    graph_indicator = []
    with open(paths["MUTAG_graph_indicator.txt"], "r") as f:
        for line in f:
            graph_indicator.append(int(line.strip()))
    graph_indicator = np.array(graph_indicator)
    num_nodes = len(graph_indicator)

    # 2. Load Node Labels (atoms categories, maps to one-hot vectors)
    node_labels = []
    with open(paths["MUTAG_node_labels.txt"], "r") as f:
        for line in f:
            node_labels.append(int(line.strip()))
    node_labels = np.array(node_labels)

    unique_node_labels = sorted(list(set(node_labels)))
    num_node_labels = len(unique_node_labels)
    # One-hot map
    node_features = np.zeros((num_nodes, num_node_labels), dtype=np.float32)
    for i, label in enumerate(node_labels):
        node_features[i, label] = 1.0

    # 3. Load Graph Labels (-1 or 1, map to 0 or 1)
    graph_labels = []
    with open(paths["MUTAG_graph_labels.txt"], "r") as f:
        for line in f:
            label = int(line.strip())
            graph_labels.append(0 if label == -1 else 1)
    graph_labels = np.array(graph_labels)
    num_graphs = len(graph_labels)

    # 4. Parse Adjacency list MUTAG_A.txt (contains directed edges, 1-based node IDs)
    # Map edges to graph index groupings
    graph_edges = {g: [] for g in range(1, num_graphs + 1)}
    with open(paths["MUTAG_A.txt"], "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if not parts:
                continue
            u = int(parts[0].strip())
            v = int(parts[1].strip())
            g = graph_indicator[u - 1] # u is 1-based
            graph_edges[g].append((u, v))

    # Slice indices and edges into separate graphs
    graphs = []
    for g in range(1, num_graphs + 1):
        # Nodes belonging to graph g
        node_indices = np.where(graph_indicator == g)[0]
        n_nodes = len(node_indices)

        # Local index mapping
        local_map = {global_idx + 1: local_idx for local_idx, global_idx in enumerate(node_indices)}

        x = node_features[node_indices]
        adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)

        for u, v in graph_edges[g]:
            # Map global node IDs to local indices in this graph
            lu = local_map[u]
            lv = local_map[v]
            adj[lu, lv] = 1.0
            adj[lv, lu] = 1.0

        # Add self-loops and normalize for GCN compatibility
        adj_tilde = adj + np.eye(n_nodes, dtype=np.float32)
        row_sum = np.sum(adj_tilde, axis=1)
        d_inv_sqrt = np.zeros_like(row_sum)
        np.power(row_sum, -0.5, where=row_sum > 0, out=d_inv_sqrt)
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)

        adj_normalized = d_mat_inv_sqrt @ adj_tilde @ d_mat_inv_sqrt

        graphs.append({
            "x": torch.tensor(x, dtype=torch.float32),
            "adj": torch.tensor(adj_normalized, dtype=torch.float32),
            "y": int(graph_labels[g - 1])
        })

    return graphs


# --------------------------------------------------------------------------- #
# Raw-graph helpers (for link prediction, random walks, spectral methods)
# --------------------------------------------------------------------------- #
def load_cora_raw(data_dir: str = _DATA_DIR):
    """Like load_cora but returns the RAW binary adjacency (symmetric, no self-loops)
    instead of the normalized one. Returns (features, adj_binary, labels)."""
    cora_dir = os.path.join(data_dir, "cora")
    os.makedirs(cora_dir, exist_ok=True)
    content_path = os.path.join(cora_dir, "cora.content")
    cites_path = os.path.join(cora_dir, "cora.cites")
    _download_file(CORA_CONTENT_URL, content_path)
    _download_file(CORA_CITES_URL, cites_path)

    features, labels_str, idx_map = [], [], {}
    with open(content_path, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue
            idx_map[int(parts[0])] = i
            features.append([float(x) for x in parts[1:-1]])
            labels_str.append(parts[-1])
    features = np.array(features, dtype=np.float32)
    label_map = {l: i for i, l in enumerate(sorted(set(labels_str)))}
    labels = np.array([label_map[l] for l in labels_str], dtype=np.int64)

    n = len(idx_map)
    adj = np.zeros((n, n), dtype=np.float32)
    with open(cites_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and int(parts[0]) in idx_map and int(parts[1]) in idx_map:
                a, b = idx_map[int(parts[0])], idx_map[int(parts[1])]
                if a != b:
                    adj[a, b] = adj[b, a] = 1.0
    return (torch.tensor(features), torch.tensor(adj), torch.tensor(labels))


def normalize_adj(adj_binary, self_loops: bool = True):
    """Symmetric normalization D^-1/2 (A[+I]) D^-1/2 for a torch binary adjacency."""
    a = adj_binary.clone()
    if self_loops:
        a = a + torch.eye(a.size(0), device=a.device)
    deg = a.sum(1)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d = torch.diag(d_inv_sqrt)
    return d @ a @ d


def split_link_prediction_edges(adj_binary, test_frac: float = 0.1, seed: int = 42):
    """Split positive edges into train/test and sample test negatives (non-edges).

    Returns (train_adj_binary[torch], test_pos[np int (E,2)], test_neg[np int (E,2)]).
    The encoder should only see ``train_adj_binary`` so test edges stay unseen.
    """
    a = adj_binary.cpu().numpy()
    n = a.shape[0]
    rng = np.random.default_rng(seed)
    pos = np.argwhere(np.triu(a, 1) > 0)
    rng.shuffle(pos)
    n_test = max(1, int(len(pos) * test_frac))
    test_pos, train_pos = pos[:n_test], pos[n_test:]

    train_adj = np.zeros_like(a)
    for i, j in train_pos:
        train_adj[i, j] = train_adj[j, i] = 1.0

    neg = set()
    while len(neg) < n_test:
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i != j and a[i, j] == 0 and (i, j) not in neg and (j, i) not in neg:
            neg.add((i, j))
    test_neg = np.array(list(neg))
    return torch.tensor(train_adj, dtype=torch.float32), test_pos, test_neg


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


def build_argparser(description: str, epochs: int = 100, lr: float = 1e-2):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--lr", type=float, default=lr)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--variant", type=str, default=None)
    p.add_argument("--limit", type=int, default=None, help="limit node count for quick checks")
    return p


# --------------------------------------------------------------------------- #
# Visualizations: t-SNE Embedding Projection
# --------------------------------------------------------------------------- #
def plot_tsne_embeddings(embeddings: torch.Tensor, labels: torch.Tensor, save_path: str,
                         title: str = "t-SNE Projection of Cora GNN Hidden Node Embeddings",
                         legend_title: str = "Subject Categories"):
    """Projects high-dimensional node or graph embeddings into 2D using t-SNE and saves a scatter plot."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping t-SNE plot: {e})")
        return

    print("Computing t-SNE projection of embeddings...")
    emb_np = embeddings.detach().cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Adjust perplexity dynamically if number of samples is extremely small
    perplexity = min(30, max(5, len(emb_np) // 4))

    # Fit t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    emb_2d = tsne.fit_transform(emb_np)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6.5))
    scatter = ax.scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=labels_np, cmap="tab10",
        alpha=0.8, edgecolors="black", linewidths=0.3, s=20
    )
    legend = ax.legend(*scatter.legend_elements(), title=legend_title, loc="upper right")
    ax.add_artist(legend)
    ax.set_title(title)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved t-SNE embeddings plot -> {save_path}")

