"""
08. Node2vec / DeepWalk — random-walk node embeddings (Perozzi 2014; Grover & Leskovec 2016)
============================================================================================

The pre-GNN way to learn from graph structure. Treat the graph like a corpus:
generate many random walks (sentences of nodes), then train skip-gram embeddings
so that nodes appearing near each other in walks get similar vectors. Crucially it
uses **no node features** — all signal comes from connectivity — yet the learned
embeddings already cluster Cora's classes well (homophily).

    DeepWalk      = unbiased random walks (p = q = 1).
    Node2vec      = biased walks with return (p) and in-out (q) parameters that
                    interpolate between BFS-like (structural) and DFS-like
                    (homophily) exploration (`--variant "p,q"`, e.g. "1,0.5").

Pipeline: walks -> skip-gram with negative sampling -> linear probe on the train
split for node-classification accuracy + a t-SNE of the embeddings.

Run:
    python "08.node2vec.py" --epochs 100
    python "08.node2vec.py" --variant 1,0.5 --epochs 100
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import gnn_common as mc


def generate_walks(neighbors, num_walks, walk_length, p, q):
    nodes = list(range(len(neighbors)))
    walks = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for start in nodes:
            if not neighbors[start]:
                continue
            walk = [start]
            while len(walk) < walk_length:
                cur = walk[-1]
                nbrs = neighbors[cur]
                if not nbrs:
                    break
                if len(walk) == 1 or (p == 1 and q == 1):
                    walk.append(random.choice(nbrs))
                else:                                        # node2vec 2nd-order bias
                    prev = walk[-2]
                    weights = []
                    for nb in nbrs:
                        if nb == prev:
                            weights.append(1.0 / p)
                        elif nb in neighbors[prev]:
                            weights.append(1.0)
                        else:
                            weights.append(1.0 / q)
                    walk.append(random.choices(nbrs, weights=weights)[0])
            walks.append(walk)
    return walks


def main():
    p_arg = mc.build_argparser("Node2vec / DeepWalk", epochs=100, lr=1e-2)
    args = p_arg.parse_args()
    p, q = (float(v) for v in args.variant.split(",")) if args.variant else (1.0, 1.0)
    device = mc.get_device(args.device)

    _, adj_raw, labels = mc.load_cora_raw()
    n = adj_raw.size(0)
    neighbors = [torch.nonzero(adj_raw[i]).flatten().tolist() for i in range(n)]

    print(f"Generating random walks (p={p}, q={q})...")
    walks = generate_walks(neighbors, num_walks=10, walk_length=40, p=p, q=q)

    # Build (center, context) pairs within a window, capped for speed
    window, max_pairs = 5, 800000
    pairs = []
    for walk in walks:
        for i, c in enumerate(walk):
            for j in range(max(0, i - window), min(len(walk), i + window + 1)):
                if i != j:
                    pairs.append((c, walk[j]))
    random.shuffle(pairs)
    pairs = np.array(pairs[:max_pairs], dtype=np.int64)
    print(f"{len(walks)} walks -> {len(pairs):,} training pairs")

    dim, n_neg, batch = 128, 5, 1024
    emb = nn.Embedding(n, dim).to(device)
    ctx = nn.Embedding(n, dim).to(device)
    nn.init.uniform_(emb.weight, -0.5 / dim, 0.5 / dim)
    optimizer = torch.optim.Adam(list(emb.parameters()) + list(ctx.parameters()), lr=args.lr)

    passes = max(1, args.epochs // 30)
    print(f"Training skip-gram (neg-sampling) for {passes} pass(es)...")
    print("-" * 64)
    for ep in range(passes):
        perm = np.random.permutation(len(pairs))
        total = 0.0
        for k in range(0, len(pairs) - batch, batch):
            idx = perm[k:k + batch]
            c = torch.tensor(pairs[idx, 0], device=device)
            o = torch.tensor(pairs[idx, 1], device=device)
            neg = torch.randint(0, n, (len(idx), n_neg), device=device)
            ec, eo, en = emb(c), ctx(o), ctx(neg)
            pos = torch.nn.functional.logsigmoid((ec * eo).sum(1))
            negl = torch.nn.functional.logsigmoid(-(en * ec.unsqueeze(1)).sum(2)).sum(1)
            loss = -(pos + negl).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        print(f"Pass {ep + 1}/{passes} | skip-gram loss {total / max(1, len(pairs)//batch):.4f}")
    print("-" * 64)

    embeddings = emb.weight.detach()
    emb_np = embeddings.cpu().numpy()
    labels_np = labels.numpy()
    _, _, _, splits = mc.load_cora()
    train_mask = splits["train_mask"].numpy(); test_mask = splits["test_mask"].numpy()
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=500).fit(emb_np[train_mask], labels_np[train_mask])
        print(f"Node2vec linear-probe Test Accuracy: {clf.score(emb_np[test_mask], labels_np[test_mask]):.4f}")
    except Exception as e:
        print(f"(skipping linear probe: {e})")

    save_dir = os.path.dirname(os.path.abspath(__file__))
    mc.plot_tsne_embeddings(embeddings, labels, os.path.join(save_dir, "node2vec_cora_tsne.png"),
                            title="t-SNE of Node2vec embeddings (structure only, Cora)")


if __name__ == "__main__":
    main()
