"""
12. MLP with Entity Embeddings (Guo & Berkhahn, 2016)
=====================================================

The foundational deep-tabular idea: instead of feeding a high-cardinality
categorical feature to the network as a sparse one-hot vector, map each category
to a small *learned* dense vector (an "entity embedding"), exactly like word
embeddings in NLP. Numeric features are standardized and passed through directly.
Everything then feeds a plain multilayer perceptron.

On Cover Type the binary wilderness-area (4 levels) and soil-type (40 levels)
columns arrive one-hot; this script folds each block back into a single
categorical index and embeds it, so the network learns a geometry over soil
types rather than treating all 40 as orthogonal.

Architecture Diagram / Layout:
    numeric x_num [N, 10] ---------------------------\
    wilderness id -> Embed(4  -> e1) ----------------->  concat -> MLP -> softmax
    soil id       -> Embed(40 -> e2) ----------------/   [hidden, hidden] -> [C]

Variants (`--variant`):
    embed        : entity embeddings for categoricals (the paper's model)
    plain        : no embeddings — raw one-hot columns into the same MLP (the
                   ablation showing what embeddings buy)
    learnable-act: the `embed` model but with a learnable per-channel activation
                   (a small learnable basis, spline-like) replacing ReLU. This is
                   the control for the KAN comparison (script 17): it isolates
                   whether a learnable *activation* helps, separately from KAN's
                   edge-wise topology. See Yu et al., "KAN or MLP: A Fairer
                   Comparison" (2024).

Key insights / educational takeaways:
    * Learned embeddings capture similarity between categories and share
      statistical strength; one-hots cannot.
    * This script prints its exact trainable-parameter count and per-epoch time
      so it can be parameter-matched against the KAN (script 17). At KAN defaults
      (grid G=5, order k=3) each KAN edge holds ~G+k+2 = 10 coefficients vs. 1
      per MLP weight, so a KAN-matched MLP is ~3x wider (raise --hidden).

Run:
    python "12.mlp-embeddings.py" --variant embed --dataset covtype --epochs 30
    python "12.mlp-embeddings.py" --variant plain --dataset covtype
    python "12.mlp-embeddings.py" --limit 2000 --epochs 2        # fast smoke test
"""

import os
import numpy as np
import torch
import torch.nn as nn
import tabular_common as mc


# --------------------------------------------------------------------------- #
# Learnable per-channel activation (spline-like basis) for the KAN ablation.
# --------------------------------------------------------------------------- #
class LearnableActivation(nn.Module):
    """Per-channel learnable mixture over a fixed nonlinear basis.

    act(x)_c = a_c * SiLU(x) + b_c * tanh(x) + c_c * x
    Initialized to SiLU (a=1, b=c=0) so training starts from a sane nonlinearity
    and each channel can then reshape its own activation — the cheap stand-in for
    KAN's learnable spline activations.
    """
    def __init__(self, dim):
        super().__init__()
        self.a = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.c = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.a * torch.nn.functional.silu(x) + self.b * torch.tanh(x) + self.c * x


class EmbeddingMLP(nn.Module):
    def __init__(self, n_numeric, cat_cardinalities, n_classes, hidden=128,
                 depth=2, learnable_act=False):
        super().__init__()
        self.n_numeric = n_numeric
        self.embeddings = nn.ModuleList()
        emb_dim_total = 0
        for card in cat_cardinalities:
            d = min(16, (card + 1) // 2)  # embedding size heuristic
            self.embeddings.append(nn.Embedding(card, d))
            emb_dim_total += d

        in_dim = n_numeric + emb_dim_total
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(LearnableActivation(hidden) if learnable_act else nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden))
            in_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        num = x[:, :self.n_numeric]
        parts = [num]
        for i, emb in enumerate(self.embeddings):
            idx = x[:, self.n_numeric + i].long()
            parts.append(emb(idx))
        h = torch.cat(parts, dim=1) if len(parts) > 1 else num
        return self.head(self.backbone(h))


# --------------------------------------------------------------------------- #
# Feature preparation: build model inputs per variant.
# --------------------------------------------------------------------------- #
def prepare_features(X_train, X_test, dataset, variant):
    """Return (Xtr, Xte, n_numeric, cat_cardinalities).

    For covtype in embedding mode, fold the wilderness/soil one-hot blocks into
    integer indices appended after the standardized numeric columns. Otherwise
    every column is treated as numeric (plain one-hot MLP).
    """
    use_embed = (variant != "plain") and (dataset == "covtype")
    if not use_embed:
        Xtr, Xte = mc.standardize(X_train, X_test)
        return Xtr, Xte, Xtr.shape[1], []

    num = mc.COVTYPE_N_QUANT
    wild = mc.COVTYPE_WILDERNESS_COLS
    soil = mc.COVTYPE_SOIL_COLS

    def build(X):
        num_std = X[:, :num]
        wild_idx = np.argmax(X[:, wild], axis=1)
        soil_idx = np.argmax(X[:, soil], axis=1)
        return num_std, np.stack([wild_idx, soil_idx], axis=1)

    ntr, ctr = build(X_train)
    nte, cte = build(X_test)
    # Standardize only the numeric block (fit on train).
    ntr_s, nte_s = mc.standardize(ntr, nte)
    Xtr = np.concatenate([ntr_s, ctr.astype(np.float32)], axis=1)
    Xte = np.concatenate([nte_s, cte.astype(np.float32)], axis=1)
    return Xtr.astype(np.float32), Xte.astype(np.float32), num, [len(wild), len(soil)]


def main():
    p = mc.build_argparser("MLP with Entity Embeddings", lr=1e-3)
    p.add_argument("--variant", choices=["embed", "plain", "learnable-act"],
                   default="embed")
    p.add_argument("--hidden", type=int, default=128, help="MLP hidden width")
    p.add_argument("--depth", type=int, default=2, help="number of hidden layers")
    args = p.parse_args()
    mc.set_seed(args.seed)
    device = mc.get_device(args.device)

    X_train, X_test, y_train, y_test, feature_names, class_names = \
        mc.load_classification_dataset(args.dataset)
    if args.limit is not None:
        X_train, y_train = X_train[:args.limit], y_train[:args.limit]

    Xtr, Xte, n_numeric, cards = prepare_features(X_train, X_test, args.dataset,
                                                  args.variant)
    print(f"Variant={args.variant} | numeric={n_numeric} | "
          f"categorical cardinalities={cards}")

    train_loader, test_loader = mc.get_dataloaders_from_arrays(
        Xtr, y_train, Xte, y_test, batch_size=args.batch_size)

    model = EmbeddingMLP(n_numeric, cards, len(class_names), hidden=args.hidden,
                         depth=args.depth, learnable_act=(args.variant == "learnable-act"))

    mc.train(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr,
             device=device)
    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report_classification(y_true, y_pred, y_prob, class_names=class_names,
                             model_name=f"MLP-{args.variant}",
                             save_dir=None if args.no_figure else
                             os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
