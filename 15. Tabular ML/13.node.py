"""
13. Neural Oblivious Decision Ensembles — NODE (Popov, Morozov & Babenko, 2019)
===============================================================================

The conceptual bridge between the decision trees of Phases 1-2 and neural
networks. NODE builds an ensemble of *oblivious* decision trees — trees that use
the **same feature and threshold at every node of a given depth level** — and
makes them fully differentiable so the whole ensemble trains end-to-end by
gradient descent.

An oblivious tree of depth D is defined by, for each of its D levels:
  (1) a soft choice of which input feature to test, and
  (2) a learned threshold + temperature.
Because every level is a single soft split shared across the level, a depth-D
tree has exactly 2^D leaves and its leaf-assignment is the outer product of D
independent soft binary decisions — cheap and differentiable.

Architecture Diagram / Layout:
    x [N, F]
      |  per (tree, level): feature choice = softmax(selection_logits) over F
      |                     response  = x . choice          [N, T, D]
      |                     decide    = sigmoid((response - threshold) * temp)
      |  outer-product the D soft decisions -> leaf weights  [N, T, 2^D]
      |  leaf responses                                      [T, 2^D, C]
      -> per-tree output = leaf_weights . leaf_responses     [N, T, C]
      -> average over the T trees                            -> logits [N, C]

Deviation from the paper (documented per SPEC): the paper selects features and
thresholds with the sparse `entmax15` transform; this compact implementation
uses plain softmax for feature selection and a sigmoid split, which keeps the
code short and stable while preserving the oblivious-tree mechanism. The result
is soft (not sparse) feature selection.

Key insights / educational takeaways:
    * A differentiable ensemble of trees: the split thresholds are learned by
      backprop instead of greedy gain search — trees you can drop into an
      autograd graph.
    * Oblivious trees are deliberately weak/regularized individually (one split
      per level); the ensemble average is what carries the accuracy.
    * Sits conceptually between GBDTs (scripts 03/10/11) and the attention-based
      deep models that follow (14-15).

Run:
    python "13.node.py" --dataset covtype --epochs 30
    python "13.node.py" --limit 2000 --epochs 2        # fast smoke test
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import tabular_common as mc


class ODST(nn.Module):
    """One layer of Oblivious Differentiable Sparse Trees (softmax variant)."""
    def __init__(self, in_features, n_trees, depth, tree_dim):
        super().__init__()
        self.n_trees, self.depth, self.tree_dim = n_trees, depth, tree_dim
        self.feature_logits = nn.Parameter(torch.randn(in_features, n_trees, depth))
        self.thresholds = nn.Parameter(torch.randn(n_trees, depth))
        self.log_temp = nn.Parameter(torch.zeros(n_trees, depth))
        self.leaf_response = nn.Parameter(
            torch.randn(n_trees, 2 ** depth, tree_dim) * 0.1)

    def forward(self, x):
        # Soft feature selection per (tree, level).
        choice = F.softmax(self.feature_logits, dim=0)          # (F, T, D)
        response = torch.einsum("bf,ftd->btd", x, choice)        # (B, T, D)
        decide = torch.sigmoid((response - self.thresholds) * torch.exp(-self.log_temp))
        probs = torch.stack([1.0 - decide, decide], dim=-1)      # (B, T, D, 2)

        # Outer-product the D soft binary decisions into 2^D leaf weights.
        leaf_w = torch.ones(x.size(0), self.n_trees, 1, device=x.device)
        for d in range(self.depth):
            leaf_w = (leaf_w.unsqueeze(-1) * probs[:, :, d, :].unsqueeze(-2))
            leaf_w = leaf_w.reshape(x.size(0), self.n_trees, -1)  # (B, T, 2^(d+1))

        out = torch.einsum("btl,tlk->btk", leaf_w, self.leaf_response)  # (B, T, C)
        return out.mean(dim=1)                                   # (B, C)


class NODE(nn.Module):
    def __init__(self, in_features, n_classes, n_trees=128, depth=4):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.odst = ODST(in_features, n_trees, depth, n_classes)

    def forward(self, x):
        return self.odst(self.bn(x))


def main():
    p = mc.build_argparser("Neural Oblivious Decision Ensembles (NODE)", lr=1e-2)
    p.add_argument("--n-trees", type=int, default=128)
    p.add_argument("--depth", type=int, default=4, help="oblivious tree depth (2^depth leaves)")
    args = p.parse_args()
    mc.set_seed(args.seed)
    device = mc.get_device(args.device)

    X_train, X_test, y_train, y_test, feature_names, class_names = \
        mc.load_classification_dataset(args.dataset)
    if args.limit is not None:
        X_train, y_train = X_train[:args.limit], y_train[:args.limit]
    Xtr, Xte = mc.standardize(X_train, X_test)

    train_loader, test_loader = mc.get_dataloaders_from_arrays(
        Xtr, y_train, Xte, y_test, batch_size=args.batch_size)

    model = NODE(Xtr.shape[1], len(class_names), n_trees=args.n_trees, depth=args.depth)
    mc.train(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr,
             device=device)
    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report_classification(y_true, y_pred, y_prob, class_names=class_names,
                             model_name="NODE",
                             save_dir=None if args.no_figure else
                             os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
