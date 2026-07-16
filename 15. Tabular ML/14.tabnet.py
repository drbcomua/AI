"""
14. TabNet (Arik & Pfister, 2019)
=================================

TabNet brings *instance-wise feature selection* to tabular deep learning. It
processes the input in several sequential "decision steps"; at each step an
**attentive transformer** emits a sparse feature mask (via sparsemax) that
decides which features this step is allowed to look at, for *this specific
sample*. A prior tracks how much each feature has already been used so later
steps are nudged toward fresh features. The masks, aggregated over steps, form a
built-in feature-importance explanation you can compare directly against the
tree importances of Phases 1-2.

Architecture Diagram / Layout:
    x [N, F] -> initial BN
    prior = 1
    for step s = 1..S:
        mask_s = sparsemax(prior * AttentiveTransformer(a))     [N, F]  (sparse)
        prior  = prior * (gamma - mask_s)                       (usage penalty)
        f_s    = FeatureTransformer(mask_s * x)  -> [decision d_s | attention a]
        output += ReLU(d_s)                                     (accumulate decisions)
    logits = Linear(output)
    aggregate_mask = sum_s mask_s (weighted) -> per-feature importance

Simplifications (documented per SPEC):
    * Ghost Batch Normalization is omitted (plain BatchNorm1d) — it mainly helps
      very large batches; noted here for faithfulness.
    * The sparsity-entropy regularization term on the masks is dropped so the
      script can reuse the shared cross-entropy training loop; sparsemax already
      yields genuinely sparse masks.

Key insights / educational takeaways:
    * sparsemax (Martins & Astudillo, 2016) is the star: unlike softmax it can
      output exact zeros, so each step attends to a hard subset of features.
    * Instance-wise selection: different samples get different feature masks —
      strictly more expressive than a single global importance ranking.
    * The aggregate mask figure (saved by this script) is TabNet's answer to
      tree feature importances — overlay it mentally on scripts 02/08/11.

Run:
    python "14.tabnet.py" --dataset covtype --epochs 30
    python "14.tabnet.py" --limit 2000 --epochs 2        # fast smoke test
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tabular_common as mc


def sparsemax(z):
    """Sparsemax over the last dimension (Martins & Astudillo, 2016)."""
    z_sorted, _ = torch.sort(z, dim=-1, descending=True)
    k = torch.arange(1, z.size(-1) + 1, device=z.device, dtype=z.dtype)
    z_cumsum = z_sorted.cumsum(dim=-1)
    support = (1 + k * z_sorted) > z_cumsum          # which top-k stay in support
    k_support = support.sum(dim=-1, keepdim=True).clamp(min=1)
    tau = (z_cumsum.gather(-1, k_support - 1) - 1) / k_support
    return torch.clamp(z - tau, min=0.0)


class GLUBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 2 * out_dim)
        self.bn = nn.BatchNorm1d(2 * out_dim)

    def forward(self, x):
        return F.glu(self.bn(self.fc(x)), dim=-1)


class FeatureTransformer(nn.Module):
    """Map raw/masked features -> [decision | attention] representation."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.b1 = GLUBlock(in_dim, out_dim)
        self.b2 = GLUBlock(out_dim, out_dim)

    def forward(self, x):
        h = self.b1(x)
        return (h + self.b2(h)) * (0.5 ** 0.5)   # normalized residual


class TabNet(nn.Module):
    def __init__(self, in_features, n_classes, n_d=32, n_a=32, n_steps=3, gamma=1.3):
        super().__init__()
        self.in_features = in_features
        self.n_d, self.n_a, self.n_steps, self.gamma = n_d, n_a, n_steps, gamma
        self.bn = nn.BatchNorm1d(in_features)
        self.feat = FeatureTransformer(in_features, n_d + n_a)
        self.attentive = nn.Sequential(
            nn.Linear(n_a, in_features), nn.BatchNorm1d(in_features))
        self.head = nn.Linear(n_d, n_classes)

    def _forward(self, x, collect_masks=False):
        x = self.bn(x)
        prior = torch.ones_like(x)
        a = self.feat(x)[:, self.n_d:]           # initial attention features
        out = torch.zeros(x.size(0), self.n_d, device=x.device)
        agg = torch.zeros_like(x) if collect_masks else None

        for _ in range(self.n_steps):
            mask = sparsemax(prior * self.attentive(a))
            prior = prior * (self.gamma - mask)
            f = self.feat(mask * x)
            d = F.relu(f[:, :self.n_d])
            out = out + d
            a = f[:, self.n_d:]
            if collect_masks:
                agg = agg + mask * d.sum(dim=1, keepdim=True)
        logits = self.head(out)
        return (logits, agg) if collect_masks else logits

    def forward(self, x):
        return self._forward(x)

    @torch.no_grad()
    def aggregate_mask(self, loader, device):
        self.eval()
        total = None
        for x, _ in loader:
            _, agg = self._forward(x.to(device), collect_masks=True)
            s = agg.sum(dim=0).cpu().numpy()
            total = s if total is None else total + s
        return total


def main():
    p = mc.build_argparser("TabNet", lr=2e-2)
    p.add_argument("--n-steps", type=int, default=3)
    p.add_argument("--n-d", type=int, default=32, help="decision dimension")
    p.add_argument("--gamma", type=float, default=1.3, help="feature reuse penalty")
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

    model = TabNet(Xtr.shape[1], len(class_names), n_d=args.n_d, n_a=args.n_d,
                   n_steps=args.n_steps, gamma=args.gamma)
    mc.train(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr,
             device=device)
    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report_classification(y_true, y_pred, y_prob, class_names=class_names,
                             model_name="TabNet",
                             save_dir=None if args.no_figure else _here())

    if not args.no_figure:
        masks = model.aggregate_mask(test_loader, device)
        mc.plot_feature_importances(
            np.clip(masks, 0, None), feature_names,
            os.path.join(_here(), "tabnet_feature_masks.png"),
            "TabNet Aggregate Feature Masks (instance-wise attention)")


def _here():
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    main()
