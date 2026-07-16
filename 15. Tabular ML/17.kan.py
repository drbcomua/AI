"""
17. Kolmogorov-Arnold Network — KAN (Liu et al., 2024)
======================================================

KANs move the nonlinearity from the *nodes* to the *edges*. A standard MLP puts
a fixed activation (ReLU) on each neuron and learns linear weights on each edge:

    MLP edge:  x -> w * x                    (learn one scalar w)
    MLP node:  sum -> ReLU(sum)              (fixed activation)

A KAN instead puts a *learnable univariate function* on every edge — implemented
here as a B-spline plus a residual SiLU term — and simply sums at the nodes:

    KAN edge:  x -> phi(x) = base_w * SiLU(x) + sum_i c_i * B_i(x)   (learn c_i, base_w)
    KAN node:  sum of incoming phi(x)         (no separate activation)

The B-spline basis B_i is defined on a grid (grid size G, order k); each edge
carries G+k spline coefficients plus a base weight, so a KAN edge holds ~G+k+2
learnable numbers where an MLP edge holds 1.

Architecture Diagram / Layout:
    x [N, F]
      -> BatchNorm (keep inputs inside the spline grid range)
      -> KANLinear(F -> H):   each of F*H edges is a learned 1-D spline
      -> BatchNorm
      -> KANLinear(H -> C):   each of H*C edges is a learned 1-D spline
      -> logits [N, C]

This script produces a figure no other script does: the **learned univariate
spline on each input feature** (first-layer edge functions averaged over hidden
units) — "how the network responds to each feature", KAN's interpretability
selling point.

HONEST TAKEAWAY (per Yu et al., "KAN or MLP: A Fairer Comparison", 2024):
    On tabular benchmarks, a *parameter-matched* MLP equals or beats a KAN, and
    KANs train ~2-5x slower per epoch. KAN's real advantage is interpretability
    and symbolic/function-fitting tasks (see `18. Symbolic Regression/`), NOT
    tabular accuracy. Do not read a KAN win here without matching parameters and
    seeds first.

KAN-vs-MLP fair-comparison protocol (shared with script 12):
    * Parameter accounting: this script prints its exact trainable-parameter
      count. Each KAN edge holds ~G+k+2 coefficients vs. 1 per MLP weight, so to
      parameter-match you must WIDEN script 12's MLP until its printed parameter
      count equals this one's (Yu et al. report the matched MLP is ~3x wider at
      G=5, k=3). Compare printed counts, not layer widths.
    * Seeds: pass --seed and average over >=10 seeds before any accuracy claim on
      these small datasets.
    * Cost: the shared training loop prints wall-clock seconds/epoch; expect KAN
      to be several times slower than the script-12 MLP at matched parameters.

Run:
    python "17.kan.py" --dataset covtype --epochs 30 --hidden 32
    python "17.kan.py" --limit 2000 --epochs 2        # fast smoke test
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tabular_common as mc


class KANLinear(nn.Module):
    """A KAN layer: a learnable B-spline (+ SiLU residual) on every edge.

    efficient-kan style. spline_weight holds (out, in, G+k) coefficients; the
    base term gives a well-behaved gradient path like a normal linear+SiLU layer.
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 grid_range=(-3.0, 3.0)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0])
        self.register_buffer("grid", grid.expand(in_features, -1).contiguous())

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order))
        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)
        nn.init.normal_(self.spline_weight, std=0.1 / (grid_size + spline_order))

    def b_splines(self, x):
        """Cox-de Boor recursion. x: (batch, in) -> (batch, in, G+k) bases."""
        grid = self.grid                                   # (in, G+2k+1)
        x = x.unsqueeze(-1)                                # (batch, in, 1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)])
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases                                       # (batch, in, G+k)

    def forward(self, x):
        base = F.linear(F.silu(x), self.base_weight)       # (batch, out)
        bases = self.b_splines(x)                          # (batch, in, G+k)
        spline = torch.einsum("bic,oic->bo", bases, self.spline_weight)
        return base + spline

    @torch.no_grad()
    def edge_response(self, feature_idx, xs):
        """Average spline response phi(x) for one input feature over all outputs.

        xs: (M,) 1-D grid of feature values. Returns (M,) mean edge response —
        used for the per-feature interpretability figure.
        """
        x = torch.zeros(len(xs), self.in_features)
        x[:, feature_idx] = xs
        bases = self.b_splines(x)[:, feature_idx, :]       # (M, G+k)
        resp = bases @ self.spline_weight[:, feature_idx, :].T   # (M, out)
        base = F.silu(xs).unsqueeze(1) * self.base_weight[:, feature_idx]
        return (resp + base).mean(dim=1).cpu().numpy()


class KAN(nn.Module):
    def __init__(self, in_features, n_classes, hidden=32, grid_size=5, spline_order=3):
        super().__init__()
        self.bn_in = nn.BatchNorm1d(in_features)
        self.layer1 = KANLinear(in_features, hidden, grid_size, spline_order)
        self.bn_hid = nn.BatchNorm1d(hidden)
        self.layer2 = KANLinear(hidden, n_classes, grid_size, spline_order)

    def forward(self, x):
        x = self.bn_in(x)
        x = self.layer1(x)
        x = self.bn_hid(x)
        return self.layer2(x)


def plot_feature_splines(model, feature_names, save_path, max_features=16):
    """Plot the learned univariate spline on each input feature (first layer)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping KAN spline plot: {e})")
        return

    n = min(model.layer1.in_features, max_features)
    xs = torch.linspace(-3, 3, 200)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.2 * rows),
                             squeeze=False)
    for j in range(n):
        ax = axes[j // cols][j % cols]
        ax.plot(xs.numpy(), model.layer1.edge_response(j, xs), color="crimson", lw=1.5)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_title(feature_names[j][:18], fontsize=8)
        ax.tick_params(labelsize=6)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("KAN learned univariate response per input feature (layer 1)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved KAN per-feature spline figure -> {save_path}")


def main():
    p = mc.build_argparser("Kolmogorov-Arnold Network (KAN)", lr=1e-3)
    p.add_argument("--hidden", type=int, default=32, help="KAN hidden width")
    p.add_argument("--grid-size", type=int, default=5, help="B-spline grid size G")
    p.add_argument("--spline-order", type=int, default=3, help="B-spline order k")
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

    model = KAN(Xtr.shape[1], len(class_names), hidden=args.hidden,
                grid_size=args.grid_size, spline_order=args.spline_order)
    print(f"KAN edge coefficients per edge: G+k = {args.grid_size + args.spline_order} "
          f"spline + 1 base (match script 12's MLP by trainable-param count below)")

    mc.train(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr,
             device=device)
    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report_classification(y_true, y_pred, y_prob, class_names=class_names,
                             model_name="KAN",
                             save_dir=None if args.no_figure else _here())

    if not args.no_figure:
        model.to("cpu")
        plot_feature_splines(model, feature_names,
                             os.path.join(_here(), "kan_feature_splines.png"))


def _here():
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    main()
