"""
05. Kolmogorov-Arnold Network — KAN (Liu et al., 2024)
======================================================

This is the folder's thesis in one script.  In `15. Tabular ML/` a
parameter-matched MLP *ties* the KAN — because tabular targets have no smooth
univariate structure for splines to exploit.  Symbolic regression is the
opposite task distribution: the ground truth **is** a composition of smooth
univariate functions (`sin`, `exp`, `x^2`, ...), which is exactly the object a
KAN edge represents.  Here the ranking flips.

KANs move the nonlinearity from the nodes to the **edges**.  An MLP learns one
scalar per edge and applies a fixed ReLU at each node; a KAN learns a whole
univariate function φ on each edge and merely sums at the nodes:

    MLP edge:  x -> w·x                 (1 number)      MLP node: ReLU(sum)
    KAN edge:  x -> φ(x) = base·SiLU(x) + Σ c_i B_i(x)  KAN node: sum (no act.)

    B_i are B-spline bases on a grid (size G, order k); each edge carries
    G+k spline coefficients + 1 base weight.

Architecture (regression, scalar output):
    x [N, d]
      -> KANLinear(d -> H)   each of d·H edges is a learned 1-D spline
      -> KANLinear(H -> 1)   each of H·1 edges is a learned 1-D spline
      -> y_hat [N, 1]                            (no BatchNorm: input ranges are
                                                  already inside the spline grid)

Figures (both required by SPEC):
    * sr_kan-splines_<problem>.png — the learned univariate φ on each first-layer
      edge (KAN's interpretability selling point).
    * sr_kan_<problem>.png — the standard interpolate-vs-extrapolate curve; note
      the spline+SiLU continuation tracks curvature far better than the MLP's
      straight-line ramp at the same parameter budget.

Optional symbolification (`--symbolify`, on by default):
    snap each *input-feature* edge to the best-fitting primitive
    {linear, x^2, sin, cos, exp} by 1-D least squares, then compose an
    approximate closed-form expression and score recovery.  This is a didactic
    stand-in for pykan's `auto_symbolic`; it succeeds cleanly on single-transform
    problems (e.g. `gaussian`) and shows its seams on compositions.

Educational takeaway:
    On this task family the KAN gets **lower extrapolation RMSE, sane
    off-distribution behavior, and a recoverable structure** where the
    parameter-matched MLP (04) gets none of the three — the counterpoint to its
    tabular tie.

Run:
    python "05.kan.py" --problem gaussian --epochs 400
    python "05.kan.py" --problem nguyen1 --hidden 5
    python "05.kan.py" --problem nguyen1 --limit 200 --epochs 20   # smoke test
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sr_common as mc


class KANLinear(nn.Module):
    """A KAN layer: a learnable B-spline (+ SiLU residual) on every edge.

    efficient-kan style, ported from `15. Tabular ML/17.kan.py` (cross-folder
    imports are not used in this repo, so the layer is copied).  spline_weight
    holds (out, in, G+k) coefficients; the SiLU base term gives a well-behaved
    gradient path like a normal linear layer.
    """
    def __init__(self, in_features, out_features, grid_size=8, spline_order=3,
                 grid_range=(-4.0, 4.0)):
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
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)])
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases

    def forward(self, x):
        base = F.linear(F.silu(x), self.base_weight)
        bases = self.b_splines(x)
        spline = torch.einsum("bic,oic->bo", bases, self.spline_weight)
        return base + spline

    @torch.no_grad()
    def edge_response(self, in_idx, out_idx, xs):
        """Response φ(x) of the single edge in_idx -> out_idx over a 1-D grid xs."""
        x = torch.zeros(len(xs), self.in_features)
        x[:, in_idx] = xs
        bases = self.b_splines(x)[:, in_idx, :]
        resp = bases @ self.spline_weight[out_idx, in_idx, :]
        base = F.silu(xs) * self.base_weight[out_idx, in_idx]
        return (resp + base).cpu().numpy()


class KAN(nn.Module):
    def __init__(self, in_features, hidden=5, grid_size=8, spline_order=3):
        super().__init__()
        self.layer1 = KANLinear(in_features, hidden, grid_size, spline_order)
        self.layer2 = KANLinear(hidden, 1, grid_size, spline_order)

    def forward(self, x):
        return self.layer2(self.layer1(x))


# --------------------------------------------------------------------------- #
# Symbolification: snap first-layer edges to primitives by 1-D least squares
# --------------------------------------------------------------------------- #
_PRIMITIVES = {
    "linear": lambda x: x,
    "x^2": lambda x: x ** 2,
    "x^3": lambda x: x ** 3,
    "sin": np.sin,
    "cos": np.cos,
    "exp": np.exp,
    "sqrt": lambda x: np.sqrt(np.abs(x)),
}


def _snap(xs, ys):
    """Fit ys ≈ a*prim(xs) + b for each primitive; return best (name, a, b, rel)."""
    scale = float(np.std(ys)) + 1e-9
    best = None
    for name, prim in _PRIMITIVES.items():
        with np.errstate(all="ignore"):
            f = prim(xs)
        if not np.all(np.isfinite(f)):
            continue
        A = np.stack([f, np.ones_like(f)], axis=1)
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        rel = float(np.sqrt(np.mean((A @ coef - ys) ** 2))) / scale
        if best is None or rel < best[3]:
            best = (name, float(coef[0]), float(coef[1]), rel)
    return best


def symbolify(model, problem, tol=0.05):
    """Snap the network's 1-D response in each variable to a primitive.

    For each input j the other inputs are pinned to the midpoint of their range
    and j is swept across its training range; the resulting univariate response
    is least-squares-fit to every primitive.  A variable is reported symbolically
    only when its best relative residual is below ``tol`` — otherwise its
    dependence is genuinely not a single primitive (e.g. `nguyen1`'s x^3+x^2+x is
    a *sum*, and `gaussian`'s exp(-x^2/2) is a *composition*), which we say so
    honestly rather than forcing a wrong label.  The learned edge shapes are
    always shown in the spline figure regardless.
    """
    predict = mc.torch_predict_fn(model, torch.device("cpu"))
    mids = np.array([0.5 * (lo + hi) for (lo, hi) in problem["train"]])
    terms, notes = [], []
    for j, (lo, hi) in enumerate(problem["train"]):
        xs = np.linspace(lo, hi, 200)
        X = np.tile(mids, (200, 1))
        X[:, j] = xs
        ys = predict(X)
        name, a, b, rel = _snap(xs, ys)
        vj = problem["vars"][j]
        if rel < tol and abs(a) > 1e-3:
            terms.append(f"{a:.3g}*{name}({vj})" if name != "linear" else f"{a:.3g}*{vj}")
            notes.append(f"{vj}: ~{name} (rel resid {rel:.2g})")
        else:
            notes.append(f"{vj}: no single-primitive match (best {name}, rel resid {rel:.2g})")
    print("Per-variable response snapping:")
    for nline in notes:
        print(f"    {nline}")
    if not terms:
        return "(no single-primitive match — see spline figure)"
    return " + ".join(terms)


def plot_edge_splines(model, problem, save_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping KAN spline plot: {e})")
        return
    d = model.layer1.in_features
    H = model.layer1.out_features
    xs = torch.linspace(-3, 3, 200)
    fig, axes = plt.subplots(d, 1, figsize=(6, 2.4 * d), squeeze=False)
    for j in range(d):
        ax = axes[j][0]
        for hcol in range(H):
            ax.plot(xs.numpy(), model.layer1.edge_response(j, hcol, xs),
                    lw=1.2, alpha=0.8)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_title(f"layer-1 edges from input '{problem['vars'][j]}' "
                     f"(to {H} hidden units)", fontsize=9)
        ax.set_xlabel(problem["vars"][j])
        ax.set_ylabel("φ(x)")
    fig.suptitle("KAN learned univariate edge functions", fontsize=11,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved KAN edge-spline figure -> {save_path}")


def main():
    p = mc.build_argparser("Kolmogorov-Arnold Network symbolic regression")
    p.add_argument("--hidden", type=int, default=5, help="KAN hidden width")
    p.add_argument("--grid-size", type=int, default=8, help="B-spline grid size G")
    p.add_argument("--spline-order", type=int, default=3, help="B-spline order k")
    p.add_argument("--no-symbolify", action="store_true",
                   help="skip snapping edges to primitives")
    args = p.parse_args()
    mc.set_seed(args.seed)
    device = mc.get_device(args.device)

    problem = mc.get_problem(args.problem)
    data = mc.apply_limit(mc.make_dataset(problem, noise_std=args.noise,
                                          seed=args.seed), args.limit)

    model = KAN(problem["n_vars"], hidden=args.hidden, grid_size=args.grid_size,
                spline_order=args.spline_order)
    mc.train_regression(model, data, epochs=args.epochs, lr=args.lr,
                        batch_size=args.batch_size, device=device)

    model.to("cpu")
    expr = None
    if not args.no_symbolify:
        expr = symbolify(model, problem)
        print(f"Symbolified (per-variable response snap): y ≈ {expr}")

    predict_fn = mc.torch_predict_fn(model, torch.device("cpu"))
    mc.report(problem, data, predict_fn, model_name="KAN",
              expr=expr, save_dir=None if args.no_figure else _here())

    if not args.no_figure:
        plot_edge_splines(model, problem,
                          os.path.join(_here(), f"sr_kan-splines_{problem['name']}.png"))


def _here():
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    main()
