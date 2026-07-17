"""
sr_common.py
============

Shared utilities for the Symbolic Regression demos in this folder.

Symbolic regression asks a harder question than ordinary regression: not merely
"fit y from X", but "recover the *closed-form expression* that generated the
data". Every script here is scored on the same three numbers so the paradigms
compare directly:

    1. Interpolation RMSE   — held-out samples inside the training range.
    2. Extrapolation RMSE   — held-out samples OUTSIDE the training range
                              (where black-box fits fall apart and structural
                              methods keep working).
    3. Recovery             — did the method emit a human-readable expression,
                              and does it match ground truth?  Proxy per SPEC:
                              RMSE < 1e-6 on a freshly-sampled wide domain.

Everything each architecture script needs lives here so the scripts stay focused
on the *method*:

    Problems / data
        * PROBLEMS               -> registry of ~8 ground-truth formulas
        * get_problem(name)      -> one problem's metadata
        * make_dataset(...)      -> train/val/test + extrapolation splits

    Command line
        * build_argparser(...)   -> the shared CLI (--problem/--noise/--seed/...)
        * get_device(...) / set_seed(...)

    Neural training (Phase 2 scripts 03-05)
        * to_tensors(...)        -> float32 tensors on device
        * train_regression(...)  -> MSE loop printing per-epoch loss + time
        * torch_predict_fn(...)  -> wrap a torch model as a numpy predict_fn
        * count_parameters(model)

    Reporting / recovery
        * rmse / r2
        * check_recovery(predict_fn, problem) -> (recovered_bool, wide_rmse)
        * report(problem, data, predict_fn, ...) -> the standard 3-number report
                                                    + predicted-vs-true figure

A ``predict_fn`` is the common currency between the classical (numpy) and neural
(torch) scripts: any callable ``X[n, d] -> y[n]``.  Because every method exposes
one, ``report`` and ``check_recovery`` work identically for GP, SINDy, EQL, the
MLP, and the KAN.
"""

from __future__ import annotations

import os
import time
import argparse

import numpy as np

# --------------------------------------------------------------------------- #
# Ground-truth problem registry
# --------------------------------------------------------------------------- #
# Each problem is a dict:
#   name    : CLI identifier
#   expr    : ground-truth expression (as a readable string)
#   n_vars  : number of input variables
#   fn      : callable X[n, n_vars] -> y[n]  (the ground truth)
#   train   : list of (lo, hi) sampling ranges per variable (interpolation)
#   extrap  : list of (lo, hi) ranges per variable, OUTSIDE `train`
#
# `train` and `extrap` are disjoint per variable so the extrapolation split is
# genuinely outside the fitted region.  The variable names in `vars` are only
# used for pretty-printing recovered expressions.

def _c(*ranges):
    return [tuple(r) for r in ranges]


PROBLEMS = {
    # ---- Nguyen-style univariate polynomials / composites -------------------
    "nguyen1": dict(
        expr="x^3 + x^2 + x", n_vars=1, vars=["x"],
        fn=lambda X: X[:, 0] ** 3 + X[:, 0] ** 2 + X[:, 0],
        train=_c((-1.0, 1.0)), extrap=_c((1.0, 3.0)),
    ),
    "nguyen5": dict(
        expr="sin(x^2) * cos(x) - 1", n_vars=1, vars=["x"],
        fn=lambda X: np.sin(X[:, 0] ** 2) * np.cos(X[:, 0]) - 1.0,
        train=_c((-1.5, 1.5)), extrap=_c((1.5, 3.0)),
    ),
    "nguyen6": dict(
        expr="sin(x) + sin(x + x^2)", n_vars=1, vars=["x"],
        fn=lambda X: np.sin(X[:, 0]) + np.sin(X[:, 0] + X[:, 0] ** 2),
        train=_c((-1.5, 1.5)), extrap=_c((1.5, 3.0)),
    ),
    "gaussian": dict(
        expr="exp(-x^2 / 2)", n_vars=1, vars=["x"],
        fn=lambda X: np.exp(-(X[:, 0] ** 2) / 2.0),
        train=_c((-2.0, 2.0)), extrap=_c((2.0, 4.0)),
    ),
    # ---- Feynman-style physics (2-3 variables) ------------------------------
    "kinetic": dict(  # kinetic energy  E = 1/2 m v^2
        expr="0.5 * m * v^2", n_vars=2, vars=["m", "v"],
        fn=lambda X: 0.5 * X[:, 0] * X[:, 1] ** 2,
        train=_c((1.0, 3.0), (1.0, 3.0)), extrap=_c((3.0, 5.0), (3.0, 5.0)),
    ),
    "distance": dict(  # Euclidean norm  r = sqrt(x^2 + y^2)
        expr="sqrt(x^2 + y^2)", n_vars=2, vars=["x", "y"],
        fn=lambda X: np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2),
        train=_c((1.0, 3.0), (1.0, 3.0)), extrap=_c((3.0, 5.0), (3.0, 5.0)),
    ),
    "coulomb": dict(  # Coulomb force (constants folded to 1): F = q1 q2 / r^2
        expr="q1 * q2 / r^2", n_vars=3, vars=["q1", "q2", "r"],
        fn=lambda X: X[:, 0] * X[:, 1] / (X[:, 2] ** 2),
        train=_c((1.0, 2.0), (1.0, 2.0), (1.0, 2.0)),
        extrap=_c((2.0, 3.0), (2.0, 3.0), (2.0, 3.0)),
    ),
    "rational": dict(  # a rational form  x y / (1 + x^2)
        expr="x * y / (1 + x^2)", n_vars=2, vars=["x", "y"],
        fn=lambda X: X[:, 0] * X[:, 1] / (1.0 + X[:, 0] ** 2),
        train=_c((-2.0, 2.0), (-2.0, 2.0)), extrap=_c((2.0, 4.0), (2.0, 4.0)),
    ),
}


def get_problem(name: str) -> dict:
    if name not in PROBLEMS:
        raise ValueError(f"unknown --problem {name!r}; choose from {list(PROBLEMS)}")
    p = dict(PROBLEMS[name])
    p["name"] = name
    return p


# --------------------------------------------------------------------------- #
# Data synthesis
# --------------------------------------------------------------------------- #
class Dataset:
    """Container for the four splits plus problem metadata."""

    def __init__(self, problem, X_train, y_train, X_val, y_val,
                 X_test, y_test, X_extrap, y_extrap, noise):
        self.problem = problem
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        self.X_extrap, self.y_extrap = X_extrap, y_extrap
        self.noise = noise
        self.n_vars = problem["n_vars"]


def _sample(domains, n, rng):
    cols = [rng.uniform(lo, hi, size=n) for (lo, hi) in domains]
    return np.stack(cols, axis=1).astype(np.float64)


def make_dataset(problem, n_train=600, n_eval=200, noise_std=0.0, seed=42):
    """Sample train / val / test (interpolation) + extrapolation splits.

    Noise is Gaussian with standard deviation ``noise_std * std(y_train)`` and is
    added to the **train and val targets only** — test and extrapolation targets
    stay clean so fit quality and recovery are measured against the truth.
    """
    rng = np.random.default_rng(seed)
    fn = problem["fn"]

    X_train = _sample(problem["train"], n_train, rng)
    X_val = _sample(problem["train"], n_eval, rng)
    X_test = _sample(problem["train"], n_eval, rng)
    X_extrap = _sample(problem["extrap"], n_eval, rng)

    y_train = fn(X_train)
    y_val = fn(X_val)
    y_test = fn(X_test)
    y_extrap = fn(X_extrap)

    if noise_std > 0:
        sigma = noise_std * float(np.std(y_train) + 1e-12)
        y_train = y_train + rng.normal(0, sigma, size=y_train.shape)
        y_val = y_val + rng.normal(0, sigma, size=y_val.shape)

    return Dataset(problem, X_train, y_train, X_val, y_val,
                   X_test, y_test, X_extrap, y_extrap, noise_std)


# --------------------------------------------------------------------------- #
# Command-line interface
# --------------------------------------------------------------------------- #
def build_argparser(description: str, *, lr: float = 1e-2, epochs: int = 200,
                    batch_size: int = 128):
    """Shared CLI for every symbolic-regression script.

    Classical scripts (01-02) ignore the neural flags; neural scripts (03-05)
    use all of them.  ``--limit`` caps the number of training samples for a fast
    smoke test (GP additionally shrinks its population x generations budget).
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--problem", type=str, default="nguyen1", choices=list(PROBLEMS),
                   help="ground-truth formula to recover")
    p.add_argument("--noise", type=float, default=0.0,
                   help="Gaussian noise on train targets, in units of std(y)")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    # Neural-training flags (Phase 2).
    p.add_argument("--epochs", type=int, default=epochs, help="neural training epochs")
    p.add_argument("--batch-size", type=int, default=batch_size, help="neural batch size")
    p.add_argument("--lr", type=float, default=lr, help="neural learning rate")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    # Universal.
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N training samples (quick smoke test)")
    p.add_argument("--no-figure", action="store_true", help="do not save figures")
    return p


def get_device(prefer: str = "auto"):
    import torch
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def apply_limit(data: Dataset, limit):
    """Shrink the training split to the first ``limit`` samples (smoke test)."""
    if limit is None:
        return data
    data.X_train = data.X_train[:limit]
    data.y_train = data.y_train[:limit]
    return data


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    ss_res = float(np.sum((y_pred - y_true) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# --------------------------------------------------------------------------- #
# Recovery check (numeric symbolic-equivalence proxy)
# --------------------------------------------------------------------------- #
def check_recovery(predict_fn, problem, tol=1e-6, seed=12345):
    """Numeric proxy for symbolic equivalence.

    Samples a fresh **wide** domain (the union of the train and extrapolation
    ranges, widened another 20%) and returns ``(recovered, wide_rmse)`` where
    ``recovered = wide_rmse < tol``.  A method that recovered the true structure
    with the right constants matches everywhere; one that merely fit the training
    region does not.
    """
    rng = np.random.default_rng(seed)
    wide = []
    for (t_lo, t_hi), (e_lo, e_hi) in zip(problem["train"], problem["extrap"]):
        lo = min(t_lo, e_lo)
        hi = max(t_hi, e_hi)
        span = hi - lo
        wide.append((lo - 0.2 * span, hi + 0.2 * span))
    X = _sample(wide, 500, rng)
    y_true = problem["fn"](X)
    try:
        y_pred = np.asarray(predict_fn(X), np.float64).reshape(-1)
    except Exception as e:  # a malformed expression should not crash the report
        print(f"(recovery check: predict_fn raised {e})")
        return False, float("inf")
    if not np.all(np.isfinite(y_pred)):
        return False, float("inf")
    wr = rmse(y_true, y_pred)
    return bool(wr < tol), wr


# --------------------------------------------------------------------------- #
# The standard report
# --------------------------------------------------------------------------- #
def report(problem, data: Dataset, predict_fn, *, model_name="model",
           expr=None, save_dir=None):
    """Print the folder-standard three numbers and save the standard figure.

    Returns a dict of metrics for cross-script tabulation.  ``expr`` is the
    recovered expression string (or ``None`` for black-box methods, which score
    "not recoverable").
    """
    y_test_pred = np.asarray(predict_fn(data.X_test), np.float64).reshape(-1)
    y_ext_pred = np.asarray(predict_fn(data.X_extrap), np.float64).reshape(-1)

    interp_rmse = rmse(data.y_test, y_test_pred)
    interp_r2 = r2(data.y_test, y_test_pred)
    extrap_rmse = rmse(data.y_extrap, y_ext_pred)

    recovered, wide_rmse = check_recovery(predict_fn, problem)

    name = problem["name"]
    print(f"\n============  SYMBOLIC-REGRESSION REPORT: {model_name}  ============")
    print(f"Problem               : {name}   ( y = {problem['expr']} )")
    print(f"Noise on train        : {data.noise:.3g} * std(y)")
    print(f"Interpolation RMSE    : {interp_rmse:.6g}")
    print(f"Interpolation R^2     : {interp_r2:.6f}")
    print(f"Extrapolation RMSE    : {extrap_rmse:.6g}")
    if expr is not None:
        print(f"Recovered expression  : {expr}")
        print(f"Recovery (wide RMSE)  : {wide_rmse:.3g}  -> "
              f"{'RECOVERED' if recovered else 'not recovered'} (tol 1e-6)")
    else:
        print("Recovered expression  : (black-box — not recoverable)")
    print("=" * 66 + "\n")

    if save_dir is not None:
        _save_figure(problem, data, predict_fn, model_name, expr, save_dir)

    return dict(model=model_name, problem=name, interp_rmse=interp_rmse,
                interp_r2=interp_r2, extrap_rmse=extrap_rmse,
                recovered=recovered, wide_rmse=wide_rmse, expr=expr)


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name).strip("-").lower()


def _save_figure(problem, data, predict_fn, model_name, expr, save_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return

    out = os.path.join(save_dir, f"sr_{_slug(model_name)}_{problem['name']}.png")

    if problem["n_vars"] == 1:
        _save_curve_figure(problem, data, predict_fn, model_name, expr, out, plt)
    else:
        _save_scatter_figure(problem, data, predict_fn, model_name, expr, out, plt)


def _save_curve_figure(problem, data, predict_fn, model_name, expr, out, plt):
    """1-D problems: prediction vs. ground truth over train AND extrap range."""
    (t_lo, t_hi) = problem["train"][0]
    (e_lo, e_hi) = problem["extrap"][0]
    lo, hi = min(t_lo, e_lo), max(t_hi, e_hi)
    xs = np.linspace(lo, hi, 400).reshape(-1, 1)
    y_true = problem["fn"](xs)
    y_pred = np.asarray(predict_fn(xs), np.float64).reshape(-1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvspan(t_lo, t_hi, color="0.9", label="training range")
    ax.plot(xs[:, 0], y_true, color="black", lw=2, label=f"truth: {problem['expr']}")
    ax.plot(xs[:, 0], y_pred, color="crimson", lw=1.8, ls="--", label="prediction")
    ax.scatter(data.X_train[:, 0], data.y_train, s=10, alpha=0.35,
               color="steelblue", label="train samples")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = f"{model_name} on {problem['name']} — interpolate (grey) vs. extrapolate"
    if expr:
        title += f"\nrecovered: {expr}"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="best")
    _finish(fig, out, plt)


def _save_scatter_figure(problem, data, predict_fn, model_name, expr, out, plt):
    """Multivariate problems: predicted-vs-true scatter for test and extrap."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, X, y, tag in [(axes[0], data.X_test, data.y_test, "interpolation"),
                          (axes[1], data.X_extrap, data.y_extrap, "extrapolation")]:
        y_pred = np.asarray(predict_fn(X), np.float64).reshape(-1)
        ax.scatter(y, y_pred, s=14, alpha=0.5,
                   color="steelblue" if tag == "interpolation" else "crimson",
                   edgecolors="none")
        lo = float(min(y.min(), y_pred.min()))
        hi = float(max(y.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="perfect")
        ax.set_xlabel("true y")
        ax.set_ylabel("predicted y")
        ax.set_title(f"{tag}  (RMSE {rmse(y, y_pred):.3g})", fontsize=10)
        ax.legend(fontsize=8)
    sup = f"{model_name} on {problem['name']}  ( y = {problem['expr']} )"
    if expr:
        sup += f"   recovered: {expr}"
    fig.suptitle(sup, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved figure -> {out}")


def _finish(fig, out, plt):
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved figure -> {out}")


# --------------------------------------------------------------------------- #
# Neural training helpers (Phase 2: scripts 03-05)
# --------------------------------------------------------------------------- #
def to_tensors(X, y, device):
    import torch
    Xt = torch.tensor(np.asarray(X, np.float32), device=device)
    yt = torch.tensor(np.asarray(y, np.float32), device=device).reshape(-1, 1)
    return Xt, yt


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_regression(model, data: Dataset, *, epochs=200, lr=1e-2, batch_size=128,
                     device=None, weight_decay=0.0, extra_loss=None,
                     log_every=None):
    """Full-batch/mini-batch MSE training loop shared by scripts 03-05.

    ``extra_loss(model) -> scalar`` lets a script add a regularizer (EQL's L1
    sparsity).  Prints per-epoch train/val loss and wall-clock time and reports
    the exact trainable-parameter count (the folder's methodological point).
    """
    import torch

    device = device or get_device()
    model.to(device)
    Xtr, ytr = to_tensors(data.X_train, data.y_train, device)
    Xva, yva = to_tensors(data.X_val, data.y_val, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = torch.nn.MSELoss()
    n = Xtr.shape[0]
    log_every = log_every or max(1, epochs // 10)

    print(f"Device: {device} | trainable params: {count_parameters(model):,}")
    print("-" * 60)
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n, device=device)
        running = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            out = model(Xtr[idx])
            loss = mse(out, ytr[idx])
            if extra_loss is not None:
                loss = loss + extra_loss(model)
            loss.backward()
            opt.step()
            running += loss.item() * idx.numel()
        dt = time.time() - t0
        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val = mse(model(Xva), yva).item()
            print(f"Epoch {epoch:4d}/{epochs} | train_loss {running / n:.6f} | "
                  f"val_loss {val:.6f} | {dt * 1e3:.1f} ms/epoch")
    print("-" * 60)
    return model


def torch_predict_fn(model, device=None):
    """Wrap a trained torch model as a numpy ``predict_fn`` for report/recovery."""
    import torch
    device = device or get_device()
    model.to(device)
    model.eval()

    def predict(X):
        Xt = torch.tensor(np.asarray(X, np.float32), device=device)
        with torch.no_grad():
            out = model(Xt)
        return out.cpu().numpy().reshape(-1)

    return predict
