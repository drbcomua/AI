"""
02. Sparse Regression — SINDy-style STLSQ (Brunton, Proctor & Kutz, 2016)
=========================================================================

Symbolic regression reframed as **sparse linear algebra**.  Instead of searching
expression space (script 01), fix a large *library* of candidate terms up front,
then find the few whose linear combination reproduces the data:

        y  ≈  Θ(X) · ξ ,     ξ sparse

    Θ(X)   feature library, one column per candidate term:
           1,  x0, x1, ...,  x0^2, x0*x1, ..., x0^3, ...,   (polynomials ≤ 3)
           sin(x0), cos(x0), exp(x0), ...                    (unary transforms)
    ξ      coefficient vector — most entries driven to exactly zero.

Sequentially-Thresholded Least Squares (STLSQ):

        ξ = lstsq(Θ, y)                     # ordinary least squares
        repeat:
            ξ[|ξ| < threshold] = 0          # kill small coefficients
            ξ[active] = lstsq(Θ[active], y) # refit on survivors only

Each pass removes negligible terms and refits, converging to a sparse, readable
expression.  Reading it off is trivial: print the surviving columns with their
coefficients.

Educational takeaway:
    SINDy is *instant and near-exact* — but only when the true terms live in the
    library.  On `nguyen1` (x^3 + x^2 + x, all polynomial) it recovers the
    formula to machine precision in milliseconds.  On `nguyen5`
    (sin(x^2)·cos(x) − 1, whose `x^2`-inside-`sin` term is NOT in a library of
    `sin(x)`/`cos(x)`) it cannot represent the truth at all and returns a dense,
    wrong fit.  That library dependence is the whole story: a brilliant method
    inside its basis, blind outside it.  Compare with GP (01), which builds the
    `sin(x^2)` composition itself, and EQL/KAN (03/05), which learn transforms.

Run:
    python "02.sparse-regression.py" --problem nguyen1     # in-library -> exact
    python "02.sparse-regression.py" --problem nguyen5     # out-of-library -> fails
    python "02.sparse-regression.py" --problem kinetic
"""

import os
import itertools
import numpy as np
import sr_common as mc


def build_library(X, var_names, poly_degree=3):
    """Return (Theta, term_names): polynomials up to `poly_degree` plus sin/cos/
    exp of each raw variable.

    Polynomial terms include cross-products (x0*x1, x0^2*x1, ...).  A leading
    constant column captures any bias.
    """
    n, d = X.shape
    cols = [np.ones(n)]
    names = ["1"]

    # Polynomial terms up to total degree `poly_degree` (excluding the constant).
    for deg in range(1, poly_degree + 1):
        for combo in itertools.combinations_with_replacement(range(d), deg):
            col = np.ones(n)
            for j in combo:
                col = col * X[:, j]
            cols.append(col)
            # pretty name, e.g. (0,0,1) -> x0^2*x1
            counts = {j: combo.count(j) for j in set(combo)}
            parts = [var_names[j] + (f"^{c}" if c > 1 else "") for j, c in
                     sorted(counts.items())]
            names.append("*".join(parts))

    # Unary transforms of each raw variable.
    for j in range(d):
        for fname, fn in [("sin", np.sin), ("cos", np.cos), ("exp", np.exp)]:
            with np.errstate(over="ignore"):
                col = fn(X[:, j])
            cols.append(col)
            names.append(f"{fname}({var_names[j]})")

    Theta = np.stack(cols, axis=1)
    return Theta, names


def stlsq(Theta, y, threshold=0.05, n_iter=20):
    """Sequentially-thresholded least squares -> sparse coefficient vector."""
    xi, *_ = np.linalg.lstsq(Theta, y, rcond=None)
    for _ in range(n_iter):
        small = np.abs(xi) < threshold
        xi[small] = 0.0
        active = ~small
        if not active.any():
            break
        coef, *_ = np.linalg.lstsq(Theta[:, active], y, rcond=None)
        xi_new = np.zeros_like(xi)
        xi_new[active] = coef
        if np.allclose(xi_new, xi):
            xi = xi_new
            break
        xi = xi_new
    return xi


def coeffs_to_expr(xi, names):
    terms = []
    for c, name in zip(xi, names):
        if abs(c) < 1e-8:
            continue
        if name == "1":
            terms.append(f"{c:.4g}")
        else:
            terms.append(f"{c:.4g}*{name}")
    return " + ".join(terms) if terms else "0"


def main():
    p = mc.build_argparser("SINDy sparse-regression symbolic regression")
    p.add_argument("--poly-degree", type=int, default=3, help="max polynomial degree")
    p.add_argument("--threshold", type=float, default=0.05,
                   help="STLSQ sparsity threshold")
    args = p.parse_args()
    mc.set_seed(args.seed)

    problem = mc.get_problem(args.problem)
    data = mc.apply_limit(mc.make_dataset(problem, noise_std=args.noise,
                                          seed=args.seed), args.limit)
    var_names = problem["vars"]

    Theta, names = build_library(data.X_train, var_names, args.poly_degree)
    print(f"Library: {len(names)} candidate terms (poly<= {args.poly_degree} + sin/cos/exp)")
    xi = stlsq(Theta, data.y_train, threshold=args.threshold)
    expr = coeffs_to_expr(xi, names)
    n_active = int(np.sum(np.abs(xi) > 1e-8))
    print(f"Recovered {n_active} active term(s): y = {expr}")

    def predict_fn(X):
        X = np.asarray(X, np.float64)
        Th, _ = build_library(X, var_names, args.poly_degree)
        return Th @ xi

    mc.report(problem, data, predict_fn, model_name="SINDy (sparse regression)",
              expr=expr, save_dir=None if args.no_figure else _here())


def _here():
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    main()
