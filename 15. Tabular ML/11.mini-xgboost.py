"""
11. Mini-XGBoost — second-order gradient boosting from scratch (Chen & Guestrin, 2016)
======================================================================================

The one deliberately non-sklearn script in this folder: a compact, readable
implementation of XGBoost's core math in pure NumPy. Script 03 (sklearn
GradientBoosting) uses only first-order gradients; XGBoost's contribution is a
**second-order (Newton) view** of boosting — it Taylor-expands the loss to
second order, so each tree is fit using both the gradient g_i and the Hessian
h_i, and leaf values / split gains carry a Hessian denominator plus an L2
(lambda) regularizer.

The math implemented here (per boosting round, per class in a one-vs-rest
softmax):

    Taylor expand loss around current score F:
        L ~ sum_i [ g_i * w + 0.5 * h_i * w^2 ] + 0.5 * lambda * w^2
    Optimal leaf weight (close the square):
        w* = - (sum_i g_i) / (sum_i h_i + lambda)
    Gain of splitting a node into L, R (higher = better):
        gain = 0.5 * [ G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - G^2/(H+lam) ] - gamma
    Softmax gradients/Hessians (multiclass log-loss):
        p = softmax(F);  g = p_k - y_onehot_k;   h = p_k * (1 - p_k)
    Update:  F[:,k] += learning_rate * tree_k(x)

Architecture Diagram / Layout:
    F_0 = 0  (raw scores, shape [N, K])
    for round t in 1..T:
        p = softmax(F)
        for class k:
            g_k = p_k - y_k ;  h_k = p_k (1 - p_k)
            tree = grow depth<=D CART on (g_k, h_k) using gain above,
                   leaves = -G/(H+lambda)
            F[:,k] += lr * tree.predict(X)
    predict: argmax_k softmax(F)(x)

Key insights / educational takeaways:
    * Newton boosting: the Hessian rescales each leaf, so confident-but-flat
      regions (small h) move more per step — better-conditioned than plain
      gradient boosting.
    * lambda (L2 on leaf weights) and gamma (min split gain) are the two
      regularizers that made XGBoost robust; try raising them to see splits
      pruned away.
    * Exact greedy splits (this script) are what histogram binning (script 10)
      approximates for speed. Same objective, different split-finding.

This is a demonstration of the *mechanism*, kept small (exact greedy splits,
depth <= 3): it should comfortably beat the single tree (script 01) and
approach sklearn's gradient boosting (script 03) on Wine.

Run:
    python "11.mini-xgboost.py" --n-estimators 50 --lr 0.3 --max-depth 3
    python "11.mini-xgboost.py" --limit 2000        # fast smoke test
"""

import os
import numpy as np
import tabular_common as mc


# --------------------------------------------------------------------------- #
# A single regularized regression tree fit to (gradient, hessian) targets.
# --------------------------------------------------------------------------- #
class _GBTree:
    def __init__(self, max_depth=3, lam=1.0, gamma=0.0, min_child_hessian=1e-3):
        self.max_depth = max_depth
        self.lam = lam
        self.gamma = gamma
        self.min_child_hessian = min_child_hessian
        self.root = None
        self.feature_gain = None  # accumulated split gain per feature

    def fit(self, X, g, h, n_features):
        self.feature_gain = np.zeros(n_features)
        self.root = self._build(X, g, h, depth=0)
        return self

    def _leaf_weight(self, G, H):
        return -G / (H + self.lam)

    def _best_split(self, X, g, h):
        G, H = g.sum(), h.sum()
        base = G * G / (H + self.lam)
        best_gain, best = 0.0, None
        for j in range(X.shape[1]):
            xj = X[:, j]
            order = np.argsort(xj, kind="mergesort")
            xs, gs, hs = xj[order], g[order], h[order]
            Gl = np.cumsum(gs)[:-1]
            Hl = np.cumsum(hs)[:-1]
            Gr, Hr = G - Gl, H - Hl
            # Prune splits whose child Hessian mass is too small (unstable leaves).
            valid = (xs[:-1] != xs[1:]) & (Hl >= self.min_child_hessian) & \
                    (Hr >= self.min_child_hessian)
            gain = 0.5 * (Gl * Gl / (Hl + self.lam) +
                          Gr * Gr / (Hr + self.lam) - base) - self.gamma
            gain = np.where(valid, gain, -np.inf)
            i = int(np.argmax(gain))
            if gain[i] > best_gain:
                best_gain = float(gain[i])
                best = (j, 0.5 * (xs[i] + xs[i + 1]))
        return best_gain, best

    def _build(self, X, g, h, depth):
        G, H = g.sum(), h.sum()
        if depth >= self.max_depth or len(g) < 2:
            return {"leaf": self._leaf_weight(G, H)}
        gain, split = self._best_split(X, g, h)
        if split is None:
            return {"leaf": self._leaf_weight(G, H)}
        j, thr = split
        self.feature_gain[j] += gain
        mask = X[:, j] <= thr
        if mask.all() or (~mask).all():
            return {"leaf": self._leaf_weight(G, H)}
        return {
            "feat": j, "thr": thr,
            "left": self._build(X[mask], g[mask], h[mask], depth + 1),
            "right": self._build(X[~mask], g[~mask], h[~mask], depth + 1),
        }

    def predict(self, X):
        out = np.empty(len(X), dtype=np.float64)
        for i, x in enumerate(X):
            node = self.root
            while "leaf" not in node:
                node = node["left"] if x[node["feat"]] <= node["thr"] else node["right"]
            out[i] = node["leaf"]
        return out


# --------------------------------------------------------------------------- #
# Multiclass softmax boosting wrapper.
# --------------------------------------------------------------------------- #
class MiniXGBoost:
    def __init__(self, n_estimators=50, lr=0.3, max_depth=3, lam=1.0, gamma=0.0):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.lam = lam
        self.gamma = gamma
        self.trees = []            # list of per-round [tree_class0, ...]
        self.n_classes = None
        self.feature_importance_ = None

    @staticmethod
    def _softmax(F):
        F = F - F.max(axis=1, keepdims=True)
        e = np.exp(F)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X, y, verbose=True):
        X = np.asarray(X, np.float64)
        n, d = X.shape
        self.n_classes = int(y.max()) + 1
        Y = np.eye(self.n_classes)[y]           # one-hot
        F = np.zeros((n, self.n_classes))
        self.feature_importance_ = np.zeros(d)

        for t in range(self.n_estimators):
            P = self._softmax(F)
            round_trees = []
            for k in range(self.n_classes):
                g = P[:, k] - Y[:, k]           # gradient of softmax log-loss
                h = P[:, k] * (1.0 - P[:, k])   # diagonal Hessian
                tree = _GBTree(self.max_depth, self.lam, self.gamma).fit(X, g, h, d)
                F[:, k] += self.lr * tree.predict(X)
                self.feature_importance_ += tree.feature_gain
                round_trees.append(tree)
            self.trees.append(round_trees)
            if verbose and (t + 1) % max(1, self.n_estimators // 5) == 0:
                loss = -np.mean(np.log((self._softmax(F) * Y).sum(1) + 1e-12))
                acc = (self._softmax(F).argmax(1) == y).mean()
                print(f"  round {t + 1:3d}/{self.n_estimators} | "
                      f"train logloss {loss:.4f} | train acc {acc:.4f}")
        s = self.feature_importance_.sum()
        if s > 0:
            self.feature_importance_ /= s
        return self

    def decision_scores(self, X):
        X = np.asarray(X, np.float64)
        F = np.zeros((len(X), self.n_classes))
        for round_trees in self.trees:
            for k, tree in enumerate(round_trees):
                F[:, k] += self.lr * tree.predict(X)
        return F

    def predict_proba(self, X):
        return self._softmax(self.decision_scores(X))

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def main():
    p = mc.build_argparser("Mini-XGBoost (from-scratch second-order boosting)",
                           max_depth=3, n_estimators=50, lr=0.3)
    p.add_argument("--lam", type=float, default=1.0, help="L2 leaf regularization")
    p.add_argument("--gamma", type=float, default=0.0, help="minimum split gain")
    args = p.parse_args()
    mc.set_seed(args.seed)

    X_train, X_test, y_train, y_test, feature_names, class_names = \
        mc.load_classification_dataset(args.dataset)
    if args.limit is not None:
        X_train, y_train = X_train[:args.limit], y_train[:args.limit]

    print(f"Training Mini-XGBoost (rounds={args.n_estimators}, lr={args.lr}, "
          f"depth={args.max_depth}, lambda={args.lam}, gamma={args.gamma})...")
    model = MiniXGBoost(n_estimators=args.n_estimators, lr=args.lr,
                        max_depth=args.max_depth, lam=args.lam, gamma=args.gamma)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    mc.report_classification(y_test, preds, proba, class_names=class_names,
                             model_name="Mini-XGBoost",
                             save_dir=None if args.no_figure else _here())

    if not args.no_figure:
        mc.plot_feature_importances(
            model.feature_importance_, feature_names,
            os.path.join(_here(), "minixgb_feature_importances.png"),
            "Mini-XGBoost Split-Gain Importances")


def _here():
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    main()
