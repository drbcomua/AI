"""
10. Histogram-based Gradient Boosting (Ke et al. "LightGBM", 2017; Chen & Guestrin "XGBoost", 2016)
==================================================================================================

The modern, fast flavor of gradient boosting and the practical champion on large
tabular datasets. It is the paradigm behind XGBoost, LightGBM, and CatBoost.
sklearn's `HistGradientBoostingClassifier` is the dependency-free stand-in used
here (the SPEC forbids adding xgboost/lightgbm/catboost).

The single key idea over classic gradient boosting (script 03): **feature
binning**. Each continuous feature is discretized once into ~255 histogram bins.
Split finding then scans O(bins) candidate thresholds per feature instead of
O(unique values), and gradients/hessians are accumulated per bin — turning the
expensive part of tree building into cheap histogram arithmetic.

Architecture Diagram / Layout:
    Once:   bin every feature into <=255 integer bins (quantiles)
    Per boosting round t:
        compute gradients g_i, hessians h_i of the loss at current F_{t-1}
        for each node: build per-bin (sum g, sum h) histograms
                       pick the split maximizing the gain (second-order)
        add the new tree:  F_t = F_{t-1} + lr * tree_t
    Native NaN handling: missing values are sent to whichever child lowers loss.

Key insights / educational takeaways:
    * Binning is why modern GBDTs are fast: histogram building is O(n) and the
      split scan is O(bins), independent of the number of distinct feature values.
    * Same statistical engine as script 03 (gradient boosting), re-engineered for
      speed and scale — this is the algorithm to reach for on real tabular data.
    * `HistGradientBoostingClassifier` exposes no impurity importances, so this
      script reports permutation importance instead (model-agnostic).

Run:
    python "10.hist-gradient-boosting.py" --n-estimators 200 --lr 0.1 --dataset covtype
    python "10.hist-gradient-boosting.py" --limit 2000        # fast smoke test
"""

import os
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
import tabular_common as mc


def main():
    p = mc.build_argparser("Histogram Gradient Boosting Tabular Classifier",
                           max_depth=None, n_estimators=200, lr=0.1)
    args = p.parse_args()
    mc.set_seed(args.seed)

    X_train, X_test, y_train, y_test, feature_names, class_names = \
        mc.load_classification_dataset(args.dataset)
    if args.limit is not None:
        X_train, y_train = X_train[:args.limit], y_train[:args.limit]

    print(f"Training HistGradientBoosting (max_iter={args.n_estimators}, "
          f"lr={args.lr})...")
    model = HistGradientBoostingClassifier(
        max_iter=args.n_estimators, learning_rate=args.lr,
        max_depth=args.max_depth, random_state=args.seed)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    mc.report_classification(y_test, preds, proba, class_names=class_names,
                             model_name="HistGradientBoosting",
                             save_dir=None if args.no_figure else _here())

    if not args.no_figure:
        # No native importances -> permutation importance on the test set.
        print("Computing permutation importances (model-agnostic)...")
        r = permutation_importance(model, X_test, y_test, n_repeats=5,
                                   random_state=args.seed, n_jobs=-1)
        mc.plot_feature_importances(
            np.clip(r.importances_mean, 0, None), feature_names,
            os.path.join(_here(), "histgb_feature_importances.png"),
            "HistGradientBoosting Permutation Importances")


def _here():
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    main()
