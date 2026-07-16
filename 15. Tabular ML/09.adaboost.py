"""
09. AdaBoost (Freund & Schapire, 1997)
======================================

The original practical boosting algorithm. Like gradient boosting (script 03)
it builds an additive ensemble of weak learners sequentially, but the *mechanism*
is different and worth contrasting directly:

    * Gradient Boosting (script 03): each new tree is fit to the **negative
      gradient (residuals)** of the loss of the current ensemble.
    * AdaBoost (this script): each new stump is fit to a **reweighted dataset** —
      samples the current ensemble gets wrong have their weights increased, so
      the next weak learner focuses on them. Each stump then gets a vote weight
      proportional to its accuracy.

Architecture Diagram / Layout:
    w_i = 1/N                                       (init sample weights)
    repeat T times:
        fit stump h_t on data weighted by w         (weak learner: depth-1 tree)
        err_t = weighted error of h_t
        alpha_t = log((1 - err_t)/err_t)            (this stump's vote weight)
        w_i *= exp(alpha_t * [h_t(x_i) wrong])      (up-weight the mistakes)
        renormalize w
    H(x) = argmax_c  sum_t alpha_t [h_t(x) = c]     (SAMME weighted vote)

Key insights / educational takeaways:
    * Boosting-by-reweighting vs. boosting-by-residuals — same additive-ensemble
      family, two different derivations that arrive at similar places.
    * Classically run on depth-1 "decision stumps"; increasing stump depth turns
      the knob back toward gradient boosting.
    * Sensitive to label noise: outliers keep getting up-weighted, so AdaBoost
      can chase noisy points harder than gradient boosting does.

Run:
    python "09.adaboost.py" --n-estimators 200 --lr 0.5 --dataset covtype
    python "09.adaboost.py" --limit 2000        # fast smoke test
"""

import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import tabular_common as mc


def main():
    p = mc.build_argparser("AdaBoost Tabular Classifier", max_depth=1,
                           n_estimators=200, lr=0.5)
    args = p.parse_args()
    mc.set_seed(args.seed)

    X_train, X_test, y_train, y_test, feature_names, class_names = \
        mc.load_classification_dataset(args.dataset)
    if args.limit is not None:
        X_train, y_train = X_train[:args.limit], y_train[:args.limit]

    print(f"Training AdaBoost (n_estimators={args.n_estimators}, lr={args.lr}, "
          f"stump depth={args.max_depth})...")
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=args.max_depth),
        n_estimators=args.n_estimators, learning_rate=args.lr,
        random_state=args.seed)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    mc.report_classification(y_test, preds, proba, class_names=class_names,
                             model_name="AdaBoost",
                             save_dir=None if args.no_figure else _here())

    if not args.no_figure:
        mc.plot_feature_importances(
            model.feature_importances_, feature_names,
            os.path.join(_here(), "ada_feature_importances.png"),
            "AdaBoost Feature Importances")


def _here():
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    main()
