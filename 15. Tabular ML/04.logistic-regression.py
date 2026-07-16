"""
04. Logistic Regression (Cox, 1958; regularized form: Hoerl & Kennard 1970 / Tibshirani 1996)
=============================================================================================

The regularized linear baseline every tabular problem should start with. A
logistic regression fits one linear score per class and squashes it through a
softmax; regularization decides how the weight budget is spent.

Architecture Diagram / Layout:
    Input x [D features]  ->  z_c = W_c . x + b_c        (one linear score per class)
                          ->  softmax(z) [C classes]     -> P(class | x)
    Penalty on W:
        L2  (ridge)      : shrinks all weights smoothly toward 0  (dense solution)
        L1  (lasso)      : drives many weights exactly to 0        (sparse solution)
        elasticnet       : a blend of the two (l1_ratio mixes them)

Key insights / educational takeaways:
    * A well-tuned linear model is a shockingly strong baseline; on many tabular
      problems it lands within a few points of far fancier models.
    * L1 yields *sparsity* (feature selection) while L2 yields *shrinkage*
      (all features retained but shrunk) — inspect the coefficient plot to see
      how many weights L1 zeroed out versus L2.
    * Inputs are standardized so the single C penalty applies fairly to every
      feature regardless of its natural units.

Run:
    python "04.logistic-regression.py" --variant l2 --dataset covtype
    python "04.logistic-regression.py" --variant l1
    python "04.logistic-regression.py" --limit 2000        # fast smoke test
"""

import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import tabular_common as mc


def main():
    p = mc.build_argparser("Logistic Regression Tabular Classifier")
    p.add_argument("--variant", choices=["l2", "l1", "elasticnet"], default="l2",
                   help="regularization penalty")
    p.add_argument("--C", type=float, default=1.0, help="inverse regularization strength")
    args = p.parse_args()
    mc.set_seed(args.seed)

    X_train, X_test, y_train, y_test, feature_names, class_names = \
        mc.load_classification_dataset(args.dataset)
    if args.limit is not None:
        X_train, y_train = X_train[:args.limit], y_train[:args.limit]

    # Standardize so the penalty is applied fairly across features.
    X_train, X_test = mc.standardize(X_train, X_test)

    # saga supports every penalty (including L1/elasticnet on multinomial loss).
    kwargs = dict(penalty=args.variant, C=args.C, solver="saga",
                  max_iter=2000, random_state=args.seed, n_jobs=-1)
    if args.variant == "elasticnet":
        kwargs["l1_ratio"] = 0.5

    print(f"Training Logistic Regression (penalty={args.variant}, C={args.C})...")
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    # Report sparsity induced by the penalty.
    coef = model.coef_
    nnz = int(np.count_nonzero(np.abs(coef) > 1e-6))
    print(f"Non-zero coefficients : {nnz}/{coef.size} "
          f"({100 * nnz / coef.size:.1f}% dense)")

    mc.report_classification(y_test, preds, proba, class_names=class_names,
                             model_name=f"LogReg-{args.variant}",
                             save_dir=None if args.no_figure else _here())

    if not args.no_figure:
        # Mean absolute coefficient per feature (aggregated over classes).
        importance = np.abs(coef).mean(axis=0)
        mc.plot_feature_importances(
            importance, feature_names,
            os.path.join(_here(), "lr_coefficients.png"),
            f"Logistic Regression |coef| ({args.variant})")


def _here():
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    main()
