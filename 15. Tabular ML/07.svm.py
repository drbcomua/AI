"""
07. Support Vector Machine (Cortes & Vapnik, 1995)
==================================================

The max-margin classifier that was the pre-GBDT state of the art on tabular
data. An SVM finds the separating hyperplane with the *largest margin* to the
nearest points (the support vectors); the kernel trick lets it draw non-linear
boundaries by implicitly mapping features into a higher-dimensional space
without ever computing the coordinates there.

Architecture Diagram / Layout:
    linear :  decision(x) = sign( w . x + b )          margin maximized in input space
    rbf    :  decision(x) = sign( sum_i a_i y_i K(x_i, x) + b )
              K(x_i, x) = exp(-gamma ||x_i - x||^2)     implicit infinite-dim feature map
    Only the support vectors (points on/inside the margin) have a_i != 0.

Key insights / educational takeaways:
    * Max-margin objective => good generalization from few parameters; the model
      is defined entirely by its support vectors.
    * The kernel trick (rbf) buys non-linear boundaries at the cost of an
      n x n kernel matrix — training is roughly O(n^2), so SVMs do not scale to
      large n. This is exactly why GBDTs displaced them on big tabular data.
    * Features must be standardized: the RBF kernel measures Euclidean distance,
      so unscaled features distort the geometry (same lesson as k-NN).

Run:
    python "07.svm.py" --variant rbf --dataset wine
    python "07.svm.py" --variant linear --limit 2000
    # NOTE: rbf on the full covtype subsample is slow (O(n^2)); use --limit there.
"""

import os
from sklearn.svm import SVC
import tabular_common as mc


def main():
    p = mc.build_argparser("Support Vector Machine Tabular Classifier")
    p.add_argument("--variant", choices=["linear", "rbf"], default="rbf",
                   help="kernel type")
    p.add_argument("--C", type=float, default=1.0, help="soft-margin penalty")
    args = p.parse_args()
    mc.set_seed(args.seed)

    X_train, X_test, y_train, y_test, feature_names, class_names = \
        mc.load_classification_dataset(args.dataset)
    if args.limit is not None:
        X_train, y_train = X_train[:args.limit], y_train[:args.limit]

    # RBF/linear SVMs both depend on Euclidean geometry -> standardize.
    X_train, X_test = mc.standardize(X_train, X_test)

    print(f"Training SVM (kernel={args.variant}, C={args.C})...")
    model = SVC(kernel=args.variant, C=args.C, gamma="scale", random_state=args.seed)
    model.fit(X_train, y_train)
    print(f"Support vectors        : {int(model.support_vectors_.shape[0])}"
          f"/{len(X_train)} training points")

    preds = model.predict(X_test)
    # probability=True triggers an expensive internal CV; skip it and report
    # without log-loss/probabilities.
    mc.report_classification(y_test, preds, None, class_names=class_names,
                             model_name=f"SVM-{args.variant}",
                             save_dir=None if args.no_figure else
                             os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
