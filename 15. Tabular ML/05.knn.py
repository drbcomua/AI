"""
05. k-Nearest Neighbors (Cover & Hart, 1967)
============================================

The canonical *instance-based* (lazy) learner: there is no training phase at
all. To classify a point, find its k closest neighbors in feature space and let
them vote. All the "learning" is deferred to query time.

Architecture Diagram / Layout:
    query x*  ->  distances to every training point  d(x*, x_i)
              ->  keep the k smallest                 -> {neighbor labels}
              ->  vote:  uniform  (each neighbor = 1 vote)
                         distance (closer neighbors weighted 1/d)
              ->  predicted class = argmax of the (weighted) vote

Key insights / educational takeaways:
    * No parameters are fit — the "model" is the training set itself. Prediction
      cost grows with the dataset (the opposite trade-off from trees/neural nets).
    * Distances mix features additively, so an unscaled feature with a large
      numeric range dominates. This script prints accuracy WITH and WITHOUT
      standardization to make the effect concrete — scaling is not optional here.
    * k is the bias/variance dial: k=1 memorizes (low bias, high variance),
      large k oversmooths (high bias).

Run:
    python "05.knn.py" --k 7 --variant distance --dataset covtype
    python "05.knn.py" --limit 2000        # fast smoke test
"""

import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import tabular_common as mc


def main():
    p = mc.build_argparser("k-Nearest Neighbors Tabular Classifier")
    p.add_argument("--variant", choices=["uniform", "distance"], default="uniform",
                   help="neighbor vote weighting")
    p.add_argument("--k", type=int, default=5, help="number of neighbors")
    args = p.parse_args()
    mc.set_seed(args.seed)

    X_train, X_test, y_train, y_test, feature_names, class_names = \
        mc.load_classification_dataset(args.dataset)
    if args.limit is not None:
        X_train, y_train = X_train[:args.limit], y_train[:args.limit]

    # Demonstrate WHY standardization is mandatory for distance-based models.
    raw = KNeighborsClassifier(n_neighbors=args.k, weights=args.variant)
    raw.fit(X_train, y_train)
    acc_raw = accuracy_score(y_test, raw.predict(X_test))

    Xtr_s, Xte_s = mc.standardize(X_train, X_test)
    print(f"Training k-NN (k={args.k}, weights={args.variant})...")
    model = KNeighborsClassifier(n_neighbors=args.k, weights=args.variant)
    model.fit(Xtr_s, y_train)
    acc_scaled = accuracy_score(y_test, model.predict(Xte_s))

    print("-" * 64)
    print(f"Test accuracy  (raw features)      : {acc_raw * 100:.2f}%")
    print(f"Test accuracy  (standardized)      : {acc_scaled * 100:.2f}%")
    print(f"Standardization gain               : {(acc_scaled - acc_raw) * 100:+.2f} pts")

    preds = model.predict(Xte_s)
    proba = model.predict_proba(Xte_s)
    mc.report_classification(y_test, preds, proba, class_names=class_names,
                             model_name=f"kNN-k{args.k}-{args.variant}",
                             save_dir=None if args.no_figure else
                             os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
