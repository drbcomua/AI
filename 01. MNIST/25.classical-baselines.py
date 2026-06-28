"""
25. Classical (non-deep) baselines  —  what MNIST looked like before deep learning
==================================================================================

Every architecture in this folder is a neural network, but MNIST was a benchmark
for a decade *before* deep learning, and the classic results table at
yann.lecun.com is full of non-neural methods. This script runs the most famous
ones with scikit-learn so you can see how strong "shallow" learning already was —
and appreciate how little headroom the fancy convnets are actually fighting over.

    --variant knn      k-Nearest Neighbors   (the textbook MNIST baseline)
    --variant svm      RBF-kernel SVM        (the pre-2012 state of the art)
    --variant rf       Random Forest
    --variant logreg   Multinomial logistic regression (a linear classifier)

Each operates on the raw flattened 784-pixel vector scaled to [0, 1] — no
convolutions, no learned features. k-NN and the RBF SVM scale poorly to 60k
samples, so they train on a subsample by default (`--train-size`); Random Forest
and logistic regression use the full set.

Run:
    python "25.classical-baselines.py" --variant knn
    python "25.classical-baselines.py" --variant svm --train-size 10000
    python "25.classical-baselines.py" --variant rf
"""

import argparse
import os
import time

import numpy as np

import mnist_common as mc

# variant -> (builder, default_train_size, has_proba)
def _build(variant):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    return {
        "knn": (KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=-1), 20000, True),
        "svm": (SVC(kernel="rbf", C=5.0, gamma="scale"), 10000, False),
        "rf": (RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0), None, True),
        "logreg": (LogisticRegression(max_iter=200, n_jobs=-1), None, True),
    }[variant]


def main():
    p = argparse.ArgumentParser(description="Classical ML baselines on MNIST")
    p.add_argument("--variant", choices=["knn", "svm", "rf", "logreg"], default="knn")
    p.add_argument("--train-size", type=int, default=None,
                   help="subsample N training images (default: per-method sensible value)")
    p.add_argument("--no-figure", action="store_true")
    args = p.parse_args()

    X_train, y_train, X_test, y_test = mc.load_mnist()
    X_train = (X_train.reshape(len(X_train), -1).astype(np.float32) / 255.0)
    X_test = (X_test.reshape(len(X_test), -1).astype(np.float32) / 255.0)

    estimator, default_n, has_proba = _build(args.variant)
    n = args.train_size if args.train_size is not None else default_n
    if n is not None and n < len(X_train):
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(X_train))[:n]
        X_train, y_train = X_train[idx], y_train[idx]

    print(f"Estimator : {estimator.__class__.__name__}")
    print(f"Train set : {len(X_train):,} samples  |  Test set: {len(X_test):,}")
    print("-" * 64)

    t0 = time.time()
    estimator.fit(X_train, y_train)
    fit_s = time.time() - t0
    t1 = time.time()
    y_pred = estimator.predict(X_test)
    pred_s = time.time() - t1
    y_prob = estimator.predict_proba(X_test) if has_proba else None
    print(f"Fit time  : {fit_s:.1f}s  |  Predict time: {pred_s:.1f}s")
    print("-" * 64)

    mc.report(y_test, y_pred, y_prob, model_name=f"Classical-{args.variant}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
