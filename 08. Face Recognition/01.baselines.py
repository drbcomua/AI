"""
01. Eigenfaces & Fisherfaces Baselines
======================================

Classical, non-deep metric learning baseline face verification models.

Algorithms:
    * Eigenfaces (PCA): Projects images into principal component subspace maximizing variance.
    * Fisherfaces (LDA): Maximizes between-class variance while minimizing within-class variance.

Run:
    python "01.baselines.py" --variant pca
    python "01.baselines.py" --variant lda
"""

import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import face_common as mc


def run_pca_baseline(train_img, train_lbl, test_img, test_lbl, no_figure):
    """Eigenfaces PCA baseline."""
    # Flatten images for linear operations: shape [N, 3*H*W]
    N_tr, C, H, W = train_img.shape
    X_train = train_img.reshape(N_tr, -1).numpy()
    Y_train = train_lbl.numpy()

    N_te = test_img.shape[0]
    X_test = test_img.reshape(N_te, -1).numpy()
    Y_test = test_lbl.numpy()

    # Fit PCA
    n_components = min(128, X_train.shape[0] - 1)
    print(f"Fitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    pca.fit(X_train)

    # Project test images
    X_test_proj = pca.transform(X_test)

    # Generate verification test pairs on the unseen identities in the test set
    # Using local identity maps
    import torch
    test_img_t = torch.tensor(X_test_proj, dtype=torch.float32)
    test_lbl_t = torch.tensor(Y_test, dtype=torch.long)
    pairs1, pairs2, pair_labels = mc.generate_verification_pairs(test_img_t, test_lbl_t, num_pairs=1000)

    # Calculate L2 distances
    distances = np.linalg.norm(pairs1.numpy() - pairs2.numpy(), axis=1)

    # Evaluate
    print("Evaluating PCA (Eigenfaces)...")
    save_dir = os.path.dirname(os.path.abspath(__file__))
    roc_path = os.path.join(save_dir, "pca_baseline_roc.png")
    mc.plot_verification_roc(pair_labels.numpy(), distances, roc_path, "Eigenfaces PCA")


def run_lda_baseline(train_img, train_lbl, test_img, test_lbl, no_figure):
    """Fisherfaces LDA baseline."""
    N_tr, C, H, W = train_img.shape
    X_train = train_img.reshape(N_tr, -1).numpy()
    Y_train = train_lbl.numpy()

    N_te = test_img.shape[0]
    X_test = test_img.reshape(N_te, -1).numpy()
    Y_test = test_lbl.numpy()

    # Fit LDA
    num_classes = len(np.unique(Y_train))
    n_components = min(128, num_classes - 1)
    print(f"Fitting LDA with {n_components} components...")
    lda = LDA(n_components=n_components)
    lda.fit(X_train, Y_train)

    # Project test images
    X_test_proj = lda.transform(X_test)

    # Generate verification test pairs
    import torch
    test_img_t = torch.tensor(X_test_proj, dtype=torch.float32)
    test_lbl_t = torch.tensor(Y_test, dtype=torch.long)
    pairs1, pairs2, pair_labels = mc.generate_verification_pairs(test_img_t, test_lbl_t, num_pairs=1000)

    # Calculate L2 distances
    distances = np.linalg.norm(pairs1.numpy() - pairs2.numpy(), axis=1)

    # Evaluate
    print("Evaluating LDA (Fisherfaces)...")
    save_dir = os.path.dirname(os.path.abspath(__file__))
    roc_path = os.path.join(save_dir, "lda_baseline_roc.png")
    mc.plot_verification_roc(pair_labels.numpy(), distances, roc_path, "Fisherfaces LDA")


def main():
    p = mc.build_argparser("Eigenfaces & Fisherfaces LFW Baselines")
    p.add_argument("--baseline-variant", type=str, default="pca", choices=["pca", "lda"])
    args = p.parse_args()

    variant = args.variant or args.baseline_variant
    if variant not in ["pca", "lda"]:
        variant = "pca"

    # Load LFW
    images, labels = mc.load_lfw()

    # Split by identities (ensures test set consists of completely unseen people)
    train_img, train_lbl, test_img, test_lbl = mc.split_lfw_identities(images, labels)

    if variant == "pca":
        run_pca_baseline(train_img, train_lbl, test_img, test_lbl, args.no_figure)
    else:
        run_lda_baseline(train_img, train_lbl, test_img, test_lbl, args.no_figure)


if __name__ == "__main__":
    main()
