"""
tabular_common.py
==================

Shared utilities for the structured Tabular ML demos in this folder.

Everything the individual architecture scripts need lives here so each file can
focus purely on the *model*:

    Datasets
        * load_wine_dataset()            -> small 3-class classification (quick smoke tests)
        * load_covtype_dataset(...)      -> 20k-row stratified Forest Cover Type (7 classes)
        * load_classification_dataset()  -> dispatch on the --dataset flag
        * load_california_housing_dataset() -> 8-feature regression track

    Preprocessing
        * standardize(X_train, X_test)   -> z-score, fit on train only

    Command line
        * build_argparser(...)           -> the shared CLI (tree + neural flags)
        * get_device(...)                -> pick cuda / mps / cpu

    Neural training (Phase 3 deep-tabular scripts)
        * get_dataloaders_from_arrays(...) -> TensorDataset DataLoaders
        * train(...)                     -> loop printing per-epoch acc + wall-clock time
        * evaluate(...)                  -> collect y_true / y_pred / y_prob
        * count_parameters(model)        -> exact trainable-parameter count

    Reporting / plotting
        * report_classification(...)     -> research-grade metrics + confusion-matrix PNG
        * report_regression(...)         -> RMSE / MAE / R^2 + residual diagnostics
        * plot_feature_importances(...)  -> horizontal bar chart of importances / masks

The classical sklearn scripts (01-11) only touch the dataset loaders, the
argparser, and the reporting/plot helpers; the deep-tabular scripts (12-17) also
use the neural-training helpers.
"""

from __future__ import annotations

import os
import ssl
import time
import argparse

import numpy as np
from sklearn.datasets import load_wine, fetch_covtype, fetch_california_housing
from sklearn.model_selection import train_test_split

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _fetch_ssl_safe(fetch_fn, **kwargs):
    """Call an sklearn `fetch_*` under an unverified SSL context.

    macOS python.org builds often lack a usable CA bundle, which breaks the
    HTTPS download. Dataset integrity is not security-sensitive here, so we fall
    back to an unverified context and restore the original afterwards.
    """
    orig = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        return fetch_fn(**kwargs)
    finally:
        ssl._create_default_https_context = orig


# --------------------------------------------------------------------------- #
# Classification datasets
# --------------------------------------------------------------------------- #
def load_wine_dataset():
    """scikit-learn's Wine cultivar classification dataset (178 samples, 13 features, 3 classes).

    Tiny and linearly separable — everything saturates near ~98%, so it is only
    useful as a fast smoke test. Use covtype for meaningful model comparison.

    Returns:
        X_train, X_test, y_train, y_test (float32/int64 arrays),
        feature_names (list[str]), class_names (list[str])
    """
    wine = load_wine()
    X = wine.data.astype(np.float32)
    y = wine.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Loaded Wine dataset: {X.shape[0]} samples | "
          f"features {X.shape[1]} | classes {len(np.unique(y))}")
    return (X_train, X_test, y_train, y_test,
            list(wine.feature_names), list(wine.target_names))


# Forest Cover Type feature layout: 10 quantitative, then 4 one-hot wilderness
# areas, then 40 one-hot soil types (54 columns total).
_COVTYPE_QUANT = [
    "Elevation", "Aspect", "Slope",
    "Horiz_Dist_Hydrology", "Vert_Dist_Hydrology", "Horiz_Dist_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horiz_Dist_Fire_Points",
]
_COVTYPE_FEATURES = (
    _COVTYPE_QUANT
    + [f"Wilderness_Area_{i}" for i in range(1, 5)]
    + [f"Soil_Type_{i}" for i in range(1, 41)]
)
_COVTYPE_CLASSES = [
    "Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow",
    "Aspen", "Douglas-fir", "Krummholz",
]
# Column indices of the categorical (one-hot) blocks — used by 12.mlp-embeddings.
COVTYPE_N_QUANT = 10
COVTYPE_WILDERNESS_COLS = list(range(10, 14))
COVTYPE_SOIL_COLS = list(range(14, 54))


def load_covtype_dataset(n_rows: int = 20000, seed: int = 42):
    """Forest Cover Type (Blackard & Dean, 1999), stratified-subsampled and cached.

    fetch_covtype pulls 581k rows (~50s to parse); we take ``n_rows`` stratified
    samples once and cache them to ``data/covtype_{n_rows}.npz`` for instant reuse.
    On download failure (offline) we fall back to the Wine dataset so scripts
    still run.

    Returns the same 6-tuple as ``load_wine_dataset``. Targets are remapped from
    the original 1..7 labels to 0..6.
    """
    os.makedirs(_DATA_DIR, exist_ok=True)
    cache = os.path.join(_DATA_DIR, f"covtype_{n_rows}.npz")

    if os.path.exists(cache):
        d = np.load(cache)
        X, y = d["X"], d["y"]
    else:
        try:
            data = _fetch_ssl_safe(fetch_covtype)
        except Exception as e:  # pragma: no cover - network dependent
            print(f"(covtype download failed: {e}; falling back to Wine)")
            return load_wine_dataset()

        Xf = data.data.astype(np.float32)
        yf = (data.target.astype(np.int64) - 1)  # 1..7 -> 0..6

        # Stratified subsample to n_rows total.
        idx, _ = train_test_split(
            np.arange(len(yf)), train_size=n_rows, random_state=seed, stratify=yf
        )
        X, y = Xf[idx], yf[idx]
        np.savez_compressed(cache, X=X, y=y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    print(f"Loaded Cover Type dataset: {X.shape[0]} samples | "
          f"features {X.shape[1]} | classes {len(np.unique(y))}")
    return (X_train.astype(np.float32), X_test.astype(np.float32),
            y_train.astype(np.int64), y_test.astype(np.int64),
            list(_COVTYPE_FEATURES), list(_COVTYPE_CLASSES))


def load_classification_dataset(name: str = "wine"):
    """Dispatch on the ``--dataset`` flag. Returns the standard 6-tuple."""
    if name == "covtype":
        return load_covtype_dataset()
    return load_wine_dataset()


# --------------------------------------------------------------------------- #
# Regression dataset
# --------------------------------------------------------------------------- #
def load_california_housing_dataset(seed: int = 42):
    """California Housing regression (Pace & Barry, 1997): 8 features, continuous target.

    Target is the median house value in units of $100,000. 80/20 split.

    Returns:
        X_train, X_test, y_train, y_test (float32), feature_names (list[str])
    """
    data = _fetch_ssl_safe(fetch_california_housing)
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    print(f"Loaded California Housing: {X.shape[0]} samples | features {X.shape[1]}")
    return X_train, X_test, y_train, y_test, list(data.feature_names)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def standardize(X_train: np.ndarray, X_test: np.ndarray):
    """Z-score features using statistics fit on the training split only.

    Mandatory for distance/kernel/gradient models (k-NN, SVM, all neural nets).
    Constant columns (std == 0) are left unscaled to avoid divide-by-zero.
    """
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    Xtr = ((X_train - mu) / sigma).astype(np.float32)
    Xte = ((X_test - mu) / sigma).astype(np.float32)
    return Xtr, Xte


# --------------------------------------------------------------------------- #
# Command-line interface
# --------------------------------------------------------------------------- #
def build_argparser(description: str, *, max_depth: int = 5, n_estimators: int = 100,
                    lr: float = 0.1, epochs: int = 30, batch_size: int = 256,
                    default_dataset: str = "wine"):
    """Shared CLI. Tree scripts read the tree flags; neural scripts read the rest.

    Every script gets ``--dataset``, ``--limit``, ``--no-figure`` and ``--seed``;
    scripts add their own ``--variant`` / ``--k`` afterwards as needed.
    """
    p = argparse.ArgumentParser(description=description)
    # Dataset selection (classification scripts).
    p.add_argument("--dataset", type=str, default=default_dataset,
                   choices=["wine", "covtype"],
                   help="classification dataset (regression scripts ignore this)")
    # Tree / boosting hyperparameters.
    p.add_argument("--max-depth", type=int, default=max_depth, help="maximum tree depth")
    p.add_argument("--n-estimators", type=int, default=n_estimators,
                   help="number of ensemble members / boosting rounds")
    p.add_argument("--lr", type=float, default=lr,
                   help="learning rate (boosting shrinkage or neural optimizer)")
    # Neural-training hyperparameters (Phase 3).
    p.add_argument("--epochs", type=int, default=epochs, help="neural training epochs")
    p.add_argument("--batch-size", type=int, default=batch_size, help="neural batch size")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    # Universal.
    p.add_argument("--seed", type=int, default=42, help="random seed")
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
    """Seed numpy and (if available) torch for reproducible runs."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Neural training helpers (Phase 3)
# --------------------------------------------------------------------------- #
def get_dataloaders_from_arrays(X_train, y_train, X_test, y_test,
                                batch_size: int = 256):
    """Wrap numpy arrays in shuffled/ordered TensorDataset DataLoaders."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def ds(X, y):
        return TensorDataset(torch.from_numpy(np.asarray(X, np.float32)),
                             torch.from_numpy(np.asarray(y, np.int64)))

    train_loader = DataLoader(ds(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ds(X_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def count_parameters(model) -> int:
    """Exact number of trainable parameters (KAN-vs-MLP protocol requirement)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, train_loader, test_loader, *, epochs: int = 30, lr: float = 1e-3,
          device=None, weight_decay: float = 0.0, criterion=None):
    """Standard classification training loop for the deep-tabular scripts.

    Prints per-epoch train/test accuracy and wall-clock time per epoch (the
    latter is required by the KAN-vs-MLP cost-reporting protocol). ``model(x)``
    must return class logits.
    """
    import torch
    import torch.nn as nn

    device = device or get_device()
    model.to(device)
    criterion = criterion or nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_params = count_parameters(model)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 68)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        dt = time.time() - t0

        train_loss = running_loss / total
        train_acc = correct / total
        test_acc = _torch_accuracy(model, test_loader, device)
        print(f"Epoch {epoch:2d}/{epochs} | loss {train_loss:.4f} | "
              f"train_acc {train_acc:.4f} | test_acc {test_acc:.4f} | "
              f"{dt:.2f}s/epoch")
    print("-" * 68)
    return model


def _torch_accuracy(model, loader, device):
    import torch
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def evaluate(model, loader, device=None):
    """Run ``model`` over ``loader``; return (y_true, y_pred, y_prob) numpy arrays."""
    import torch
    device = device or get_device()
    model.to(device)
    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            p = torch.softmax(out, dim=1)
            ys.append(y.numpy())
            preds.append(p.argmax(1).cpu().numpy())
            probs.append(p.cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)


# --------------------------------------------------------------------------- #
# Classification reporting
# --------------------------------------------------------------------------- #
def _wilson_interval(correct: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return center - half, center + half


def report_classification(y_true, y_pred, y_prob=None, *, class_names=None,
                          model_name="model", save_dir=None):
    """Print a research-grade classification report and save a confusion-matrix PNG.

    Chance level (for the p-value against blind guessing) is inferred as
    1 / n_classes. Returns the confusion matrix.
    """
    from sklearn.metrics import (classification_report, cohen_kappa_score,
                                 confusion_matrix, log_loss, matthews_corrcoef,
                                 precision_recall_fscore_support)
    from scipy.stats import binomtest

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = len(class_names) if class_names is not None else int(y_true.max()) + 1
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    chance = 1.0 / n_classes

    n = len(y_true)
    correct = int((y_true == y_pred).sum())
    acc = correct / n
    lo, hi = _wilson_interval(correct, n)
    pval = binomtest(correct, n, p=chance, alternative="greater").pvalue

    pr_m, rc_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    pr_w, rc_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"\n================  TEST REPORT: {model_name}  ================")
    print(f"Samples               : {n}")
    print(f"Accuracy              : {acc:.4f}  (95% Wilson CI [{lo:.4f}, {hi:.4f}])")
    print(f"Error rate            : {1 - acc:.4f}")
    print(f"p-value vs chance     : {pval:.3e}  "
          f"(H0: acc <= {chance:.3f}, binomial one-sided)")
    print(f"F1  (macro / weighted): {f1_m:.4f} / {f1_w:.4f}")
    print(f"Precision (macro/wtd) : {pr_m:.4f} / {pr_w:.4f}")
    print(f"Recall    (macro/wtd) : {rc_m:.4f} / {rc_w:.4f}")
    print(f"Cohen's kappa         : {kappa:.4f}")
    print(f"Matthews corrcoef     : {mcc:.4f}")

    if y_prob is not None:
        ll = log_loss(y_true, y_prob, labels=list(range(n_classes)))
        print(f"Log loss (test)       : {ll:.4f}")

    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, labels=list(range(n_classes)),
                                target_names=class_names, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    if save_dir is not None:
        _save_confusion_png(cm, class_names, model_name, save_dir)
    print("=" * (len(model_name) + 44) + "\n")
    return cm


def _save_confusion_png(cm, class_names, model_name, save_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping confusion-matrix figure: {e})")
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=6)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = os.path.join(save_dir, f"confusion_{_slug(model_name)}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved confusion matrix figure -> {out}")


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name).strip("-").lower()


# --------------------------------------------------------------------------- #
# Regression reporting
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred):
    """Return (rmse, mae, r2)."""
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return rmse, mae, r2


def report_regression(y_true, y_pred, *, model_name="model", save_dir=None):
    """Print RMSE / MAE / R^2 and save a predicted-vs-true scatter PNG."""
    rmse, mae, r2 = regression_metrics(y_true, y_pred)
    print(f"\n================  TEST REPORT: {model_name}  ================")
    print(f"Samples               : {len(y_true)}")
    print(f"RMSE                  : {rmse:.4f}")
    print(f"MAE                   : {mae:.4f}")
    print(f"R^2                   : {r2:.4f}")
    if save_dir is not None:
        _save_regression_png(y_true, y_pred, model_name, save_dir)
    print("=" * (len(model_name) + 44) + "\n")
    return rmse, mae, r2


def _save_regression_png(y_true, y_pred, model_name, save_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping regression figure: {e})")
        return
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=6, alpha=0.3, color="teal", edgecolors="none")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="perfect")
    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value")
    ax.set_title(f"Predicted vs. True — {model_name}")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(save_dir, f"regression_{_slug(model_name)}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved regression scatter figure -> {out}")


# --------------------------------------------------------------------------- #
# Feature-importance / attention-mask plot
# --------------------------------------------------------------------------- #
def plot_feature_importances(importances: np.ndarray, feature_names: list,
                             save_path: str, title: str, top_k: int | None = 20):
    """Horizontal bar chart of importances (or aggregated attention masks).

    With many features (covtype has 54) only the ``top_k`` most important are
    shown to keep the chart readable; pass ``top_k=None`` to show all.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping feature importance plot: {e})")
        return

    importances = np.asarray(importances, dtype=float)
    order = np.argsort(importances)
    if top_k is not None and len(order) > top_k:
        order = order[-top_k:]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(order))))
    ax.barh(range(len(order)), importances[order], color="teal",
            align="center", edgecolor="black", alpha=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=8)
    ax.set_xlabel("Relative Importance Score", fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved feature importance chart -> {save_path}")
