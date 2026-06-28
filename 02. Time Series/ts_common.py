"""
ts_common.py
============

Shared utilities for the Time Series forecasting architecture demos in this folder.
Exposes data loading, device placement, training, evaluation, and reporting.
"""

from __future__ import annotations

import os
import csv
import zipfile
import urllib.request
import ssl
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Raw download parameters
JENA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
JENA_ZIP = "jena_climate_2009_2016.csv.zip"
JENA_CSV = "jena_climate_2009_2016.csv"

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Populated during loading / arg-parsing; consumed by report() and plotting so
# that individual scripts don't have to thread date/plot info through themselves.
_DATES = None         # date label for every row of the most recently loaded dataset
_TEST_DATES = None    # date label for each test-window target (aligned to predictions)
_ARGS = None          # most recently parsed CLI namespace (plot preferences, etc.)


# --------------------------------------------------------------------------- #
# Data Download & Parsing
# --------------------------------------------------------------------------- #
def _download_and_extract_jena(data_dir: str = _DATA_DIR) -> str:
    """Download and unzip Jena Climate dataset if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, JENA_CSV)
    if os.path.exists(csv_path):
        return csv_path

    zip_path = os.path.join(data_dir, JENA_ZIP)
    if not os.path.exists(zip_path):
        print(f"Downloading Jena Climate dataset from {JENA_URL}...")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(JENA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=120) as r:
            blob = r.read()
        with open(zip_path, "wb") as f:
            f.write(blob)

    print("Extracting Jena Climate CSV...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    # Clean up zip
    try:
        os.remove(zip_path)
    except OSError:
        pass

    return csv_path


def _generate_synthetic_spy(n_samples: int = 5000) -> np.ndarray:
    """Generate realistic synthetic SPY close prices using Geometric Brownian Motion

    with a daily cyclical component and random noise. This ensures the script is
    fully offline-capable and robust.
    """
    np.random.seed(42)
    # Parameters: daily drift (mu), daily volatility (sigma), starting price
    mu = 0.0003
    sigma = 0.012
    S0 = 100.0

    t = np.arange(n_samples)
    # Geometric brownian motion
    wt = np.random.normal(0, 1, n_samples)
    prices = S0 * np.exp(np.cumsum(mu - 0.5 * sigma**2 + sigma * wt))

    # Add a diurnal/weekly soft cycle overlay + high frequency noise
    cycle = 1.5 * np.sin(t * (2 * np.pi / 20)) + 0.5 * np.cos(t * (2 * np.pi / 5))
    prices += cycle
    return prices.astype(np.float32)


SPY_CSV = "spy.csv"


def _download_spy(data_dir: str = _DATA_DIR) -> str:
    """Download REAL SPY daily OHLCV history (since 1993) from Yahoo Finance, cache as CSV.

    Yahoo's chart endpoint is free and needs no API key. The CSV is cached so
    subsequent runs are fully offline.
    """
    import json
    import time
    import datetime as _dt

    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, SPY_CSV)
    if os.path.exists(path):
        return path

    now = int(time.time())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/SPY"
           f"?period1=0&period2={now}&interval=1d")
    print("Downloading real SPY daily history from Yahoo Finance...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
        payload = json.loads(r.read())

    res = payload["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
        for i, t in enumerate(ts):
            o, h, l, c, v = q["open"][i], q["high"][i], q["low"][i], q["close"][i], q["volume"][i]
            if None in (o, h, l, c, v):            # skip incomplete trading days
                continue
            writer.writerow([_dt.date.fromtimestamp(t).isoformat(), o, h, l, c, v])
    return path


def _generate_synthetic_spy_multi(n_samples: int = 8000) -> np.ndarray:
    """Offline fallback: a multivariate (OHLCV) synthetic SPY series."""
    close = _generate_synthetic_spy(n_samples)
    rng = np.random.default_rng(7)
    span = np.abs(rng.normal(0, 0.005, n_samples)) * close
    open_ = close + rng.normal(0, 0.004, n_samples) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    volume = 8e7 * (1.0 + 0.4 * np.abs(rng.normal(0, 1, n_samples)))
    return np.stack([open_, high, low, close, volume], axis=1).astype(np.float32)


def load_dataset(name: str = "jena", limit: int | None = None, data_dir: str = _DATA_DIR):
    """Load raw dataset. Return numpy float32 arrays (X, y) where target is the temperature

    for jena, and the price for spy.
    """
    global _DATES
    dates = None
    if name == "jena":
        csv_path = _download_and_extract_jena(data_dir)
        # Parse CSV. We extract: Temp (col 2), Pressure (col 1), Density (col 11).
        # We take every 6th reading (hourly) to match standard setups.
        features, dates = [], []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            p_idx = header.index("p (mbar)")
            t_idx = header.index("T (degC)")
            rho_idx = header.index("rho (g/m**3)")
            for i, row in enumerate(reader):
                if i % 6 == 0:
                    features.append([float(row[p_idx]), float(row[t_idx]), float(row[rho_idx])])
                    dates.append(row[0])
        data = np.array(features, dtype=np.float32)
        # We want to predict target: temperature (index 1 of our subset)
        target_idx = 1
    elif name == "jena_full":
        # FULL-RESOLUTION, MULTIVARIATE Jena: every 10-min reading (~420k rows) and
        # all 14 numeric channels. This is the "show scale" dataset — far more data
        # and variables than hourly 3-feature `jena`, so high-capacity models
        # (iTransformer, TimesNet, Autoformer, ...) have something to chew on.
        csv_path = _download_and_extract_jena(data_dir)
        features, dates = [], []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            cols = list(range(1, len(header)))          # every column except 'Date Time'
            target_idx = header.index("T (degC)") - 1   # -1 because col 0 is dropped
            for row in reader:
                features.append([float(row[c]) for c in cols])
                dates.append(row[0])
        data = np.array(features, dtype=np.float32)
    elif name == "spy":
        # Synthetic close price; synthesize plausible daily dates for the plot axis.
        prices = _generate_synthetic_spy(5000)
        data = prices.reshape(-1, 1)
        target_idx = 0
        start = datetime.date(2005, 1, 3)
        dates = [(start + datetime.timedelta(days=i)).isoformat() for i in range(len(data))]
    elif name == "spy_full":
        # REAL, MULTIVARIATE SPY: full daily OHLCV history (~8.4k rows since 1993).
        # Falls back to a synthetic OHLCV series if the download is unavailable.
        try:
            csv_path = _download_spy(data_dir)
            rows, dates = [], []
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                next(reader)                                  # skip header
                for row in reader:
                    rows.append([float(row[1]), float(row[2]), float(row[3]),
                                 float(row[4]), float(row[5])])   # O, H, L, C, V
                    dates.append(row[0])
            data = np.array(rows, dtype=np.float32)
        except Exception as e:
            print(f"SPY download unavailable ({e}); using synthetic multivariate fallback.")
            data = _generate_synthetic_spy_multi(8000)
            start = datetime.date(2005, 1, 3)
            dates = [(start + datetime.timedelta(days=i)).isoformat() for i in range(len(data))]
        target_idx = 3                                        # Close
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    if limit is not None:
        data = data[:limit]
        if dates is not None:
            dates = dates[:limit]

    _DATES = dates
    return data, target_idx


def get_dataloaders(name: str = "jena", batch_size: int = 128, limit: int | None = None,
                    W: int = 24, H: int = 1, data_dir: str = _DATA_DIR, num_workers: int = 0):
    """Build normalized sequence windows for train, validation, and test sets.

    Returns (train_loader, test_loader, num_features, mean, std, target_idx).
    Splits: Train 70%, Test 30%.
    """
    data, target_idx = load_dataset(name, limit, data_dir)

    n = len(data)
    train_size = int(n * 0.70)

    train_data = data[:train_size]
    test_data = data[train_size:]

    # Scale the datasets based on training statistics
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    # Avoid zero division
    std[std == 0] = 1.0

    train_norm = (train_data - mean) / std
    test_norm = (test_data - mean) / std

    # Helper to generate sequence windows
    def create_windows(arr):
        X_list, y_list = [], []
        # arr is (N, num_features)
        for i in range(len(arr) - W - H + 1):
            X_list.append(arr[i : i + W])
            # For H=1, target is single step. Extract the target feature.
            y_list.append(arr[i + W + H - 1, target_idx])
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    X_train, y_train = create_windows(train_norm)
    X_test, y_test = create_windows(test_norm)

    # Record the date label of each test-window target (test_loader is not shuffled,
    # so this stays aligned with the predictions evaluate() returns).
    global _TEST_DATES
    if _DATES is not None:
        test_dates = _DATES[train_size:]
        _TEST_DATES = [test_dates[i + W + H - 1] for i in range(len(X_test))]
    else:
        _TEST_DATES = None

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, data.shape[1], mean[target_idx], std[target_idx]


# --------------------------------------------------------------------------- #
# Device, Training, and Evaluation Loops
# --------------------------------------------------------------------------- #
class _Parser(argparse.ArgumentParser):
    """ArgumentParser that records the parsed namespace module-wide.

    This lets report()/plotting honor CLI options (e.g. --plot-window) without
    every architecture script having to pass them through explicitly.
    """
    def parse_args(self, *args, **kwargs):
        global _ARGS
        _ARGS = super().parse_args(*args, **kwargs)
        return _ARGS


def build_argparser(description: str, epochs: int = 5, batch_size: int = 128,
                    lr: float = 1e-3, default_dataset: str = "jena"):
    """Standard argparse builder for the time-series folder."""
    p = _Parser(description=description)
    p.add_argument("--dataset", type=str, default=default_dataset,
                   choices=["jena", "jena_full", "spy", "spy_full"],
                   help="dataset: 'jena' (hourly, 3 feats), 'jena_full' (10-min, 14 feats), "
                        "'spy' (synthetic close), 'spy_full' (real daily OHLCV, 5 feats)")
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--batch-size", type=int, default=batch_size)
    p.add_argument("--lr", type=float, default=lr)
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N samples of the dataset (fast smoke test)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--no-figure", action="store_true", help="do not save the forecast plot")
    p.add_argument("--variant", type=str, default=None, help="architecture variant identifier")
    p.add_argument("--plot-window", type=str, default="recent", choices=["recent", "first"],
                   help="which slice of the test set to plot ('recent' = most recent points)")
    p.add_argument("--plot-points", type=int, default=150,
                   help="number of points to show in the forecast plot")
    return p


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(model, train_loader, test_loader, *, epochs: int = 5, lr: float = 1e-3,
          device=None, criterion=None, optimizer=None):
    """Standard training loop for time-series regression."""
    device = device or get_device()
    model.to(device)
    criterion = criterion or nn.MSELoss()
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            # Match outputs to target shapes
            loss = criterion(out.squeeze(-1), y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

        train_loss = running_loss / total_samples

        # Compute validation loss (MSE) on test loader
        model.eval()
        test_loss = 0.0
        test_samples = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out.squeeze(-1), y)
                test_loss += loss.item() * y.size(0)
                test_samples += y.size(0)
        val_loss = test_loss / test_samples

        print(f"Epoch {epoch:2d}/{epochs} | train_mse {train_loss:.5f} | test_mse {val_loss:.5f}")

    print("-" * 64)
    return model


@torch.no_grad()
def evaluate(model, loader, device=None):
    """Run evaluation and collect predictions & true labels."""
    device = device or get_device()
    model.to(device)
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        ys.append(y.numpy())
        preds.append(out.squeeze(-1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds)


def report(y_true, y_pred, target_mean, target_std, *, dataset_name="jena", model_name="model",
           save_dir=None, plot_window=None, plot_points=None):
    """De-normalize values and print regression reports (MSE, MAE, MAPE, R2, Directional Acc)."""
    # Fall back to CLI choices (captured by build_argparser) when not passed explicitly.
    if plot_window is None:
        plot_window = getattr(_ARGS, "plot_window", "recent") if _ARGS is not None else "recent"
    if plot_points is None:
        plot_points = getattr(_ARGS, "plot_points", 150) if _ARGS is not None else 150
    # De-normalize back to original scale
    y_true_orig = y_true * target_std + target_mean
    y_pred_orig = y_pred * target_std + target_mean

    # Calculate regression metrics
    mse = np.mean((y_true_orig - y_pred_orig) ** 2)
    mae = np.mean(np.abs(y_true_orig - y_pred_orig))
    # Avoid zero division in MAPE
    non_zero_mask = y_true_orig != 0
    mape = np.mean(np.abs((y_true_orig[non_zero_mask] - y_pred_orig[non_zero_mask]) / y_true_orig[non_zero_mask])) * 100

    # R2 score
    ss_res = np.sum((y_true_orig - y_pred_orig) ** 2)
    ss_tot = np.sum((y_true_orig - np.mean(y_true_orig)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Directional accuracy: did we predict up/down correct?
    true_diffs = np.diff(y_true_orig)
    pred_diffs = y_pred_orig[1:] - y_true_orig[:-1] # predicted change from the current actual price
    dir_acc = np.mean((np.sign(true_diffs) == np.sign(pred_diffs)) & (true_diffs != 0)) * 100

    # Prints
    print(f"\n================  FORECAST REPORT: {model_name} ({dataset_name.upper()})  ================")
    print(f"Test samples         : {len(y_true)}")
    print(f"Mean Squared Error   : {mse:.4f}")
    print(f"Mean Absolute Error  : {mae:.4f}")
    print(f"Mean Abs Pct Error   : {mape:.2f}%")
    print(f"R-squared (R2)       : {r2:.4f}")
    print(f"Directional Accuracy : {dir_acc:.2f}%")

    if save_dir is not None:
        _save_forecast_png(y_true_orig, y_pred_orig, dataset_name, model_name, save_dir,
                           plot_window=plot_window, plot_points=plot_points, dates=_TEST_DATES)
    print("=" * (len(model_name) + len(dataset_name) + 33) + "\n")


def _save_forecast_png(y_true, y_pred, dataset_name, model_name, save_dir,
                       plot_window="recent", plot_points=150, dates=None):
    """Save a plot of actual vs. predicted values for a slice of the test set.

    ``plot_window`` selects which slice: 'recent' (default) shows the most recent
    points, 'first' shows the start of the test period. The plotted time range is
    written into the title.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping forecast plot: {e})")
        return

    n = len(y_true)
    n_plot = min(plot_points, n)
    sl = slice(n - n_plot, n) if plot_window == "recent" else slice(0, n_plot)
    yt, yp = y_true[sl], y_pred[sl]

    sliced_dates = dates[sl] if (dates is not None and len(dates) == n) else None
    if sliced_dates is not None:
        rng = f"{sliced_dates[0]} .. {sliced_dates[-1]}"
    else:
        rng = f"steps {sl.start}..{sl.stop - 1}"

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = range(len(yt))
    ax.plot(x, yt, label="Ground Truth", color="#1f77b4", linewidth=1.5)
    ax.plot(x, yp, label="Predicted", color="#ff7f0e", linestyle="--", linewidth=1.5)
    ax.set_title(f"Forecast — {model_name} on {dataset_name.upper()}  "
                 f"({plot_window} {len(yt)}: {rng})")

    if sliced_dates is not None:                       # sparse, readable date ticks
        n_ticks = min(6, len(yt))
        ticks = [int(round(t)) for t in np.linspace(0, len(yt) - 1, n_ticks)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([sliced_dates[t] for t in ticks], rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("Date")
    else:
        ax.set_xlabel("Time step")

    ax.set_ylabel("T (degC)" if dataset_name.startswith("jena") else "USD ($)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    out = os.path.join(save_dir, f"forecast_{model_name}_{dataset_name}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved forecast comparison plot -> {out}")
