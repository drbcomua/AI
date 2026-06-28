"""
02. Linear & MLP Forecasters
============================

Autoregressive Linear models and Multi-Layer Perceptrons for time-series forecasting.

Architecture Diagram / Layout:
    Input [Batch, W, Features] 
       -> Flatten [Batch, W * Features]
       -> Linear [Batch, W * Features -> Hidden] -> ReLU (if MLP)
       -> Linear [Batch, Hidden -> 1]            (or directly [W * Features -> 1] for Linear)
       -> Output [Batch, 1]

Run:
    python "02.linear-mlp.py" --dataset jena --variant linear --epochs 5
    python "02.linear-mlp.py" --dataset spy --variant mlp --epochs 5
"""

import os
import torch
import torch.nn as nn
import ts_common as mc


class LinearAR(nn.Module):
    """Simple linear autoregressive model."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: [Batch, W, Features]
        x_flat = x.flatten(start_dim=1) # [Batch, W * Features]
        return self.fc(x_flat)


class MLPForecaster(nn.Module):
    """Standard Multi-Layer Perceptron time-series forecasting model."""
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x: [Batch, W, Features]
        x_flat = x.flatten(start_dim=1) # [Batch, W * Features]
        return self.net(x_flat)


def main():
    p = mc.build_argparser("Linear & MLP time-series models")
    args = p.parse_args()

    # Default W=24 (window size)
    W = 24
    device = mc.get_device(args.device)

    train_loader, test_loader, num_features, mean, std = mc.get_dataloaders(
        args.dataset, args.batch_size, args.limit, W=W
    )

    input_dim = W * num_features
    variant = args.variant or "linear"

    if variant == "linear":
        model = LinearAR(input_dim)
        model_name = "Linear-AR"
    elif variant == "mlp":
        model = MLPForecaster(input_dim)
        model_name = "MLP"
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Train
    mc.train(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)

    # Evaluate
    y_true, y_pred = mc.evaluate(model, test_loader, device=device)

    # Report
    mc.report(y_true, y_pred, mean, std, dataset_name=args.dataset,
              model_name=model_name,
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
