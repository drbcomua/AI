"""
07. N-BEATS / N-HiTS (Oreshkin et al., 2020 / Challu et al., 2023)
=================================================================

Pure-MLP forecasters built on *doubly residual stacking*: each block predicts a
"backcast" (its explanation of the input, which is subtracted off) and a
"forecast" (its additive contribution to the output). Stacking blocks lets the
model peel the signal apart layer by layer — no recurrence, no attention.

Architecture Diagram / Layout:
    residual = input series [B, W]
    for each block:
        h = MLP(residual)
        backcast, forecast = Linear(h)->W , Linear(h)->H
        residual = residual - backcast        # doubly residual
        output  += forecast
    -> output [B, H]

Key insights / educational takeaways:
    * N-BEATS: generic fully-connected blocks; interpretable trend/seasonality
      variants are possible by constraining the basis.
    * N-HiTS: each block first *pools* the input at a different rate, so different
      blocks specialize in different frequencies (multi-rate hierarchy). This is
      cheaper and better on long horizons.
    (Multivariate inputs are first collapsed to a single series via a linear layer.)

Run:
    python "07.nbeats-nhits.py" --dataset jena --variant nbeats --epochs 5
    python "07.nbeats-nhits.py" --dataset spy --variant nhits --epochs 5
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import ts_common as mc


class Block(nn.Module):
    """One doubly-residual block. `pool` > 1 makes it an N-HiTS multi-rate block."""
    def __init__(self, seq_len: int, horizon: int, hidden: int = 128, pool: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.pool = pool
        pooled = seq_len // pool
        self.mlp = nn.Sequential(
            nn.Linear(pooled, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.theta_b = nn.Linear(hidden, pooled)          # backcast (in pooled space)
        self.theta_f = nn.Linear(hidden, horizon)         # forecast

    def forward(self, x):                                 # x: [B, W]
        z = x
        if self.pool > 1:
            z = F.max_pool1d(x.unsqueeze(1), self.pool).squeeze(1)     # multi-rate downsample
        h = self.mlp(z)
        backcast = self.theta_b(h)
        forecast = self.theta_f(h)
        if self.pool > 1:                                 # interpolate backcast back to W
            backcast = F.interpolate(backcast.unsqueeze(1), size=self.seq_len,
                                     mode="linear", align_corners=False).squeeze(1)
        return backcast, forecast


class NBeats(nn.Module):
    def __init__(self, num_features: int, seq_len: int = 24, horizon: int = 1, pools=(1, 1, 1)):
        super().__init__()
        self.collapse = nn.Linear(num_features, 1)
        self.blocks = nn.ModuleList([Block(seq_len, horizon, pool=p) for p in pools])

    def forward(self, x):                                 # [B, W, F]
        residual = self.collapse(x).squeeze(-1)           # [B, W]
        forecast = 0.0
        for block in self.blocks:
            b, f = block(residual)
            residual = residual - b
            forecast = forecast + f
        return forecast                                   # [B, H]


def main():
    args = mc.build_argparser("N-BEATS / N-HiTS Forecasters").parse_args()
    W = 24
    device = mc.get_device(args.device)

    train_loader, test_loader, num_features, mean, std = mc.get_dataloaders(
        args.dataset, args.batch_size, args.limit, W=W)

    variant = args.variant or "nbeats"
    if variant == "nbeats":
        model, model_name = NBeats(num_features, W, pools=(1, 1, 1)), "N-BEATS"
    elif variant == "nhits":
        model, model_name = NBeats(num_features, W, pools=(4, 2, 1)), "N-HiTS"
    else:
        raise ValueError(f"Unknown variant: {variant}")

    mc.train(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)
    y_true, y_pred = mc.evaluate(model, test_loader, device=device)
    mc.report(y_true, y_pred, mean, std, dataset_name=args.dataset, model_name=model_name,
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
