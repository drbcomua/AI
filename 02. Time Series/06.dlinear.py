"""
06. DLinear / NLinear (Zeng et al., 2023 — "Are Transformers Effective for Time Series Forecasting?")
====================================================================================================

A deliberately provocative baseline: a *single linear layer* that, on the standard
long-term forecasting benchmarks, matched or beat the elaborate transformers of
its day. It is the time-series analogue of "do we even need attention?".

Architecture Diagram / Layout:
    DLinear:
        Input [Batch, W, F] -> Series Decomposition (moving-avg)
              -> Trend  [B, W, F] --Linear(W->1)--\
              -> Season [B, W, F] --Linear(W->1)----(+)--> [B, F] --Linear(F->1)--> [B, 1]
    NLinear:
        Input [B, W, F] -> subtract last value -> Linear(W->1) -> add last value
              -> [B, F] -> Linear(F->1) -> [B, 1]

Key insights / educational takeaways:
    * DLinear splits the series into trend + seasonal parts and learns a separate
      linear map for each — surprisingly strong on data with clear trend/season.
    * NLinear simply normalizes by the last value, neutralizing distribution shift
      between train and test — a one-line trick that helps a lot on noisy series.
    * If a linear layer rivals your transformer, the transformer may be overkill.

Run:
    python "06.dlinear.py" --dataset jena --variant dlinear --epochs 5
    python "06.dlinear.py" --dataset spy --variant nlinear --epochs 5
"""

import os
import torch
import torch.nn as nn
import ts_common as mc


class MovingAvg(nn.Module):
    """Moving average that preserves length via edge replication padding."""
    def __init__(self, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.avg = nn.AvgPool1d(kernel, stride=1, padding=0)

    def forward(self, x):                                  # x: [B, W, F]
        pad = (self.kernel - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, self.kernel - 1 - pad, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel: int = 5):
        super().__init__()
        self.moving_avg = MovingAvg(kernel)

    def forward(self, x):
        trend = self.moving_avg(x)
        return x - trend, trend                           # seasonal, trend


class DLinear(nn.Module):
    def __init__(self, num_features: int, seq_len: int = 24, kernel: int = 5):
        super().__init__()
        self.decomp = SeriesDecomp(kernel)
        self.lin_seasonal = nn.Linear(seq_len, 1)
        self.lin_trend = nn.Linear(seq_len, 1)
        self.combine = nn.Linear(num_features, 1)

    def forward(self, x):                                  # [B, W, F]
        seasonal, trend = self.decomp(x)
        s = self.lin_seasonal(seasonal.permute(0, 2, 1)).squeeze(-1)   # [B, F]
        t = self.lin_trend(trend.permute(0, 2, 1)).squeeze(-1)         # [B, F]
        return self.combine(s + t)                                     # [B, 1]


class NLinear(nn.Module):
    def __init__(self, num_features: int, seq_len: int = 24):
        super().__init__()
        self.lin = nn.Linear(seq_len, 1)
        self.combine = nn.Linear(num_features, 1)

    def forward(self, x):                                  # [B, W, F]
        last = x[:, -1:, :]                                # [B, 1, F]
        out = self.lin((x - last).permute(0, 2, 1)).squeeze(-1)        # [B, F]
        return self.combine(out + last.squeeze(1))                     # [B, 1]


def main():
    args = mc.build_argparser("DLinear / NLinear Forecasters").parse_args()
    W = 24
    device = mc.get_device(args.device)

    train_loader, test_loader, num_features, mean, std = mc.get_dataloaders(
        args.dataset, args.batch_size, args.limit, W=W)

    variant = args.variant or "dlinear"
    if variant == "dlinear":
        model, model_name = DLinear(num_features, W), "DLinear"
    elif variant == "nlinear":
        model, model_name = NLinear(num_features, W), "NLinear"
    else:
        raise ValueError(f"Unknown variant: {variant}")

    mc.train(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)
    y_true, y_pred = mc.evaluate(model, test_loader, device=device)
    mc.report(y_true, y_pred, mean, std, dataset_name=args.dataset, model_name=model_name,
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
