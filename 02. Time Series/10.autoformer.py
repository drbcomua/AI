"""
10. Autoformer (Wu et al., 2021 — "Decomposition Transformers with Auto-Correlation")
=====================================================================================

Autoformer rethinks attention for time series. Two ideas:
    * **Series decomposition** is baked *inside* the network: every block splits
      its input into trend (moving average) and seasonal (remainder) and processes
      the seasonal part, progressively refining the trend.
    * **Auto-Correlation** replaces dot-product attention. Instead of comparing
      individual timesteps, it uses the FFT to find the dominant *periods* in the
      series and aggregates sub-series that are one period apart — attention over
      *lags* rather than positions.

Architecture Diagram / Layout:
    Input [B, W, F] -> Linear embed [B, W, d]
       repeat: AutoCorrelation -> decomp -> FeedForward -> decomp
       -> mean over time [B, d] -> Linear -> [B, 1]

Key insights / educational takeaways:
    * Auto-Correlation is O(W log W) via the FFT and is naturally suited to data
      with strong periodicity (Jena's daily/yearly cycles).
    * This is a faithful but compact Autoformer (encoder-only, single-step head).

Run:
    python "10.autoformer.py" --dataset jena --epochs 5
    python "10.autoformer.py" --dataset jena_full --epochs 3
"""

import math
import os
import warnings
import torch
import torch.nn as nn
import ts_common as mc

# torch.fft on the Apple MPS backend emits a benign output-resize deprecation
# warning; the FFT result is correct, so silence the noise.
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")


class MovingAvg(nn.Module):
    def __init__(self, kernel: int = 5):
        super().__init__()
        self.kernel = kernel
        self.avg = nn.AvgPool1d(kernel, stride=1, padding=0)

    def forward(self, x):
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
        return x - trend, trend


class AutoCorrelation(nn.Module):
    """FFT-based auto-correlation with time-delay aggregation."""
    def __init__(self, d_model: int, factor: int = 2):
        super().__init__()
        self.factor = factor
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):                                  # [B, L, d]
        B, L, d = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        # autocorrelation along time via FFT (per channel), averaged over channels
        qf = torch.fft.rfft(q.permute(0, 2, 1), dim=-1)
        kf = torch.fft.rfft(k.permute(0, 2, 1), dim=-1)
        corr = torch.fft.irfft(qf * torch.conj(kf), n=L, dim=-1)       # [B, d, L]
        mean_corr = corr.mean(dim=1)                                   # [B, L] per-delay score
        k_top = max(1, int(self.factor * math.log(L)))
        weights, delays = torch.topk(mean_corr, k_top, dim=-1)         # [B, k]
        weights = torch.softmax(weights, dim=-1)
        # time-delay aggregation: roll v by each top delay, weight, sum (vectorized)
        base = torch.arange(L, device=x.device).view(1, L)
        agg = torch.zeros_like(v)
        for i in range(k_top):
            idx = (base + delays[:, i:i + 1]) % L                      # [B, L]
            rolled = torch.gather(v, 1, idx.unsqueeze(-1).expand(B, L, d))
            agg = agg + rolled * weights[:, i].view(B, 1, 1)
        return self.out(agg)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 128, kernel: int = 5):
        super().__init__()
        self.autocorr = AutoCorrelation(d_model)
        self.decomp1 = SeriesDecomp(kernel)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.decomp2 = SeriesDecomp(kernel)

    def forward(self, x):
        seasonal, _ = self.decomp1(x + self.autocorr(x))
        seasonal, _ = self.decomp2(seasonal + self.ff(seasonal))
        return seasonal


class Autoformer(nn.Module):
    def __init__(self, num_features: int, d_model: int = 64, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Linear(num_features, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):                                  # [B, W, F]
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x.mean(dim=1))                   # [B, 1]


def main():
    args = mc.build_argparser("Autoformer Forecaster").parse_args()
    W = 24
    device = mc.get_device(args.device)

    train_loader, test_loader, num_features, mean, std = mc.get_dataloaders(
        args.dataset, args.batch_size, args.limit, W=W)

    model = Autoformer(num_features)
    mc.train(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)
    y_true, y_pred = mc.evaluate(model, test_loader, device=device)
    mc.report(y_true, y_pred, mean, std, dataset_name=args.dataset, model_name="Autoformer",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
