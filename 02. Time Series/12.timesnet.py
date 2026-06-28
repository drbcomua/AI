"""
12. TimesNet (Wu et al., 2023 — "Temporal 2D-Variation Modeling for General Time Series")
========================================================================================

TimesNet's trick: a 1D series often contains several overlapping periods (daily,
weekly, yearly...). It uses the FFT to find the dominant periods, then *folds* the
1D series into a 2D tensor of shape (num_periods x period) so that:
    * moving down a column = variation WITHIN a period (intra-period)
    * moving across a row   = variation BETWEEN periods (inter-period)
A 2D convolution (an inception block) then captures both at once. Outputs for the
top-k periods are combined, weighted by their FFT amplitude.

Architecture Diagram / Layout:
    Input [B, W, F] -> Linear embed [B, W, d]
       TimesBlock:  FFT -> top-k periods -> reshape each to 2D -> 2D Inception conv
                    -> reshape back -> amplitude-weighted sum  (+ residual)
       -> mean over time [B, d] -> Linear -> [B, 1]

Key insights / educational takeaways:
    * Recasting temporal modeling as 2D vision lets convnets exploit multi-period
      structure — a genuinely different inductive bias from RNN/attention.
    * Periods are chosen from the batch-averaged spectrum so the fold is uniform.

Run:
    python "12.timesnet.py" --dataset jena --epochs 5
    python "12.timesnet.py" --dataset jena_full --epochs 3
"""

import math
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import ts_common as mc

# torch.fft on the Apple MPS backend emits a benign output-resize deprecation
# warning; the FFT result is correct, so silence the noise.
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")


class Inception2D(nn.Module):
    """Parallel 2D convolutions of several kernel sizes, summed."""
    def __init__(self, in_ch: int, out_ch: int, kernels=(1, 3, 5)):
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv2d(in_ch, out_ch, k, padding=k // 2) for k in kernels])

    def forward(self, x):
        return sum(b(x) for b in self.branches) / len(self.branches)


class TimesBlock(nn.Module):
    def __init__(self, d_model: int, k: int = 2):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(Inception2D(d_model, d_model), nn.GELU(),
                                  Inception2D(d_model, d_model))

    def forward(self, x):                                  # [B, L, d]
        B, L, d = x.shape
        xf = torch.fft.rfft(x, dim=1)                     # [B, F, d]
        amp = xf.abs().mean(0).mean(-1)                   # [F] batch+channel mean spectrum
        amp[0] = 0.0                                      # drop the DC component
        k = min(self.k, amp.numel() - 1)
        _, top = torch.topk(amp, max(1, k))               # scalar frequency indices
        per_batch_amp = torch.softmax(xf.abs().mean(-1)[:, top], dim=-1)   # [B, k]

        outs = []
        for j, f in enumerate(top.tolist()):
            period = max(1, L // max(1, f))
            pad = (math.ceil(L / period) * period) - L
            xp = F.pad(x.transpose(1, 2), (0, pad)).transpose(1, 2) if pad else x
            rows = xp.shape[1] // period
            img = xp.reshape(B, rows, period, d).permute(0, 3, 1, 2)       # [B, d, rows, period]
            img = self.conv(img)
            back = img.permute(0, 2, 3, 1).reshape(B, rows * period, d)[:, :L, :]
            outs.append(back)
        out = torch.stack(outs, dim=-1)                                    # [B, L, d, k]
        out = (out * per_batch_amp.view(B, 1, 1, -1)).sum(-1)             # amplitude-weighted
        return out + x                                                     # residual


class TimesNet(nn.Module):
    def __init__(self, num_features: int, d_model: int = 32, n_blocks: int = 2):
        super().__init__()
        self.embed = nn.Linear(num_features, d_model)
        self.blocks = nn.ModuleList([TimesBlock(d_model) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):                                  # [B, W, F]
        x = self.embed(x)
        for block in self.blocks:
            x = self.norm(block(x))
        return self.head(x.mean(dim=1))                   # [B, 1]


def main():
    args = mc.build_argparser("TimesNet Forecaster").parse_args()
    W = 24
    device = mc.get_device(args.device)

    train_loader, test_loader, num_features, mean, std = mc.get_dataloaders(
        args.dataset, args.batch_size, args.limit, W=W)

    model = TimesNet(num_features)
    mc.train(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)
    y_true, y_pred = mc.evaluate(model, test_loader, device=device)
    mc.report(y_true, y_pred, mean, std, dataset_name=args.dataset, model_name="TimesNet",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
