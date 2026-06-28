"""
08. TSMixer (Chen et al., 2023 — Google, "An All-MLP Architecture for Time Series Forecasting")
==============================================================================================

TSMixer is the MLP-Mixer idea ported to forecasting: stack blocks that alternate
*time-mixing* and *feature-mixing* MLPs, each with a residual connection. No
attention, no convolution — just two kinds of fully-connected mixing.

Architecture Diagram / Layout:
    Input [B, W, F]
    repeat N times:
        time-mixing    : LayerNorm -> transpose -> MLP(W->W) -> transpose -> (+)
        feature-mixing : LayerNorm -> MLP(F->F) -> (+)
    -> Linear(W->1) over time -> [B, F] -> Linear(F->1) -> [B, 1]

Key insights / educational takeaways:
    * Time-mixing shares information *across timesteps*; feature-mixing shares it
      *across variables*. Interleaving the two is enough to be competitive with
      transformers while staying cheap and easy to train.
    * A clean demonstration that "mixing" (not specifically attention) is the
      operation that matters — the same lesson as MLP-Mixer in vision.

Run:
    python "08.tsmixer.py" --dataset jena --epochs 5
    python "08.tsmixer.py" --dataset spy --epochs 5
"""

import os
import torch.nn as nn
import ts_common as mc


class MixerBlock(nn.Module):
    def __init__(self, seq_len: int, num_features: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(num_features)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len), nn.ReLU(), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(num_features)
        self.feat_mlp = nn.Sequential(
            nn.Linear(num_features, num_features), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):                                  # [B, W, F]
        y = self.norm1(x).transpose(1, 2)                 # [B, F, W]
        x = x + self.time_mlp(y).transpose(1, 2)          # time mixing
        x = x + self.feat_mlp(self.norm2(x))              # feature mixing
        return x


class TSMixer(nn.Module):
    def __init__(self, num_features: int, seq_len: int = 24, n_blocks: int = 4):
        super().__init__()
        self.blocks = nn.Sequential(
            *[MixerBlock(seq_len, num_features) for _ in range(n_blocks)])
        self.temporal = nn.Linear(seq_len, 1)
        self.head = nn.Linear(num_features, 1)

    def forward(self, x):                                  # [B, W, F]
        x = self.blocks(x)
        x = self.temporal(x.transpose(1, 2)).squeeze(-1)  # [B, F]
        return self.head(x)                               # [B, 1]


def main():
    args = mc.build_argparser("TSMixer Forecaster").parse_args()
    W = 24
    device = mc.get_device(args.device)

    train_loader, test_loader, num_features, mean, std = mc.get_dataloaders(
        args.dataset, args.batch_size, args.limit, W=W)

    model = TSMixer(num_features, W)
    mc.train(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)
    y_true, y_pred = mc.evaluate(model, test_loader, device=device)
    mc.report(y_true, y_pred, mean, std, dataset_name=args.dataset, model_name="TSMixer",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
