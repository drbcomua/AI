"""
04. Temporal Convolutional Networks (TCN)
=========================================

1D Dilated Causal Convolutional networks for time-series forecasting.

Architecture Diagram / Layout:
    Input [Batch, W, Features] -> Transpose [Batch, Features, W]
       -> Causal Pad & Dilated Conv1d [Batch, Hidden, W] -> Relu -> Normalization
       -> Causal Pad & Dilated Conv1d [Batch, Hidden, W] -> Relu -> Normalization
       -> Residual Connection (if channels differ, project input)
       -> Output from Last Sequence Step [Batch, Hidden]
       -> Linear [Batch, Hidden -> 1]
       -> Output [Batch, 1]

Key insights / educational takeaways:
    * Dilated convolutions allow the model's receptive field to grow exponentially with depth.
    * Causal masking guarantees that predictions at step t only depend on historical steps <= t (no future data leakage).
    * Compared to RNNs, TCNs can process long sequences in parallel during training, avoiding sequential dependency bottlenecks.

Run:
    python "04.tcn.py" --dataset jena --epochs 5
    python "04.tcn.py" --dataset spy --epochs 5
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import ts_common as mc


class CausalConv1d(nn.Module):
    """1D Convolution with causal padding on the left to prevent future leakage."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        # x shape: [Batch, Channels, W]
        # Pad on the left side of the time dimension (last dimension)
        x_padded = F.pad(x, (self.padding, 0))
        return self.conv(x_padded)


class TemporalBlock(nn.Module):
    """Residual TCN Block with causal padding, dilation, and normalization."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.norm1, self.relu1, self.drop1,
            self.conv2, self.norm2, self.relu2, self.drop2
        )

        # Skip connection: match channel dimensions if needed
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNForecaster(nn.Module):
    """Temporal Convolutional Network for time-series regression."""
    def __init__(self, input_dim: int, num_channels: list[int] = [32, 32], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_c = input_dim if i == 0 else num_channels[i - 1]
            out_c = num_channels[i]
            layers.append(
                TemporalBlock(in_c, out_c, kernel_size, dilation=dilation_size, dropout=dropout)
            )
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # Input shape: [Batch, W, Features]
        # Transpose to match 1D convolution: [Batch, Features, W]
        x = x.transpose(1, 2)
        out = self.tcn(x) # [Batch, Hidden, W]
        # Gather last output step
        last_step = out[:, :, -1] # [Batch, Hidden]
        return self.fc(last_step)


def main():
    p = mc.build_argparser("Temporal Convolutional Network Forecaster")
    args = p.parse_args()

    W = 24
    device = mc.get_device(args.device)

    train_loader, test_loader, num_features, mean, std = mc.get_dataloaders(
        args.dataset, args.batch_size, args.limit, W=W
    )

    model = TCNForecaster(input_dim=num_features, num_channels=[32, 32], kernel_size=3)
    model_name = "TCN"

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
