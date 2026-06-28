"""
02. GoogLeNet / Inception  (Szegedy et al., 2014 — "Going Deeper with Convolutions")
====================================================================================

GoogLeNet won ILSVRC-2014 and introduced the **Inception module**: instead of
choosing a single filter size, run several in parallel (1x1, 3x3, 5x5) plus a
pooling branch, and concatenate their outputs along the channel axis. The 1x1
convolutions act as cheap "bottlenecks" that reduce channels before the
expensive 3x3 / 5x5 paths, keeping the network deep yet computationally lean.

You already have Xception (the "extreme" Inception); this script fills in the
original Inception module it descends from.

This is a **compact, MNIST-sized adaptation** (not the full 22-layer ImageNet
network): a small stem followed by two Inception modules and global average
pooling. The Inception module itself is faithful to the paper.

Run:
    python "02.googlenet-inception.py" --epochs 5
    python "02.googlenet-inception.py" --limit 2000
"""

import os

import torch
import torch.nn as nn

import mnist_common as mc


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU (BN added per Inception-v2 for stable training)."""

    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Inception(nn.Module):
    """The four-branch Inception module: 1x1 | 1x1->3x3 | 1x1->5x5 | pool->1x1."""

    def __init__(self, in_ch, c1, c3r, c3, c5r, c5, pool_proj):
        super().__init__()
        self.b1 = ConvBlock(in_ch, c1, kernel_size=1)
        self.b2 = nn.Sequential(
            ConvBlock(in_ch, c3r, kernel_size=1),
            ConvBlock(c3r, c3, kernel_size=3, padding=1),
        )
        self.b3 = nn.Sequential(
            ConvBlock(in_ch, c5r, kernel_size=1),
            ConvBlock(c5r, c5, kernel_size=5, padding=2),
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_ch, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class MiniGoogLeNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(1, 32, kernel_size=3, padding=1),   # 28x28
            ConvBlock(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),                              # -> 14x14
        )
        self.inception1 = Inception(64, 32, 32, 64, 8, 16, 16)   # -> 128 ch
        self.inception2 = Inception(128, 64, 48, 96, 16, 32, 32)  # -> 224 ch
        self.pool = nn.MaxPool2d(2)                      # -> 7x7
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                     # global average pooling
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(224, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.pool(x)
        return self.head(x)


def main():
    args = mc.build_argparser("Mini-GoogLeNet (Inception) on MNIST").parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = MiniGoogLeNet()

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name="GoogLeNet-Inception",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
