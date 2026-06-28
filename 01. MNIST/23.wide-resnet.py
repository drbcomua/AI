"""
23. Wide ResNet (WRN)  (Zagoruyko & Komodakis, 2016 — "Wide Residual Networks")
===============================================================================

Wide ResNet challenges the "deeper is better" orthodoxy: a *shallower but wider*
residual network trains faster and is at least as accurate. Two design choices:

  * **Width multiplier k.** Multiply the channel count in every residual block by
    k (k=10 is common). Most of WRN's gains come from this.
  * **Pre-activation BasicBlock with dropout.** Each block is
    BN -> ReLU -> 3x3 conv -> BN -> ReLU -> Dropout -> 3x3 conv, plus the
    residual (the "pre-activation" ordering from ResNet-v2).

Naming is `WRN-d-k` for depth d and width k. Depth must be 6n+4:

    --variant 16-8    (n=2, k=8)
    --variant 28-10   (n=4, k=10)   the classic CIFAR champion
    --variant 40-4    (n=6, k=4)

MNIST-scaled stem; three residual groups with global average pooling.

Run:
    python "23.wide-resnet.py" --variant 28-10 --epochs 5
    python "23.wide-resnet.py" --variant 16-8 --limit 2000
"""

import os

import torch.nn as nn

import mnist_common as mc

# variant -> (depth, widen_factor)
VARIANTS = {"16-8": (16, 8), "28-10": (28, 10), "40-4": (40, 4)}


class BasicBlock(nn.Module):
    """Pre-activation residual block: BN-ReLU-conv -> BN-ReLU-dropout-conv (+ skip)."""

    def __init__(self, in_ch, out_ch, stride, dropout=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.equal = (in_ch == out_ch and stride == 1)
        self.shortcut = None if self.equal else \
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.relu(self.bn1(x))
        shortcut = x if self.equal else self.shortcut(out)   # v2: shortcut sees pre-activated x
        out = self.conv1(out)
        out = self.drop(self.relu(self.bn2(out)))
        out = self.conv2(out)
        return out + shortcut


class WideResNet(nn.Module):
    def __init__(self, depth, k, num_classes=10, dropout=0.3):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth must be 6n+4"
        n = (depth - 4) // 6
        widths = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = nn.Conv2d(1, widths[0], 3, padding=1, bias=False)
        self.group1 = self._group(widths[0], widths[1], n, stride=1, dropout=dropout)
        self.group2 = self._group(widths[1], widths[2], n, stride=2, dropout=dropout)
        self.group3 = self._group(widths[2], widths[3], n, stride=2, dropout=dropout)
        self.bn = nn.BatchNorm2d(widths[3])
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(widths[3], num_classes))

    @staticmethod
    def _group(in_ch, out_ch, n, stride, dropout):
        blocks = [BasicBlock(in_ch, out_ch, stride, dropout)]
        for _ in range(1, n):
            blocks.append(BasicBlock(out_ch, out_ch, 1, dropout))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.group1(x); x = self.group2(x); x = self.group3(x)
        x = self.relu(self.bn(x))
        return self.head(x)


def main():
    p = mc.build_argparser("Wide ResNet on MNIST")
    p.add_argument("--variant", choices=list(VARIANTS), default="28-10")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    depth, k = VARIANTS[args.variant]
    model = WideResNet(depth, k)

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name=f"WRN-{args.variant}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
