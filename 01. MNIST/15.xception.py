"""
15. Xception  (Chollet, 2017 — "Deep Learning with Depthwise Separable Convolutions")
=====================================================================================

Xception ("Extreme Inception") reinterprets an Inception module as the limit case
where cross-channel correlations and spatial correlations are mapped *completely
separately* — i.e. the whole network is built from **depthwise separable
convolutions** (a per-channel spatial conv followed by a 1x1 pointwise conv),
wrapped in residual connections.

It is organized into three flows, kept here but MNIST-scaled:

    Entry flow   : a small conv stem + residual separable-conv blocks (downsample)
    Middle flow  : several identical residual separable-conv blocks (repeated)
    Exit flow    : a final residual block + separable convs -> global average pool

Run:
    python "15.xception.py" --epochs 5
    python "15.xception.py" --limit 2000
"""

import os

import torch.nn as nn

import mnist_common as mc


class SeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, padding=kernel // 2, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.pw(self.dw(x)))


class XBlock(nn.Module):
    """Residual block of two separable convs; `downsample` halves the spatial size."""

    def __init__(self, in_ch, out_ch, downsample=False, first_relu=True):
        super().__init__()
        self.first_relu = first_relu
        self.sep1 = SeparableConv(in_ch, out_ch)
        self.sep2 = SeparableConv(out_ch, out_ch)
        self.relu = nn.ReLU(inplace=False)   # x is shared with the residual skip — don't modify in place
        self.pool = nn.MaxPool2d(3, 2, 1) if downsample else nn.Identity()
        if downsample or in_ch != out_ch:
            stride = 2 if downsample else 1
            self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                                      nn.BatchNorm2d(out_ch))
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.sep1(self.relu(x)) if self.first_relu else self.sep1(x)
        out = self.sep2(self.relu(out))
        out = self.pool(out)
        return out + residual


class Xception(nn.Module):
    def __init__(self, num_classes=10, middle_blocks=4):
        super().__init__()
        # Entry flow
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.entry = nn.Sequential(
            XBlock(64, 128, downsample=True, first_relu=False),   # 28 -> 14
            XBlock(128, 256, downsample=True),                    # 14 -> 7
        )
        # Middle flow (repeated identical residual blocks, no downsample)
        self.middle = nn.Sequential(*[XBlock(256, 256) for _ in range(middle_blocks)])
        # Exit flow
        self.exit = XBlock(256, 512, downsample=True)             # 7 -> 4
        self.tail = nn.Sequential(
            SeparableConv(512, 768), nn.ReLU(inplace=True),
            SeparableConv(768, 1024), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(1024, num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.entry(x)
        x = self.middle(x)
        x = self.exit(x)
        x = self.tail(x)
        return self.head(x)


def main():
    args = mc.build_argparser("Xception (MNIST-scaled) on MNIST").parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = Xception()

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name="Xception",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
