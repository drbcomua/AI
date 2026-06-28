"""
12. MobileNet V1 / V2 / V3  (Howard et al. 2017 / Sandler et al. 2018 / Howard et al. 2019)
===========================================================================================

The MobileNet line chases maximum accuracy per FLOP for phones — and each
version introduces one big idea:

    --variant v1   Depthwise-separable convolution: split a normal conv into a
                   per-channel 3x3 (depthwise) + a 1x1 (pointwise) mixing conv.
                   ~8-9x cheaper for a tiny accuracy cost.

    --variant v2   Inverted residual + linear bottleneck: expand channels with a
                   1x1, do the cheap depthwise in the *wide* space, project back
                   down with a *linear* 1x1, and add a residual across the
                   narrow ends.

    --variant v3   Adds squeeze-and-excitation channel attention and the
                   hard-swish activation on top of v2's blocks (architecture
                   found by neural-architecture search).

All three are MNIST-scaled (fewer blocks, smaller widths, strides tuned for
28x28) but keep each version's signature block intact.

Run:
    python "12.mobilenet.py" --variant v2 --epochs 5
    python "12.mobilenet.py" --variant v3 --limit 2000
"""

import os

import torch
import torch.nn as nn

import mnist_common as mc


# --------------------------------------------------------------------------- #
# MobileNetV1 — depthwise separable convolutions
# --------------------------------------------------------------------------- #
def dw_separable(in_ch, out_ch, stride):
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        # (out_ch, stride)
        cfg = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2),
               (512, 1), (512, 1), (1024, 2), (1024, 1)]
        layers, in_ch = [], 32
        for out_ch, s in cfg:
            layers.append(dw_separable(in_ch, out_ch, s))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(in_ch, num_classes))

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


# --------------------------------------------------------------------------- #
# MobileNetV2 — inverted residual + linear bottleneck
# --------------------------------------------------------------------------- #
class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand):
        super().__init__()
        hidden = in_ch * expand
        self.use_res = stride == 1 and in_ch == out_ch
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(in_ch, hidden, 1, bias=False),
                       nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=False),             # linear (no activation)
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
        )
        # (expand, out_ch, repeats, stride)
        cfg = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2),
               (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 1), (6, 320, 1, 1)]
        layers, in_ch = [], 32
        for t, c, n, s in cfg:
            for i in range(n):
                layers.append(InvertedResidual(in_ch, c, s if i == 0 else 1, t))
                in_ch = c
        self.features = nn.Sequential(*layers)
        last = 1280
        self.tail = nn.Sequential(nn.Conv2d(in_ch, last, 1, bias=False),
                                  nn.BatchNorm2d(last), nn.ReLU6(inplace=True))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.2), nn.Linear(last, num_classes))

    def forward(self, x):
        return self.head(self.tail(self.features(self.stem(x))))


# --------------------------------------------------------------------------- #
# MobileNetV3 — SE attention + hard-swish
# --------------------------------------------------------------------------- #
class SqueezeExcite(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1), nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x):
        return x * self.fc(x)


class BneckV3(nn.Module):
    def __init__(self, in_ch, hidden, out_ch, kernel, stride, use_se, act):
        super().__init__()
        self.use_res = stride == 1 and in_ch == out_ch
        Act = nn.Hardswish if act == "hs" else nn.ReLU
        layers = [nn.Conv2d(in_ch, hidden, 1, bias=False),
                  nn.BatchNorm2d(hidden), Act(inplace=True)]
        layers += [nn.Conv2d(hidden, hidden, kernel, stride=stride,
                             padding=kernel // 2, groups=hidden, bias=False),
                   nn.BatchNorm2d(hidden), Act(inplace=True)]
        if use_se:
            layers.append(SqueezeExcite(hidden))
        layers += [nn.Conv2d(hidden, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 16, 3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(16), nn.Hardswish(inplace=True))
        # (in, hidden, out, kernel, stride, SE, activation)  -- MobileNetV3-Small style
        cfg = [
            (16, 16, 16, 3, 2, True, "re"),
            (16, 72, 24, 3, 2, False, "re"),
            (24, 88, 24, 3, 1, False, "re"),
            (24, 96, 40, 5, 2, True, "hs"),
            (40, 240, 40, 5, 1, True, "hs"),
            (40, 120, 48, 5, 1, True, "hs"),
            (48, 144, 48, 5, 1, True, "hs"),
            (48, 288, 96, 5, 1, True, "hs"),
            (96, 576, 96, 5, 1, True, "hs"),
        ]
        self.features = nn.Sequential(*[BneckV3(*c) for c in cfg])
        self.tail = nn.Sequential(nn.Conv2d(96, 576, 1, bias=False),
                                  nn.BatchNorm2d(576), nn.Hardswish(inplace=True))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(576, 1024), nn.Hardswish(inplace=True), nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.head(self.tail(self.features(self.stem(x))))


BUILDERS = {"v1": MobileNetV1, "v2": MobileNetV2, "v3": MobileNetV3}


def main():
    p = mc.build_argparser("MobileNet on MNIST")
    p.add_argument("--variant", choices=list(BUILDERS), default="v2")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = BUILDERS[args.variant]()

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name=f"MobileNet{args.variant.upper()}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
