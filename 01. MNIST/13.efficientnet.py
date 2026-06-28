"""
13. EfficientNet-B0 … B7  (Tan & Le, 2019 — "Rethinking Model Scaling")
======================================================================

EfficientNet's insight is *compound scaling*: instead of growing depth, width,
or resolution alone, scale all three together by powers of fixed coefficients.
Starting from the B0 baseline (itself found by NAS), each step up the ladder
multiplies width by ~1.1^phi and depth by ~1.2^phi:

    --variant b0  (width 1.0, depth 1.0)      ...      --variant b7  (2.0, 3.1)

The building block is **MBConv** — MobileNetV2's inverted residual plus
squeeze-and-excitation and stochastic-depth-friendly residuals — exactly as in
the paper. The base channel/repeat table and the official width/depth
coefficients are used verbatim; `round_filters`/`round_repeats` apply the
scaling. For MNIST the input resolution is fixed at 28x28 and stem/stage strides
are kept gentle so the digit isn't downsampled away.

Run:
    python "13.efficientnet.py" --variant b0 --epochs 5
    python "13.efficientnet.py" --variant b4 --limit 2000
"""

import math
import os

import torch
import torch.nn as nn

import mnist_common as mc

# variant -> (width_mult, depth_mult).  (Resolution is irrelevant at 28x28.)
COEFFS = {
    "b0": (1.0, 1.0), "b1": (1.0, 1.1), "b2": (1.1, 1.2), "b3": (1.2, 1.4),
    "b4": (1.4, 1.8), "b5": (1.6, 2.2), "b6": (1.8, 2.6), "b7": (2.0, 3.1),
}

# Base (B0) stage table: (expand, channels, repeats, stride, kernel)
BASE_CFG = [
    (1, 16, 1, 1, 3),
    (6, 24, 2, 1, 3),    # stride relaxed 2->1 for tiny inputs
    (6, 40, 2, 2, 5),
    (6, 80, 3, 2, 3),
    (6, 112, 3, 1, 5),
    (6, 192, 4, 2, 5),
    (6, 320, 1, 1, 3),
]


def round_filters(c, width, divisor=8):
    c *= width
    new_c = max(divisor, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:                      # never round down more than 10%
        new_c += divisor
    return int(new_c)


def round_repeats(r, depth):
    return int(math.ceil(depth * r))


class SqueezeExcite(nn.Module):
    def __init__(self, ch, se_ch):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, se_ch, 1), nn.SiLU(inplace=True),
            nn.Conv2d(se_ch, ch, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expand, stride, kernel, se_ratio=0.25):
        super().__init__()
        hidden = in_ch * expand
        self.use_res = stride == 1 and in_ch == out_ch
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(in_ch, hidden, 1, bias=False),
                       nn.BatchNorm2d(hidden), nn.SiLU(inplace=True)]
        layers += [nn.Conv2d(hidden, hidden, kernel, stride=stride, padding=kernel // 2,
                             groups=hidden, bias=False),
                   nn.BatchNorm2d(hidden), nn.SiLU(inplace=True),
                   SqueezeExcite(hidden, max(1, int(in_ch * se_ratio))),
                   nn.Conv2d(hidden, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class EfficientNet(nn.Module):
    def __init__(self, width, depth, num_classes=10, dropout=0.2):
        super().__init__()
        stem_ch = round_filters(32, width)
        self.stem = nn.Sequential(nn.Conv2d(1, stem_ch, 3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(stem_ch), nn.SiLU(inplace=True))
        layers, in_ch = [], stem_ch
        for expand, ch, repeats, stride, kernel in BASE_CFG:
            out_ch = round_filters(ch, width)
            for i in range(round_repeats(repeats, depth)):
                layers.append(MBConv(in_ch, out_ch, expand, stride if i == 0 else 1, kernel))
                in_ch = out_ch
        self.features = nn.Sequential(*layers)
        head_ch = round_filters(1280, width)
        self.tail = nn.Sequential(nn.Conv2d(in_ch, head_ch, 1, bias=False),
                                  nn.BatchNorm2d(head_ch), nn.SiLU(inplace=True))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(head_ch, num_classes))

    def forward(self, x):
        return self.head(self.tail(self.features(self.stem(x))))


def main():
    p = mc.build_argparser("EfficientNet on MNIST")
    p.add_argument("--variant", choices=list(COEFFS), default="b0")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    width, depth = COEFFS[args.variant]
    model = EfficientNet(width, depth)

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name=f"EfficientNet-{args.variant.upper()}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
