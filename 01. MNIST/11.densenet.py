"""
11. DenseNet-121 / 169 / 201  (Huang, Liu, van der Maaten & Weinberger, 2017)
=============================================================================

DenseNet takes the residual idea to its limit: inside a dense block, every layer
receives the feature maps of *all* preceding layers (concatenation, not
addition). This maximizes feature reuse, strengthens gradient flow, and is
remarkably parameter-efficient. Between blocks, "transition" layers compress and
downsample.

The three variants differ only in the number of layers per dense block:

    --variant 121   blocks = (6, 12, 24, 16)
    --variant 169   blocks = (6, 12, 32, 32)
    --variant 201   blocks = (6, 12, 48, 32)

This is the DenseNet-BC design (Bottleneck layers + Compression 0.5), faithful
to the paper. For MNIST the growth rate is scaled to 12 (vs 32) and the stem is a
plain 3x3 conv so a 28x28 digit survives.

Run:
    python "11.densenet.py" --variant 121 --epochs 5
    python "11.densenet.py" --variant 201 --limit 2000
"""

import os

import torch
import torch.nn as nn

import mnist_common as mc

BLOCKS = {"121": (6, 12, 24, 16), "169": (6, 12, 32, 32), "201": (6, 12, 48, 32)}


class DenseLayer(nn.Module):
    """BN-ReLU-1x1 (bottleneck, 4*k) -> BN-ReLU-3x3 (k); output concatenated."""

    def __init__(self, in_ch, growth, bn_size=4):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, bn_size * growth, 1, bias=False),
            nn.BatchNorm2d(bn_size * growth), nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth, growth, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return torch.cat([x, self.block(x)], dim=1)


class Transition(nn.Sequential):
    """Compress channels (x compression) and halve spatial size."""

    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.AvgPool2d(2),
        )


class DenseNet(nn.Module):
    def __init__(self, block_config, growth=12, compression=0.5, num_classes=10):
        super().__init__()
        ch = 2 * growth
        self.stem = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
        )
        layers = []
        for i, n in enumerate(block_config):
            for _ in range(n):
                layers.append(DenseLayer(ch, growth))
                ch += growth
            if i != len(block_config) - 1:                  # transition after all but last
                out_ch = int(ch * compression)
                layers.append(Transition(ch, out_ch))
                ch = out_ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


def main():
    p = mc.build_argparser("DenseNet-BC on MNIST")
    p.add_argument("--variant", choices=list(BLOCKS), default="121")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = DenseNet(BLOCKS[args.variant])

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name=f"DenseNet-{args.variant}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
