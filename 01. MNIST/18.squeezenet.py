"""
18. SqueezeNet  (Iandola, Han, Moskewicz, Ashraf, Dally & Keutzer, 2016)
========================================================================

SqueezeNet reaches AlexNet-level accuracy with ~50x fewer parameters by being
ruthless about parameter budget. Its **Fire module** is the trick:

    squeeze : a 1x1 conv that *shrinks* the channel count (cheap)
    expand  : a 1x1 conv and a 3x3 conv run in parallel on the squeezed tensor,
              concatenated back together

Most channels never touch an expensive 3x3 filter, and there are no fully
connected layers at all (global average pooling produces the logits).

    --variant 1.0   original layout (pool after conv1, fire2-4, fire5-8)
    --variant 1.1   pools moved earlier -> ~2.4x less compute, same accuracy

Faithful Fire-module design, MNIST-scaled (stride-1 3x3 stem so a 28x28 digit
isn't thrown away).

Run:
    python "18.squeezenet.py" --variant 1.0 --epochs 5
    python "18.squeezenet.py" --variant 1.1 --limit 2000
"""

import os

import torch
import torch.nn as nn

import mnist_common as mc


class Fire(nn.Module):
    def __init__(self, in_ch, squeeze, e1, e3):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(in_ch, squeeze, 1), nn.ReLU(inplace=True))
        self.expand1 = nn.Sequential(nn.Conv2d(squeeze, e1, 1), nn.ReLU(inplace=True))
        self.expand3 = nn.Sequential(nn.Conv2d(squeeze, e3, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([self.expand1(x), self.expand3(x)], dim=1)


class SqueezeNet(nn.Module):
    def __init__(self, variant="1.0", num_classes=10):
        super().__init__()
        pool = lambda: nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)
        if variant == "1.0":
            self.features = nn.Sequential(
                nn.Conv2d(1, 96, 3, padding=1), nn.ReLU(inplace=True), pool(),   # 28->14
                Fire(96, 16, 64, 64), Fire(128, 16, 64, 64), Fire(128, 32, 128, 128), pool(),  # ->7
                Fire(256, 32, 128, 128), Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192), Fire(384, 64, 256, 256), pool(),        # ->4
                Fire(512, 64, 256, 256),
            )
            final_in = 512
        else:  # 1.1 — pools earlier, cheaper
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True), pool(),   # 28->14
                Fire(64, 16, 64, 64), Fire(128, 16, 64, 64), pool(),             # ->7
                Fire(128, 32, 128, 128), Fire(256, 32, 128, 128), pool(),        # ->4
                Fire(256, 48, 192, 192), Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256), Fire(512, 64, 256, 256),
            )
            final_in = 512
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(final_in, num_classes, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    p = mc.build_argparser("SqueezeNet on MNIST")
    p.add_argument("--variant", choices=["1.0", "1.1"], default="1.0")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = SqueezeNet(args.variant)

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name=f"SqueezeNet-{args.variant}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
