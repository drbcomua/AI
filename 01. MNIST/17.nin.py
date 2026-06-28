"""
17. Network-in-Network (NiN)  (Lin, Chen & Yan, 2014)
=====================================================

NiN introduced two ideas that the whole field then absorbed:

  * **1x1 convolutions ("mlpconv").** Instead of a single linear filter per
    receptive field, stack 1x1 convs after a normal conv so each spatial
    location is processed by a tiny per-pixel MLP — a far richer, non-linear
    feature extractor. Every architecture after this (Inception, ResNet
    bottlenecks, MobileNet pointwise convs...) leans on 1x1 convs.

  * **Global Average Pooling (GAP) instead of fully-connected layers.** The last
    mlpconv emits one feature map per class; average each to a single number and
    softmax. No giant FC layer, far fewer parameters, and built-in spatial
    invariance. GAP is now standard in essentially every convnet in this folder.

Faithful CIFAR-style NiN, adapted to a 1x28x28 input.

Run:
    python "17.nin.py" --epochs 5
    python "17.nin.py" --limit 2000
"""

import os

import torch.nn as nn

import mnist_common as mc


def mlpconv(in_ch, out_ch, mid_ch, kernel, padding):
    """A conv followed by two 1x1 convs — NiN's 'micro-network' per patch."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, mid_ch, 1), nn.ReLU(inplace=True),
        nn.Conv2d(mid_ch, mid_ch, 1), nn.ReLU(inplace=True),
    )


class NiN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            mlpconv(1, 96, 96, kernel=5, padding=2),
            nn.MaxPool2d(3, stride=2, padding=1), nn.Dropout(0.5),     # 28 -> 14
            mlpconv(96, 128, 128, kernel=5, padding=2),
            nn.MaxPool2d(3, stride=2, padding=1), nn.Dropout(0.5),     # 14 -> 7
            # Final mlpconv emits exactly num_classes feature maps...
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1), nn.ReLU(inplace=True),
        )
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())  # ...then GAP -> logits

    def forward(self, x):
        return self.gap(self.features(x))


def main():
    args = mc.build_argparser("Network-in-Network on MNIST").parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = NiN()

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name="NiN",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
