"""
21. SE-ResNet  (Hu, Shen & Sun, 2018 — "Squeeze-and-Excitation Networks", ILSVRC-2017 winner)
=============================================================================================

Squeeze-and-Excitation adds cheap *channel attention* to any convnet. After a
block computes its feature maps, an SE unit:

    squeeze   : global-average-pool each channel to one number  (global context)
    excite    : a tiny two-layer MLP (bottleneck + sigmoid) turns that into a
                per-channel gate in [0, 1]
    scale     : multiply every channel by its gate

So the network learns, per input, which feature channels to amplify or suppress.
Dropping SE into ResNet's bottleneck (this file) gave SENet the 2017 ImageNet
win for ~1% extra compute.

This is the bottleneck ResNet from `10.resnet.py` with an SE module added to each
block; variants differ only in depth:

    --variant 50  (3,4,6,3)   --variant 101  (3,4,23,3)   --variant 152  (3,8,36,3)

Run:
    python "21.se-resnet.py" --variant 50 --epochs 5
    python "21.se-resnet.py" --variant 152 --limit 2000
"""

import os

import torch.nn as nn

import mnist_common as mc

STAGES = {"50": (3, 4, 6, 3), "101": (3, 4, 23, 3), "152": (3, 8, 36, 3)}


class SEModule(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(1, ch // reduction), 1), nn.ReLU(inplace=True),
            nn.Conv2d(max(1, ch // reduction), ch, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, planes, stride=1, downsample=None):
        super().__init__()
        out_ch = planes * self.expansion
        self.conv1 = nn.Conv2d(in_ch, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.se = SEModule(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.se(self.bn3(self.conv3(out)))     # SE recalibration before the residual add
        return self.relu(out + identity)


class SEResNet(nn.Module):
    def __init__(self, layers, base=16, num_classes=10):
        super().__init__()
        self.in_ch = base
        self.stem = nn.Sequential(nn.Conv2d(1, base, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(base), nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(base, layers[0], 1)
        self.layer2 = self._make_layer(base * 2, layers[1], 2)
        self.layer3 = self._make_layer(base * 4, layers[2], 2)
        self.layer4 = self._make_layer(base * 8, layers[3], 2)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(base * 8 * SEBottleneck.expansion, num_classes))

    def _make_layer(self, planes, blocks, stride):
        out_ch = planes * SEBottleneck.expansion
        downsample = None
        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(nn.Conv2d(self.in_ch, out_ch, 1, stride=stride, bias=False),
                                       nn.BatchNorm2d(out_ch))
        blk = [SEBottleneck(self.in_ch, planes, stride, downsample)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            blk.append(SEBottleneck(self.in_ch, planes))
        return nn.Sequential(*blk)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.head(x)


def main():
    p = mc.build_argparser("SE-ResNet on MNIST")
    p.add_argument("--variant", choices=list(STAGES), default="50")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = SEResNet(STAGES[args.variant])

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name=f"SE-ResNet-{args.variant}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
