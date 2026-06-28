"""
10. ResNet-50 / 101 / 152  (He, Zhang, Ren & Sun, 2015 — "Deep Residual Learning")
==================================================================================

ResNet's residual shortcut, ``y = F(x) + x``, lets gradients flow straight
through, so networks could finally go *very* deep without degrading. These three
variants are the deep "bottleneck" family and differ only in how many blocks
each stage gets:

    --variant resnet50    stages = (3, 4,  6, 3)
    --variant resnet101   stages = (3, 4, 23, 3)
    --variant resnet152   stages = (3, 8, 36, 3)

The **Bottleneck** block (1x1 reduce -> 3x3 -> 1x1 expand, expansion 4) is
faithful to the paper. Two MNIST adaptations: a CIFAR-style 3x3 stem (the
original 7x7/stride-2 + maxpool would throw away a 28x28 digit immediately), and
the channel widths are scaled to base 16 (vs 64) so even ResNet-152 trains
quickly here. This is the place to ask "do more layers work?" — with residual
connections the answer is finally yes, though MNIST is far too easy to reward
the extra depth much.

Run:
    python "10.resnet.py" --variant resnet50 --epochs 5
    python "10.resnet.py" --variant resnet152 --limit 2000
"""

import os

import torch.nn as nn

import mnist_common as mc

STAGES = {
    "resnet50": (3, 4, 6, 3),
    "resnet101": (3, 4, 23, 3),
    "resnet152": (3, 8, 36, 3),
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class ResNet(nn.Module):
    def __init__(self, layers, base=16, num_classes=10):
        super().__init__()
        self.in_ch = base
        # CIFAR/MNIST-style stem: keep full 28x28 resolution.
        self.stem = nn.Sequential(
            nn.Conv2d(1, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(base, layers[0], stride=1)
        self.layer2 = self._make_layer(base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base * 8, layers[3], stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(base * 8 * Bottleneck.expansion, num_classes),
        )

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        out_ch = planes * Bottleneck.expansion
        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        blocks_list = [Bottleneck(self.in_ch, planes, stride, downsample)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            blocks_list.append(Bottleneck(self.in_ch, planes))
        return nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.head(x)


def main():
    p = mc.build_argparser("ResNet (bottleneck) on MNIST")
    p.add_argument("--variant", choices=list(STAGES), default="resnet50")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = ResNet(STAGES[args.variant])

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name=f"ResNet-{args.variant[6:]}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
