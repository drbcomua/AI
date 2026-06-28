"""
22. ResNeXt  (Xie, Girshick, Dollar, Tu & He, 2017 — "Aggregated Residual Transformations")
===========================================================================================

ResNeXt adds a third axis to the depth/width design space: **cardinality**, the
number of parallel transformation paths inside a block. Instead of one wide 3x3
convolution, a ResNeXt bottleneck splits the work into `groups` independent
branches (implemented efficiently as a single *grouped* convolution) and sums
them. The paper's headline result: raising cardinality improves accuracy more
effectively than making the network deeper or wider.

It reuses ResNet's bottleneck skeleton; only the middle 3x3 conv changes — it
becomes grouped, with the width set by `cardinality x width_per_group`:

    --variant 50   ResNeXt-50  (32 x 4d)   stages (3,4,6,3)
    --variant 101  ResNeXt-101 (32 x 8d)   stages (3,4,23,3)

MNIST-scaled (CIFAR-style stem, scaled base width) but cardinality is kept at the
paper's 32.

Run:
    python "22.resnext.py" --variant 50 --epochs 5
    python "22.resnext.py" --variant 101 --limit 2000
"""

import os

import torch.nn as nn

import mnist_common as mc

# variant -> (stages, width_per_group)
CONFIGS = {"50": ((3, 4, 6, 3), 4), "101": ((3, 4, 23, 3), 8)}
CARDINALITY = 32


class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, planes, stride, width, downsample=None):
        super().__init__()
        # `width` channels for the grouped conv (cardinality * width_per_group, scaled by planes).
        self.conv1 = nn.Conv2d(in_ch, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=1,
                               groups=CARDINALITY, bias=False)        # the aggregated paths
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class ResNeXt(nn.Module):
    def __init__(self, layers, width_per_group, base=32, num_classes=10):
        super().__init__()
        self.base = base
        self.width_per_group = width_per_group
        self.in_ch = base
        self.stem = nn.Sequential(nn.Conv2d(1, base, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(base), nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(base, layers[0], 1)
        self.layer2 = self._make_layer(base * 2, layers[1], 2)
        self.layer3 = self._make_layer(base * 4, layers[2], 2)
        self.layer4 = self._make_layer(base * 8, layers[3], 2)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(base * 8 * ResNeXtBottleneck.expansion, num_classes))

    def _make_layer(self, planes, blocks, stride):
        out_ch = planes * ResNeXtBottleneck.expansion
        # grouped-conv width: scales with planes, keeps groups divisible by CARDINALITY
        width = CARDINALITY * self.width_per_group * (planes // self.base)
        downsample = None
        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(nn.Conv2d(self.in_ch, out_ch, 1, stride=stride, bias=False),
                                       nn.BatchNorm2d(out_ch))
        blk = [ResNeXtBottleneck(self.in_ch, planes, stride, width, downsample)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            blk.append(ResNeXtBottleneck(self.in_ch, planes, 1, width))
        return nn.Sequential(*blk)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.head(x)


def main():
    p = mc.build_argparser("ResNeXt (32 x Nd) on MNIST")
    p.add_argument("--variant", choices=list(CONFIGS), default="50")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    layers, wpg = CONFIGS[args.variant]
    model = ResNeXt(layers, wpg)

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name=f"ResNeXt-{args.variant}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
