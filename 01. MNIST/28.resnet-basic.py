"""
28. ResNet-18 / 34  (He, Zhang, Ren & Sun, 2015 — the BasicBlock ResNets)
=========================================================================

`10.resnet.py` covers the deep "bottleneck" ResNets (50/101/152). The two
shallow members of the family — ResNet-18 and ResNet-34 — use the simpler
**BasicBlock** instead: just two 3x3 convolutions plus the residual shortcut, no
1x1 bottleneck. These are the ResNets people actually reach for on small problems.

    --variant 18   stages = (2, 2, 2, 2)
    --variant 34   stages = (3, 4, 6, 3)

To make depth comparable across the whole ResNet family in this folder, the base
width is kept at 16 (the same scaling used in `10.resnet.py`) and the stem is the
CIFAR/MNIST-style 3x3. So you can line up 18 -> 34 -> 50 -> 101 -> 152 and watch
what depth (and the bottleneck design) actually buys on MNIST: very little, which
is the honest lesson — the dataset is too easy to reward it.

Run:
    python "28.resnet-basic.py" --variant 18 --epochs 5
    python "28.resnet-basic.py" --variant 34 --limit 2000
"""

import os

import torch.nn as nn

import mnist_common as mc

STAGES = {"18": (2, 2, 2, 2), "34": (3, 4, 6, 3)}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet(nn.Module):
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
                                  nn.Linear(base * 8 * BasicBlock.expansion, num_classes))

    def _make_layer(self, planes, blocks, stride):
        out_ch = planes * BasicBlock.expansion
        downsample = None
        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(nn.Conv2d(self.in_ch, out_ch, 1, stride=stride, bias=False),
                                       nn.BatchNorm2d(out_ch))
        blk = [BasicBlock(self.in_ch, planes, stride, downsample)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            blk.append(BasicBlock(self.in_ch, planes))
        return nn.Sequential(*blk)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.head(x)


def main():
    p = mc.build_argparser("ResNet-18/34 (BasicBlock) on MNIST")
    p.add_argument("--variant", choices=list(STAGES), default="18")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = ResNet(STAGES[args.variant])

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name=f"ResNet-{args.variant}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
