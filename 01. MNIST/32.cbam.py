"""
32. CBAM — Convolutional Block Attention Module  (Woo, Park, Lee & Kweon, 2018)
===============================================================================

SE-ResNet (`21.se-resnet.py`) reweights *channels*. CBAM argues attention
should also be *spatial* — "what" to emphasize and "where" — so it chains two
lightweight modules inside each block:

  * **Channel attention.** Squeeze the feature map with *both* average- and
    max-pooling, push each through a shared MLP, add, sigmoid -> a per-channel
    gate (like SE, but using max-pool too).
  * **Spatial attention.** Pool across the *channel* axis with avg and max to get
    two HxW maps, concatenate, run a 7x7 conv, sigmoid -> a per-location gate.

Both gates are applied in sequence. CBAM drops into any convnet for ~free; here
it sits in a BasicBlock ResNet so you can compare it directly with plain ResNet
(`28.resnet-basic.py`) and channel-only SE-ResNet.

    --variant 18   stages = (2, 2, 2, 2)      --variant 34   stages = (3, 4, 6, 3)

Run:
    python "32.cbam.py" --variant 18 --epochs 5
    python "32.cbam.py" --variant 34 --limit 2000
"""

import os

import torch
import torch.nn as nn

import mnist_common as mc

STAGES = {"18": (2, 2, 2, 2), "34": (3, 4, 6, 3)}


class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        hidden = max(1, ch // reduction)
        self.mlp = nn.Sequential(nn.Linear(ch, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, ch))

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.mlp(x.mean(dim=(2, 3)))                       # global average pool
        mx = self.mlp(x.amax(dim=(2, 3)))                        # global max pool
        gate = torch.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * gate


class SpatialAttention(nn.Module):
    def __init__(self, kernel=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)                        # pool across channels
        mx = x.amax(dim=1, keepdim=True)
        gate = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * gate


class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.channel = ChannelAttention(ch)
        self.spatial = SpatialAttention()

    def forward(self, x):
        return self.spatial(self.channel(x))                    # channel then spatial


class CBAMBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.cbam = CBAM(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.cbam(self.bn2(self.conv2(out)))              # attention before the residual add
        return self.relu(out + identity)


class CBAMResNet(nn.Module):
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
                                  nn.Linear(base * 8, num_classes))

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.in_ch != planes:
            downsample = nn.Sequential(nn.Conv2d(self.in_ch, planes, 1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes))
        blk = [CBAMBlock(self.in_ch, planes, stride, downsample)]
        self.in_ch = planes
        for _ in range(1, blocks):
            blk.append(CBAMBlock(self.in_ch, planes))
        return nn.Sequential(*blk)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.head(x)


def main():
    p = mc.build_argparser("CBAM-ResNet on MNIST")
    p.add_argument("--variant", choices=list(STAGES), default="18")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = CBAMResNet(STAGES[args.variant])

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name=f"CBAM-ResNet-{args.variant}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
