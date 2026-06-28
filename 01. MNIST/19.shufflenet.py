"""
19. ShuffleNet V2  (Ma, Zhang, Zheng & Sun, 2018 — "Practical Guidelines for Efficient CNN Design")
===================================================================================================

ShuffleNet V2 is MobileNet's efficiency rival, built around two ideas:

  * **Channel split + channel shuffle.** Group convolutions are cheap but trap
    information inside each group. ShuffleNet splits channels in two, sends one
    half through a lightweight branch (1x1 -> depthwise 3x3 -> 1x1), keeps the
    other as an identity, concatenates, then *shuffles* channels so the next
    layer's groups see a mix from both halves.

  * **Hardware-aware design guidelines** — equal input/output channel widths,
    fewer branches, etc. — rather than minimizing FLOPs alone.

Width is the knob (more channels = more accurate, slower):

    --variant 0.5x | 1.0x | 1.5x | 2.0x

Faithful V2 units (basic + spatial-downsampling), MNIST-scaled stem.

Run:
    python "19.shufflenet.py" --variant 1.0x --epochs 5
    python "19.shufflenet.py" --variant 0.5x --limit 2000
"""

import os

import torch
import torch.nn as nn

import mnist_common as mc

# variant -> stage output channels [stage2, stage3, stage4, conv5]
WIDTHS = {
    "0.5x": [48, 96, 192, 1024],
    "1.0x": [116, 232, 464, 1024],
    "1.5x": [176, 352, 704, 1024],
    "2.0x": [244, 488, 976, 2048],
}


def channel_shuffle(x, groups=2):
    b, c, h, w = x.shape
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class ShuffleUnit(nn.Module):
    """stride=1: split + identity branch. stride=2: both branches downsample."""

    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.stride = stride
        branch_ch = out_ch // 2
        if stride == 1:
            in_branch = in_ch // 2          # channel split
            self.branch1 = nn.Identity()
        else:
            in_branch = in_ch
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch, branch_ch, 1, bias=False),
                nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
            )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_branch, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
            nn.Conv2d(branch_ch, branch_ch, 3, stride=stride, padding=1,
                      groups=branch_ch, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.Conv2d(branch_ch, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.branch2(x2)], dim=1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, widths, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(           # MNIST-scaled: stride 1, no early maxpool
            nn.Conv2d(1, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24), nn.ReLU(inplace=True),
        )
        repeats = [4, 8, 4]
        in_ch = 24
        stages = []
        for out_ch, n in zip(widths[:3], repeats):
            stages.append(ShuffleUnit(in_ch, out_ch, stride=2))
            for _ in range(n - 1):
                stages.append(ShuffleUnit(out_ch, out_ch, stride=1))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.conv5 = nn.Sequential(nn.Conv2d(in_ch, widths[3], 1, bias=False),
                                   nn.BatchNorm2d(widths[3]), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(widths[3], num_classes))

    def forward(self, x):
        return self.head(self.conv5(self.stages(self.stem(x))))


def main():
    p = mc.build_argparser("ShuffleNet V2 on MNIST")
    p.add_argument("--variant", choices=list(WIDTHS), default="1.0x")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = ShuffleNetV2(WIDTHS[args.variant])

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name=f"ShuffleNetV2-{args.variant}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
