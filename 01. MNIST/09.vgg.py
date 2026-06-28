"""
09. VGG-16 / VGG-19  (Simonyan & Zisserman, 2014 — "Very Deep ... for Large-Scale Image Recognition")
=====================================================================================================

VGG's thesis: depth via *uniform, tiny* filters. Stack many 3x3 convolutions
(two 3x3s see a 5x5 region, three see 7x7 — with fewer parameters and more
non-linearity), double the channel count after every 2x2 max-pool, and finish
with fully-connected layers.

    --variant vgg16   13 conv layers + 3 FC   (config "D")
    --variant vgg19   16 conv layers + 3 FC   (config "E")

This keeps VGG's exact block structure and channel schedule. Two MNIST-aware
adaptations: BatchNorm after each conv (the standard "VGG-BN" variant, which
trains far better than the un-normalized 2014 original), and the pools use
`ceil_mode` + a final adaptive pool so the five downsampling stages survive a
28x28 input.

Run:
    python "09.vgg.py" --variant vgg16 --epochs 5
    python "09.vgg.py" --variant vgg19 --limit 2000
"""

import os

import torch.nn as nn

import mnist_common as mc

# "M" = max-pool. Numbers = conv output channels. These are the canonical configs.
CONFIGS = {
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
              512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
              512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def make_features(cfg):
    layers, in_ch = [], 1
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(2, ceil_mode=True))   # ceil so 28x28 never hits 0
        else:
            layers += [nn.Conv2d(in_ch, v, kernel_size=3, padding=1, bias=False),
                       nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_ch = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, variant: str, num_classes: int = 10):
        super().__init__()
        self.features = make_features(CONFIGS[variant])
        self.avgpool = nn.AdaptiveAvgPool2d(1)               # collapse to 512x1x1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)))


def main():
    p = mc.build_argparser("VGG on MNIST")
    p.add_argument("--variant", choices=list(CONFIGS), default="vgg16")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = VGG(args.variant)

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name=f"VGG-{args.variant[3:]}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
