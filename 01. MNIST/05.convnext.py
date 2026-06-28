"""
05. ConvNeXt  (Liu, Mao, Wu, Feichtenhofer, Darrell & Xie, 2022 — "A ConvNet for the 2020s")
===========================================================================================

ConvNeXt "modernizes" a plain ResNet by borrowing design choices from Vision
Transformers while staying purely convolutional — and matches/beats Swin
Transformers on ImageNet. It is the natural modern counterpoint to the ViT you
already have. Each **ConvNeXt block** uses:

    * a 7x7 *depthwise* convolution (large receptive field, like attention windows)
    * LayerNorm (instead of BatchNorm)
    * an inverted bottleneck MLP: 1x1 expand x4 -> GELU -> 1x1 project
    * a learnable per-channel "layer scale" and a residual connection

plus a patchify stem and separate downsampling layers between stages.

This is a small MNIST-sized ConvNeXt (2 stages); the block design is faithful to
the paper.

Run:
    python "05.convnext.py" --epochs 5
    python "05.convnext.py" --limit 2000
"""

import os

import torch
import torch.nn as nn

import mnist_common as mc


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim of an (N, C, H, W) tensor."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)          # NCHW -> NHWC
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)       # back to NCHW


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)   # inverted bottleneck
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)          # NCHW -> NHWC for the MLP / norm
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)          # back to NCHW
        return shortcut + x


class MiniConvNeXt(nn.Module):
    def __init__(self, num_classes: int = 10, dims=(64, 128), depths=(2, 2)):
        super().__init__()
        # Patchify stem: 2x2 non-overlapping patches (28 -> 14).
        self.stem = nn.Sequential(
            nn.Conv2d(1, dims[0], kernel_size=2, stride=2),
            LayerNorm2d(dims[0]),
        )
        self.stage1 = nn.Sequential(*[ConvNeXtBlock(dims[0]) for _ in range(depths[0])])
        # Downsampling layer between stages (14 -> 7).
        self.down = nn.Sequential(
            LayerNorm2d(dims[0]),
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(*[ConvNeXtBlock(dims[1]) for _ in range(depths[1])])
        self.norm = nn.LayerNorm(dims[1], eps=1e-6)
        self.head = nn.Linear(dims[1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down(x)
        x = self.stage2(x)
        x = x.mean(dim=[2, 3])             # global average pooling
        x = self.norm(x)
        return self.head(x)


def main():
    args = mc.build_argparser("Mini-ConvNeXt on MNIST").parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = MiniConvNeXt()

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name="ConvNeXt",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
