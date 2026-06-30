"""
11. DenseNet3D (Huang et al., 2017, adapted to volumes)
=======================================================

DenseNet's idea ported to 3D: inside a dense block every layer receives the
concatenated feature maps of all preceding layers. This maximizes feature reuse
and strengthens gradient flow — especially valuable in 3D, where deep volumetric
nets are parameter- and memory-hungry, so the parameter efficiency of dense
connectivity pays off. Transition layers compress channels and downsample between
blocks.

    DenseLayer: BN-ReLU-Conv1x1x1 (bottleneck) -> BN-ReLU-Conv3x3x3 (growth) -> concat
    Transition: BN-ReLU-Conv1x1x1 (compress) -> AvgPool3d

Run:
    python "11.densenet3d.py" --epochs 5
"""

import torch
import torch.nn as nn
import voxel_common as mc


class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth, bn_size=4):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(in_ch), nn.ReLU(), nn.Conv3d(in_ch, bn_size * growth, 1, bias=False),
            nn.BatchNorm3d(bn_size * growth), nn.ReLU(),
            nn.Conv3d(bn_size * growth, growth, 3, padding=1, bias=False))

    def forward(self, x):
        return torch.cat([x, self.block(x)], dim=1)


class Transition(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(nn.BatchNorm3d(in_ch), nn.ReLU(),
                         nn.Conv3d(in_ch, out_ch, 1, bias=False), nn.AvgPool3d(2))


class DenseNet3D(nn.Module):
    def __init__(self, growth=8, block_config=(3, 6), num_classes=10):
        super().__init__()
        ch = 16
        layers = [nn.Conv3d(1, ch, 3, padding=1, bias=False)]
        for i, n in enumerate(block_config):
            for _ in range(n):
                layers.append(DenseLayer(ch, growth)); ch += growth
            if i != len(block_config) - 1:
                layers.append(Transition(ch, ch // 2)); ch //= 2
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.BatchNorm3d(ch), nn.ReLU(),
                                  nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(ch, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def main():
    args = mc.build_argparser("DenseNet3D on 3D MNIST", epochs=5).parse_args()
    device = mc.get_device(args.device)
    volumes, labels = mc.load_3d_mnist(limit=args.limit)
    s = int(len(volumes) * 0.8)
    print("Training DenseNet3D (dense volumetric connectivity)...")
    mc.train_and_eval(DenseNet3D(), volumes[:s], labels[:s], volumes[s:], labels[s:],
                      device, args, "DenseNet3D")


if __name__ == "__main__":
    main()
