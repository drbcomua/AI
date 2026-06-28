"""
34. Group-equivariant CNN (G-CNN, p4)  (Cohen & Welling, 2016)
=============================================================

An ordinary convolution is *translation*-equivariant: shift the input, the
feature map shifts. G-CNNs extend this to larger symmetry groups. This one uses
**p4** — the group of 90-degree rotations plus translations — so the network's
features rotate predictably when the input is rotated, and weights are shared
across all four orientations (4x more weight reuse, better sample efficiency on
rotation-rich data like handwriting).

Two layer types:

  * **Lifting conv (Z2 -> p4).** Convolve the input with the filter at all 4
    rotations, producing a feature map with an extra orientation axis of size 4.
  * **Group conv (p4 -> p4).** Filters now also span the 4 orientations; for each
    output rotation the filter is rotated spatially *and* cyclically shifted along
    the orientation axis, keeping everything equivariant.

A final **orientation pooling** (max over the 4 rotations) yields a rotation-
*invariant* descriptor for classification. (Caveat for MNIST: 6 vs 9 are rotations
of each other, so full rotation invariance is a slightly imperfect fit — which is
itself a nice thing to observe.)

Run:
    python "34.gcnn.py" --epochs 5
    python "34.gcnn.py" --limit 2000
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import mnist_common as mc


class P4ConvZ2(nn.Module):
    """Lifting convolution: plane (Z2) -> p4 feature with a size-4 orientation axis."""

    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.out_ch, self.padding = out_ch, padding
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, kernel, kernel))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):                                     # x: (B, in, H, W)
        # filter rotated by 0/90/180/270 deg -> one output orientation each
        ws = [torch.rot90(self.weight, r, dims=(2, 3)) for r in range(4)]
        w = torch.stack(ws, dim=1).reshape(self.out_ch * 4, x.size(1), *self.weight.shape[2:])
        y = F.conv2d(x, w, padding=self.padding)
        B, _, H, W = y.shape
        return y.view(B, self.out_ch, 4, H, W)


class P4ConvP4(nn.Module):
    """Group convolution p4 -> p4 (filter spans the 4 orientations)."""

    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.in_ch, self.out_ch, self.padding = in_ch, out_ch, padding
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, 4, kernel, kernel))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def _transformed_weight(self):
        mats = []
        for r in range(4):
            wr = torch.rot90(self.weight, r, dims=(3, 4))     # rotate filter spatially
            wr = torch.roll(wr, shifts=r, dims=2)             # cyclic shift on orientation axis
            mats.append(wr)
        w = torch.stack(mats, dim=1)                          # (out, 4, in, 4, k, k)
        k = self.weight.shape[-1]
        return w.reshape(self.out_ch * 4, self.in_ch * 4, k, k)

    def forward(self, x):                                     # x: (B, in, 4, H, W)
        B, _, _, H, W = x.shape
        xin = x.reshape(B, self.in_ch * 4, H, W)
        y = F.conv2d(xin, self._transformed_weight(), padding=self.padding)
        return y.view(B, self.out_ch, 4, y.shape[-2], y.shape[-1])


class GroupBN(nn.Module):
    """BatchNorm shared across the 4 orientations (BatchNorm3d over the group axis)."""

    def __init__(self, ch):
        super().__init__()
        self.bn = nn.BatchNorm3d(ch)

    def forward(self, x):
        return self.bn(x)


def group_spatial_pool(x):
    """2x2 spatial max-pool applied independently per orientation."""
    B, C, G, H, W = x.shape
    x = x.reshape(B, C * G, H, W)
    x = F.max_pool2d(x, 2)
    return x.view(B, C, G, H // 2, W // 2)


class GCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.lift = P4ConvZ2(1, 16)
        self.bn1 = GroupBN(16)
        self.conv2 = P4ConvP4(16, 32)
        self.bn2 = GroupBN(32)
        self.conv3 = P4ConvP4(32, 48)
        self.bn3 = GroupBN(48)
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Linear(48, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.lift(x)))            # (B,16,4,28,28)
        x = group_spatial_pool(x)                        # -> 14x14
        x = self.relu(self.bn2(self.conv2(x)))
        x = group_spatial_pool(x)                        # -> 7x7
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.amax(dim=2)                                # orientation pooling -> rotation invariant
        x = x.mean(dim=(2, 3))                           # global average pool
        return self.head(x)


def main():
    args = mc.build_argparser("Group-equivariant CNN (p4) on MNIST").parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = GCNN()

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name="G-CNN-p4",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
