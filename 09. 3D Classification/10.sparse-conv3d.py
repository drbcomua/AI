"""
10. Submanifold Sparse Convolution (Graham & van der Maaten, 2017)
=================================================================

Voxel grids are mostly empty, yet a dense Conv3d spends compute on every cell and,
worse, "dilates" features into the empty space around the object — after enough
layers the whole grid becomes active and sparsity is lost. **Submanifold sparse
convolution** keeps the set of active sites *fixed*: it only produces outputs at
voxels that were occupied in the input, so the active set never grows and empty
regions stay empty.

This is an educational, dense-tensor emulation: a normal Conv3d followed by
multiplying with the (fixed) occupancy mask, which reproduces the key behaviour —
no smearing into empty space — while the real implementation skips the empty-voxel
computation entirely for big speed/memory wins.

A plain `nn.BatchNorm3d` would silently break this emulation: it normalizes over
*every* voxel in the dense grid, so its running mean/variance are dominated by the
90%+ of voxels that are always exactly zero (empty space). That leaves the running
statistics a poor estimate of the true active-voxel distribution, which shows up as
a large train/eval accuracy gap (eval-mode running stats scored ~20-30pts worse than
the same weights evaluated with fresh batch statistics). `MaskedBatchNorm3d` below
instead computes mean/variance only over the active (masked) sites, matching how
real sparse-conv libraries (MinkowskiEngine, spconv) normalize.

Run:
    python "10.sparse-conv3d.py" --epochs 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import voxel_common as mc


class MaskedBatchNorm3d(nn.Module):
    """BatchNorm whose statistics are computed only over active (masked) voxels,
    so the empty background never pollutes the running mean/variance."""
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x, mask):
        if self.training:
            n = mask.sum().clamp(min=1)
            xm = x * mask
            mean = xm.sum(dim=[0, 2, 3, 4]) / n
            var = (xm * x).sum(dim=[0, 2, 3, 4]) / n - mean ** 2
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.clamp(min=0))
        else:
            mean, var = self.running_mean, self.running_var

        inv_std = torch.rsqrt(var + self.eps)
        return ((x - mean.view(1, -1, 1, 1, 1)) * inv_std.view(1, -1, 1, 1, 1)
                * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1))


class SubmanifoldConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.bn = MaskedBatchNorm3d(out_ch)

    def forward(self, x, mask):
        h = self.bn(self.conv(x), mask)
        return F.relu(h) * mask                             # outputs only at active sites


class SparseConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.c1 = SubmanifoldConv(1, 16)
        self.c2 = SubmanifoldConv(16, 32)
        self.c3 = SubmanifoldConv(32, 64)
        self.classifier = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x):
        mask = (x > 0.1).float()                           # active sites, fixed throughout
        h = self.c1(x, mask)
        h = self.c2(h, mask)
        h = F.max_pool3d(h, 2)
        mask2 = (F.max_pool3d(mask, 2) > 0).float()
        h = self.c3(h, mask2)
        # global pooling over ACTIVE voxels only (masked average)
        pooled = (h * mask2).sum(dim=[2, 3, 4]) / mask2.sum(dim=[2, 3, 4]).clamp(min=1)
        return self.classifier(pooled)


def main():
    args = mc.build_argparser("Submanifold Sparse Conv3D on 3D MNIST", epochs=5).parse_args()
    device = mc.get_device(args.device)
    volumes, labels = mc.load_3d_mnist(limit=args.limit)
    s = int(len(volumes) * 0.8)
    print("Training Submanifold Sparse Conv3D...")
    mc.train_and_eval(SparseConvNet(), volumes[:s], labels[:s], volumes[s:], labels[s:],
                      device, args, "Sparse Conv3D")


if __name__ == "__main__":
    main()
