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

Run:
    python "10.sparse-conv3d.py" --epochs 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import voxel_common as mc


class SubmanifoldConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)

    def forward(self, x, mask):
        return F.relu(self.bn(self.conv(x))) * mask        # outputs only at active sites


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
