"""
05. PointNet — deep learning on raw point clouds (Qi et al., 2017)
=================================================================

A different 3D *representation*: instead of a dense voxel grid, treat the shape as
an unordered set of points (here sampled from the occupied voxels). The challenge
is permutation invariance — the network's output must not depend on point order.
PointNet solves it with a strikingly simple recipe:

    per-point shared MLP  ->  symmetric global pool (max)  ->  classifier

plus a small **T-Net** that learns a 3x3 transform to canonically align the input.
The max-pool is the key: it is permutation-invariant and learns to select the most
informative "critical points" of the shape.

Architecture Diagram / Layout:
    Voxels -> sample points [B, N, 3] -> T-Net align
       -> shared MLP (3->64->...->1024) per point
       -> global max-pool [B, 1024] -> MLP -> 10 classes

Run:
    python "05.pointnet.py" --epochs 10
"""

import torch
import torch.nn as nn
import voxel_common as mc


class TNet(nn.Module):
    """Learns a k x k input-alignment transform (initialized to identity)."""
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, k * k))

    def forward(self, x):                                   # x: [B, k, N]
        h = self.conv(x).max(dim=2).values
        return self.fc(h).view(-1, self.k, self.k) + torch.eye(self.k, device=x.device)


class PointNet(nn.Module):
    def __init__(self, num_points=256, num_classes=10):
        super().__init__()
        self.num_points = num_points
        self.tnet = TNet(3)
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, num_classes))

    def forward(self, voxels):
        x = mc.voxels_to_points(voxels, self.num_points).transpose(1, 2)   # [B, 3, N]
        x = torch.bmm(self.tnet(x), x)                     # canonical alignment
        x = self.mlp(x).max(dim=2).values                  # permutation-invariant global feature
        return self.classifier(x)


def main():
    args = mc.build_argparser("PointNet on 3D MNIST", epochs=10).parse_args()
    device = mc.get_device(args.device)
    volumes, labels = mc.load_3d_mnist(limit=args.limit)
    s = int(len(volumes) * 0.8)
    print("Training PointNet (point-cloud representation)...")
    mc.train_and_eval(PointNet(), volumes[:s], labels[:s], volumes[s:], labels[s:],
                      device, args, "PointNet")


if __name__ == "__main__":
    main()
