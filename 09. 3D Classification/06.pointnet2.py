"""
06. PointNet++ — hierarchical point-set learning (Qi et al., 2017)
=================================================================

PointNet pools over *all* points at once, so it captures global shape but misses
local geometry. PointNet++ adds a hierarchy of **Set Abstraction** layers that
mirror a CNN's growing receptive field:

    sample  : pick a subset of points as new centroids
    group   : gather each centroid's k nearest neighbours
    pointnet: a local PointNet (shared MLP + max) summarizes each neighbourhood

Stacking these builds features at coarser and coarser scales before a final global
pooling — so the network learns local patterns and how they compose.

Run:
    python "06.pointnet2.py" --epochs 10
"""

import torch
import torch.nn as nn
import voxel_common as mc


class SetAbstraction(nn.Module):
    def __init__(self, npoint, k, in_ch, out_ch):
        super().__init__()
        self.npoint, self.k = npoint, k
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch + 3, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU())

    def forward(self, xyz, feat):                          # xyz [B,N,3], feat [B,N,C] or None
        B, N, _ = xyz.shape
        cen_idx = torch.stack([torch.randperm(N, device=xyz.device)[:self.npoint] for _ in range(B)])
        centroids = mc.index_points(xyz, cen_idx)          # [B, S, 3]
        knn = torch.cdist(centroids, xyz).topk(self.k, dim=-1, largest=False).indices   # [B, S, k]
        grouped_xyz = mc.index_points(xyz, knn) - centroids.unsqueeze(2)    # [B, S, k, 3] relative
        if feat is not None:
            grouped = torch.cat([grouped_xyz, mc.index_points(feat, knn)], dim=-1)
        else:
            grouped = grouped_xyz
        grouped = grouped.permute(0, 3, 2, 1)              # [B, C+3, k, S]
        new_feat = self.mlp(grouped).max(dim=2).values     # [B, out, S]
        return centroids, new_feat.transpose(1, 2)         # feat [B, S, out]


class PointNet2(nn.Module):
    def __init__(self, num_points=256, num_classes=10):
        super().__init__()
        self.num_points = num_points
        self.sa1 = SetAbstraction(npoint=64, k=16, in_ch=0, out_ch=64)
        self.sa2 = SetAbstraction(npoint=16, k=16, in_ch=64, out_ch=128)
        self.global_mlp = nn.Sequential(
            nn.Conv1d(128 + 3, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, voxels):
        xyz = mc.voxels_to_points(voxels, self.num_points)
        c1, f1 = self.sa1(xyz, None)
        c2, f2 = self.sa2(c1, f1)
        g = torch.cat([c2, f2], dim=-1).transpose(1, 2)    # [B, 3+128, S]
        g = self.global_mlp(g).max(dim=2).values
        return self.classifier(g)


def main():
    args = mc.build_argparser("PointNet++ on 3D MNIST", epochs=10).parse_args()
    device = mc.get_device(args.device)
    volumes, labels = mc.load_3d_mnist(limit=args.limit)
    s = int(len(volumes) * 0.8)
    print("Training PointNet++ (hierarchical set abstraction)...")
    mc.train_and_eval(PointNet2(), volumes[:s], labels[:s], volumes[s:], labels[s:],
                      device, args, "PointNet++")


if __name__ == "__main__":
    main()
