"""
09. Point Transformer (Zhao et al., 2021)
=========================================

Self-attention for point clouds. Where DGCNN convolves on a kNN graph, Point
Transformer applies **vector attention** within each point's local neighbourhood,
modulated by a learned **relative-position encoding** of the 3D offsets. "Vector"
attention means the attention weight is a per-channel vector (an MLP output), not a
single scalar — strictly more expressive than standard dot-product attention.

    for each point i, over its k neighbours j:
        attn_ij = softmax_j( gamma( phi(x_i) - psi(x_j) + delta(p_i - p_j) ) )
        y_i = sum_j attn_ij * ( alpha(x_j) + delta(p_i - p_j) )

Run:
    python "09.point-transformer.py" --epochs 10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import voxel_common as mc


class PointTransformerLayer(nn.Module):
    def __init__(self, dim, k=16):
        super().__init__()
        self.k = k
        self.q, self.k_lin, self.v = (nn.Linear(dim, dim) for _ in range(3))
        self.pos_mlp = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.attn_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x, xyz):                             # x [B,N,d], xyz [B,N,3]
        idx = mc.knn_indices(xyz, self.k)
        q = self.q(x)
        k = mc.index_points(self.k_lin(x), idx)           # [B, N, k, d]
        v = mc.index_points(self.v(x), idx)
        rel = xyz.unsqueeze(2) - mc.index_points(xyz, idx)            # [B, N, k, 3]
        pos = self.pos_mlp(rel)                                       # [B, N, k, d]
        attn = F.softmax(self.attn_mlp(q.unsqueeze(2) - k + pos), dim=2)   # vector attention
        return (attn * (v + pos)).sum(dim=2)                              # [B, N, d]


class PointTransformer(nn.Module):
    def __init__(self, num_points=256, dim=64, num_classes=10):
        super().__init__()
        self.num_points = num_points
        self.embed = nn.Linear(3, dim)
        self.pt1 = PointTransformerLayer(dim)
        self.pt2 = PointTransformerLayer(dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, voxels):
        xyz = mc.voxels_to_points(voxels, self.num_points)
        x = self.embed(xyz)
        x = x + self.pt1(x, xyz)                           # residual attention blocks
        x = x + self.pt2(x, xyz)
        return self.classifier(x.max(dim=1).values)        # global pool


def main():
    args = mc.build_argparser("Point Transformer on 3D MNIST", epochs=10).parse_args()
    device = mc.get_device(args.device)
    volumes, labels = mc.load_3d_mnist(limit=args.limit)
    s = int(len(volumes) * 0.8)
    print("Training Point Transformer (attention on points)...")
    mc.train_and_eval(PointTransformer(), volumes[:s], labels[:s], volumes[s:], labels[s:],
                      device, args, "Point Transformer")


if __name__ == "__main__":
    main()
