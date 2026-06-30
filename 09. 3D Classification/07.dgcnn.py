"""
07. DGCNN — Dynamic Graph CNN / EdgeConv (Wang et al., 2019)
===========================================================

Point clouds as graphs. DGCNN builds a **k-nearest-neighbour graph** over the
points and convolves on the *edges*: for each point i and neighbour j it forms the
edge feature [x_i, x_j - x_i] (absolute + relative), passes it through an MLP, and
max-pools over neighbours. Crucially the graph is **dynamic** — recomputed in
*feature* space after every layer — so semantically similar points become
neighbours even when far apart in 3D. This directly connects 3D learning to the
message-passing GNNs of folder 07.

    EdgeConv(x): knn(x) -> MLP([x_i, x_j - x_i]) -> max over neighbours

Run:
    python "07.dgcnn.py" --epochs 10
"""

import torch
import torch.nn as nn
import voxel_common as mc


class EdgeConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=16):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(nn.Conv2d(2 * in_ch, out_ch, 1),
                                  nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2))

    def forward(self, x):                                  # x: [B, N, C]
        idx = mc.knn_indices(x, self.k)                   # dynamic graph in feature space
        neighbors = mc.index_points(x, idx)               # [B, N, k, C]
        central = x.unsqueeze(2).expand_as(neighbors)
        edge = torch.cat([central, neighbors - central], dim=-1).permute(0, 3, 2, 1)   # [B, 2C, k, N]
        return self.conv(edge).max(dim=2).values.transpose(1, 2)                        # [B, N, out]


class DGCNN(nn.Module):
    def __init__(self, num_points=256, num_classes=10):
        super().__init__()
        self.num_points = num_points
        self.ec1 = EdgeConv(3, 64)
        self.ec2 = EdgeConv(64, 64)
        self.ec3 = EdgeConv(64, 128)
        self.head_conv = nn.Sequential(nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512), nn.LeakyReLU(0.2))
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, voxels):
        x = mc.voxels_to_points(voxels, self.num_points)
        x1 = self.ec1(x); x2 = self.ec2(x1); x3 = self.ec3(x2)
        g = self.head_conv(torch.cat([x1, x2, x3], dim=-1).transpose(1, 2))   # [B, 512, N]
        feat = torch.cat([g.max(dim=2).values, g.mean(dim=2)], dim=1)         # [B, 1024]
        return self.classifier(feat)


def main():
    args = mc.build_argparser("DGCNN (EdgeConv) on 3D MNIST", epochs=10).parse_args()
    device = mc.get_device(args.device)
    volumes, labels = mc.load_3d_mnist(limit=args.limit)
    s = int(len(volumes) * 0.8)
    print("Training DGCNN (dynamic graph on points)...")
    mc.train_and_eval(DGCNN(), volumes[:s], labels[:s], volumes[s:], labels[s:],
                      device, args, "DGCNN")


if __name__ == "__main__":
    main()
