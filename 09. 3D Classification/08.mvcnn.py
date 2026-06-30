"""
08. MVCNN — Multi-View CNN (Su et al., 2015)
============================================

A third 3D *representation*: don't process the volume directly at all — **render it
to several 2D images** and let a mature 2D CNN do the work. Here the volume is
rotated to N viewpoints around its vertical axis and max-projected to silhouettes;
a single shared 2D CNN encodes each view, the per-view features are combined by a
**view-pooling** (element-wise max), and a classifier reads the pooled descriptor.

    volume -> N rendered views -> shared 2D CNN -> max-pool over views -> classify

Surprisingly, projecting to 2D and reusing strong image networks often *beats*
native 3D voxel CNNs, while using far less compute.

Run:
    python "08.mvcnn.py" --epochs 10
"""

import torch
import torch.nn as nn
import voxel_common as mc


class MVCNN(nn.Module):
    def __init__(self, num_views=6, num_classes=10):
        super().__init__()
        self.num_views = num_views
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, voxels):
        views = mc.voxels_to_multiview(voxels, self.num_views)   # [B, V, 1, H, W]
        B, V, C, H, W = views.shape
        feat = self.cnn(views.reshape(B * V, C, H, W))           # shared 2D CNN per view
        feat = feat.view(B, V, -1).max(dim=1).values             # view pooling
        return self.classifier(feat)


def main():
    args = mc.build_argparser("MVCNN on 3D MNIST", epochs=10).parse_args()
    device = mc.get_device(args.device)
    volumes, labels = mc.load_3d_mnist(limit=args.limit)
    s = int(len(volumes) * 0.8)
    print("Training MVCNN (multi-view projection representation)...")
    mc.train_and_eval(MVCNN(), volumes[:s], labels[:s], volumes[s:], labels[s:],
                      device, args, "MVCNN")


if __name__ == "__main__":
    main()
