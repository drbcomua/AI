"""
04. Spatial Transformer Network (STN)  (Jaderberg, Simonyan, Zisserman & Kavukcuoglu, 2015)
==========================================================================================

An STN is a *differentiable module* that learns to spatially warp its input
(crop, translate, rotate, scale, skew) so that downstream layers see a
canonicalized version of the object. It has three parts:

    1. Localisation network  -> predicts the parameters theta of an affine transform
    2. Grid generator        -> builds a sampling grid from theta
    3. Sampler               -> bilinearly samples the input on that grid

The whole thing is trained end-to-end with only the classification loss — no
supervision on *how* to transform. The original paper showcased it on distorted
and cluttered MNIST, making it a great fit here. This script wraps a small CNN
classifier with an STN front-end.

Tip: visualise `model.stn(x)` to literally see the network straighten the digits.

Run:
    python "04.spatial-transformer-network.py" --epochs 5
    python "04.spatial-transformer-network.py" --limit 2000
"""

import os

# Enable MPS fallback for missing operations (like grid_sampler_2d_backward)
# This MUST be set before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F

import mnist_common as mc


class STN(nn.Module):
    """Spatial transformer front-end producing a warped 1x28x28 image."""

    def __init__(self):
        super().__init__()
        # Localisation network: small CNN -> feature vector.
        self.loc = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),     # 28 -> 22
            nn.MaxPool2d(2), nn.ReLU(True),     # -> 11
            nn.Conv2d(8, 10, kernel_size=5),    # -> 7
            nn.MaxPool2d(2), nn.ReLU(True),     # -> 3
        )
        # Regressor for the 6 affine parameters (theta).
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True),
            nn.Linear(32, 6),
        )
        # Initialise to the identity transform so training starts as a no-op.
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.loc(x).flatten(1)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)


class STNClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stn = STN()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),   # 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),  # 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stn(x)          # spatially canonicalize first
        x = self.features(x)
        return self.classifier(x)


def main():
    args = mc.build_argparser("Spatial Transformer Network on MNIST").parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = STNClassifier()

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name="STN",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
