"""
29. Highway Networks  (Srivastava, Greff & Schmidhuber, 2015)
=============================================================

Highway Networks were the *direct precursor to ResNet*: the first architecture to
train very deep nets by adding learned shortcut pathways. Each layer computes a
transform H(x) and a learned **transform gate** T(x), and outputs

    y = H(x) * T(x) + x * (1 - T(x))

When the gate T is near 0 the input is "carried" through unchanged (an
information highway); when near 1 it behaves like a normal layer. The gate bias is
initialized negative so the network *starts* mostly carrying — exactly the trick
that lets gradients survive through dozens of layers. ResNet later showed the
simpler ungated identity shortcut (set T = 0.5 and drop the gate) works even
better.

This is a fully-connected Highway net on the flattened 784-pixel vector — a
deliberate contrast with `06.mlp.py`, where a plain deep MLP stops improving.
Stack more layers here and, thanks to the gates, it keeps training:

    --variant d10 | d20 | d50      (number of highway layers)

Run:
    python "29.highway.py" --variant d20 --epochs 5
    python "29.highway.py" --variant d50 --limit 4000
"""

import os

import torch
import torch.nn as nn

import mnist_common as mc

VARIANTS = {"d10": 10, "d20": 20, "d50": 50}


class HighwayLayer(nn.Module):
    def __init__(self, dim, gate_bias=-2.0):
        super().__init__()
        self.H = nn.Linear(dim, dim)
        self.T = nn.Linear(dim, dim)
        nn.init.constant_(self.T.bias, gate_bias)   # start biased toward "carry" (T ~ 0)

    def forward(self, x):
        h = torch.relu(self.H(x))
        t = torch.sigmoid(self.T(x))
        return h * t + x * (1 - t)


class HighwayNet(nn.Module):
    def __init__(self, n_layers, dim=128, num_classes=10):
        super().__init__()
        self.input = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, dim), nn.ReLU(inplace=True))
        self.highway = nn.Sequential(*[HighwayLayer(dim) for _ in range(n_layers)])
        self.output = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.output(self.highway(self.input(x)))


def main():
    p = mc.build_argparser("Highway Network on MNIST")
    p.add_argument("--variant", choices=list(VARIANTS), default="d20")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = HighwayNet(VARIANTS[args.variant])

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name=f"Highway-{args.variant}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
