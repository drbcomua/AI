"""
06. Multi-Layer Perceptron  —  "do more layers actually help?"
==============================================================

Before convolutions, digit classifiers were plain fully-connected networks
operating on the flattened 784-pixel vector. This script lets you dial the
*depth* up and down so you can watch the classic story unfold on MNIST:

    --variant linear   Linear(784->10) only            (logistic regression)
    --variant h1       1 hidden layer                   (the original "MLP")
    --variant h2       2 hidden layers
    --variant h4       4 hidden layers
    --variant h8       8 hidden layers

Things you will observe:

  * `linear` already reaches ~92% — MNIST is nearly linearly separable.
  * `h1`/`h2` jump to ~98%; one or two hidden layers is the sweet spot.
  * `h4`/`h8` do *not* keep improving. A plain deep MLP gains little and,
    without help, trains worse (vanishing gradients, no spatial prior). That
    is exactly the wall that BatchNorm/ResNet/ConvNets were invented to break.
    Each hidden layer here uses ReLU + BatchNorm so the deep variants train at
    all — try removing BatchNorm to see them fall apart.

Run:
    python "06.mlp.py" --variant h2 --epochs 5
    python "06.mlp.py" --variant linear
    python "06.mlp.py" --variant h8 --limit 4000
"""

import os

import torch.nn as nn

import mnist_common as mc

# variant -> number of hidden layers
VARIANTS = {"linear": 0, "h1": 1, "h2": 2, "h4": 4, "h8": 8}


class MLP(nn.Module):
    def __init__(self, hidden_layers: int, width: int = 256, num_classes: int = 10):
        super().__init__()
        layers = [nn.Flatten()]
        in_dim = 28 * 28
        for _ in range(hidden_layers):
            layers += [nn.Linear(in_dim, width), nn.BatchNorm1d(width), nn.ReLU(inplace=True)]
            in_dim = width
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    p = mc.build_argparser("MLP depth study on MNIST")
    p.add_argument("--variant", choices=list(VARIANTS), default="h2")
    p.add_argument("--width", type=int, default=256, help="hidden-layer width")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = MLP(VARIANTS[args.variant], width=args.width)

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name=f"MLP-{args.variant}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
