"""
07. Simple CNN  —  the modern "hello world" baseline
====================================================

The textbook convolutional baseline almost every framework ships in its MNIST
tutorial: two conv+pool stages feeding a small classifier. It adds the three
ingredients LeNet-5 lacked — ReLU (instead of tanh), max-pooling (instead of
average), and dropout — and as a result comfortably reaches ~99% with only a
few epochs while staying tiny and fast.

Architecture:

    Input 1x28x28
      -> Conv 3x3, 32 + BN + ReLU -> MaxPool 2x2     (-> 14x14)
      -> Conv 3x3, 64 + BN + ReLU -> MaxPool 2x2     (-> 7x7)
      -> Flatten -> FC 128 + ReLU + Dropout(0.5)
      -> FC 10

Use this as the reference point the bigger architectures in this folder are
(over)engineered to beat.

Run:
    python "07.simple-cnn.py" --epochs 5
    python "07.simple-cnn.py" --limit 2000
"""

import os

import torch.nn as nn

import mnist_common as mc


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                                      # -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                                      # -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    args = mc.build_argparser("Simple CNN on MNIST").parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = SimpleCNN()

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name="SimpleCNN",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
