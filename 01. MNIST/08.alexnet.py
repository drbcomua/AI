"""
08. AlexNet  (Krizhevsky, Sutskever & Hinton, 2012)
===================================================

The network that started the deep-learning boom by winning ILSVRC-2012 by a
huge margin. Its contributions — ReLU activations, dropout, data augmentation,
GPU training, and (its now-retired quirk) Local Response Normalization — became
standard practice overnight.

The original takes 224x224 inputs through 11x11/stride-4 convolutions; that
stem makes no sense on a 28x28 digit. This is a **faithful but MNIST-scaled**
AlexNet: the same 5-conv / 3-FC layout, ReLU, overlapping max-pooling, LRN, and
heavy dropout, with kernels and strides shrunk to fit 28x28.

    5 conv layers (with 2 LRN + 3 max-pools) -> 3 FC layers with dropout.

Run:
    python "08.alexnet.py" --epochs 5
    python "08.alexnet.py" --limit 2000
"""

import os

import torch.nn as nn

import mnist_common as mc


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2), nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                 # 28 -> 14
            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                 # 14 -> 7
            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                 # 7 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5), nn.Linear(256 * 4 * 4, 1024), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(1024, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    args = mc.build_argparser("AlexNet (MNIST-scaled) on MNIST").parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = AlexNet()

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name="AlexNet",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
