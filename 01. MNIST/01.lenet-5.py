"""
01. LeNet-5  (LeCun, Bottou, Bengio & Haffner, 1998)
====================================================

The original convolutional neural network, designed *specifically* for
handwritten-digit recognition on the NIST/MNIST data — so it is the natural
historical baseline for this folder.

Architecture (classic form, adapted to 28x28 inputs with padding):

    Input 1x28x28
      -> Conv 5x5, 6  + tanh   -> AvgPool 2x2          (S2)
      -> Conv 5x5, 16 + tanh   -> AvgPool 2x2          (S4)
      -> Conv 5x5, 120 + tanh  (acts as the C5 "full" layer)
      -> FC 84 + tanh                                  (F6)
      -> FC 10  (digit scores)

Key historical ideas: local receptive fields, shared weights (convolution),
sub-sampling, and a trainable end-to-end pipeline. ~60k parameters.

Run:
    python "01.lenet-5.py" --epochs 5
    python "01.lenet-5.py" --limit 2000        # fast smoke test
"""

import os

import torch.nn as nn

import mnist_common as mc


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),   # 28x28 -> 28x28
            nn.Tanh(),
            nn.AvgPool2d(2),                              # -> 14x14
            nn.Conv2d(6, 16, kernel_size=5),             # -> 10x10
            nn.Tanh(),
            nn.AvgPool2d(2),                             # -> 5x5
            nn.Conv2d(16, 120, kernel_size=5),          # -> 1x1
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    args = mc.build_argparser("LeNet-5 on MNIST").parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = LeNet5()

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name="LeNet-5",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
