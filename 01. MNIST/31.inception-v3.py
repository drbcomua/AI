"""
31. Inception-v3  (Szegedy, Vanhoucke, Ioffe, Shlens & Wojna, 2016 — "Rethinking the Inception Architecture")
============================================================================================================

Inception-v3 is the grown-up version of the GoogLeNet in `02.googlenet-inception.py`.
It keeps the multi-branch Inception idea but adds the tricks that defined the
second wave of convnets:

  * **Factorized convolutions.** A 5x5 conv becomes two stacked 3x3s; an nxn conv
    becomes an asymmetric 1xn followed by nx1. Same receptive field, far fewer
    parameters — the central idea of v3.
  * **Three Inception module types** (A: factorized 5x5; B: asymmetric nxn;
    C: expanded/wide) and **efficient grid-reduction** modules that downsample
    without a representational bottleneck.
  * **Label smoothing** (introduced in this paper) as a regularizer.

This is a faithful but MNIST-scaled v3: the factorized A/B/C modules and both
reduction modules, with channel counts shrunk for 28x28. (The auxiliary
classifier is omitted so the standard training loop applies.)

Run:
    python "31.inception-v3.py" --epochs 5
    python "31.inception-v3.py" --limit 2000
"""

import os

import torch
import torch.nn as nn

import mnist_common as mc


class ConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class InceptionA(nn.Module):
    """1x1 | 1x1->5x5 | 1x1->3x3->3x3 (factorized 5x5) | pool->1x1."""

    def __init__(self, in_ch, pool_ch):
        super().__init__()
        self.b1 = ConvBN(in_ch, 64, kernel_size=1)
        self.b2 = nn.Sequential(ConvBN(in_ch, 48, kernel_size=1),
                                ConvBN(48, 64, kernel_size=5, padding=2))
        self.b3 = nn.Sequential(ConvBN(in_ch, 64, kernel_size=1),
                                ConvBN(64, 96, kernel_size=3, padding=1),
                                ConvBN(96, 96, kernel_size=3, padding=1))
        self.b4 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1),
                                ConvBN(in_ch, pool_ch, kernel_size=1))
        self.out_ch = 64 + 64 + 96 + pool_ch

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class InceptionB(nn.Module):
    """Asymmetric factorization: nxn -> 1xn then nx1 (n=7)."""

    def __init__(self, in_ch, c7):
        super().__init__()
        self.b1 = ConvBN(in_ch, 192, kernel_size=1)
        self.b2 = nn.Sequential(
            ConvBN(in_ch, c7, kernel_size=1),
            ConvBN(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            ConvBN(c7, 192, kernel_size=(7, 1), padding=(3, 0)))
        self.b3 = nn.Sequential(
            ConvBN(in_ch, c7, kernel_size=1),
            ConvBN(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            ConvBN(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            ConvBN(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            ConvBN(c7, 192, kernel_size=(1, 7), padding=(0, 3)))
        self.b4 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1),
                                ConvBN(in_ch, 192, kernel_size=1))
        self.out_ch = 192 * 3 + 192

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class InceptionC(nn.Module):
    """Expanded module: the asymmetric pair runs in *parallel* (1x3 || 3x1)."""

    def __init__(self, in_ch):
        super().__init__()
        self.b1 = ConvBN(in_ch, 192, kernel_size=1)
        self.b2_1 = ConvBN(in_ch, 256, kernel_size=1)
        self.b2a = ConvBN(256, 128, kernel_size=(1, 3), padding=(0, 1))
        self.b2b = ConvBN(256, 128, kernel_size=(3, 1), padding=(1, 0))
        self.b3_1 = nn.Sequential(ConvBN(in_ch, 256, kernel_size=1),
                                  ConvBN(256, 256, kernel_size=3, padding=1))
        self.b3a = ConvBN(256, 128, kernel_size=(1, 3), padding=(0, 1))
        self.b3b = ConvBN(256, 128, kernel_size=(3, 1), padding=(1, 0))
        self.b4 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1),
                                ConvBN(in_ch, 192, kernel_size=1))
        self.out_ch = 192 + 256 + 256 + 192

    def forward(self, x):
        b2 = self.b2_1(x)
        b3 = self.b3_1(x)
        return torch.cat([self.b1(x),
                          self.b2a(b2), self.b2b(b2),
                          self.b3a(b3), self.b3b(b3),
                          self.b4(x)], dim=1)


class Reduction(nn.Module):
    """Efficient grid reduction: parallel strided convs + strided pool (halves size)."""

    def __init__(self, in_ch, a_out, b_mid, b_out):
        super().__init__()
        self.b1 = ConvBN(in_ch, a_out, kernel_size=3, stride=2, padding=1)
        self.b2 = nn.Sequential(ConvBN(in_ch, b_mid, kernel_size=1),
                                ConvBN(b_mid, b_mid, kernel_size=3, padding=1),
                                ConvBN(b_mid, b_out, kernel_size=3, stride=2, padding=1))
        self.b3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.out_ch = a_out + b_out + in_ch

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)


class InceptionV3(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBN(1, 32, kernel_size=3, padding=1),
            ConvBN(32, 32, kernel_size=3, padding=1),
            ConvBN(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),                                       # 28 -> 14
        )
        a1 = InceptionA(64, pool_ch=32)
        a2 = InceptionA(a1.out_ch, pool_ch=64)
        self.stageA = nn.Sequential(a1, a2)
        rA = Reduction(a2.out_ch, a_out=192, b_mid=64, b_out=96)   # 14 -> 7
        self.redA = rA
        b1 = InceptionB(rA.out_ch, c7=128)
        self.stageB = nn.Sequential(b1)
        rB = Reduction(b1.out_ch, a_out=192, b_mid=128, b_out=192)  # 7 -> 4
        self.redB = rB
        c1 = InceptionC(rB.out_ch)
        self.stageC = nn.Sequential(c1)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(c1.out_ch, num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.stageA(x)
        x = self.redA(x)
        x = self.stageB(x)
        x = self.redB(x)
        x = self.stageC(x)
        return self.head(x)


def main():
    args = mc.build_argparser("Inception-v3 (MNIST-scaled) on MNIST").parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = InceptionV3()

    # Label smoothing was introduced in the Inception-v3 paper.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device, criterion=criterion)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name="Inception-v3",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
