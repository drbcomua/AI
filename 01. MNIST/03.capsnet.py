"""
03. CapsNet — Capsule Network  (Sabour, Frosst & Hinton, 2017)
=============================================================

"Dynamic Routing Between Capsules" replaced scalar neurons with **capsules**:
small groups of neurons whose *vector* output encodes the instantiation
parameters of an entity, and whose *length* encodes the probability that the
entity is present. Capsules in one layer predict the outputs of capsules in the
next, and a **routing-by-agreement** procedure iteratively strengthens
predictions that agree (cluster together). The paper introduced and benchmarked
the idea on MNIST, so this is a very fitting demo for this folder.

Pipeline:
    Input 1x28x28
      -> Conv 9x9, 256, ReLU                     -> 20x20x256
      -> PrimaryCaps: Conv 9x9 s2 -> 32 caps x 8D -> 6x6 grid = 1152 capsules
      -> DigitCaps: 10 capsules x 16D, 3 routing iterations
      -> output = per-class capsule *lengths*  (used as the class scores)

Non-linearity: the `squash` function, which shrinks short vectors toward 0 and
long vectors toward unit length. Trained with **margin loss** (not
cross-entropy). The reconstruction-decoder regularizer from the paper is omitted
here to keep the demo focused on routing.

Note: routing is compute-heavy; 2-3 epochs already reach ~99% test accuracy.

Run:
    python "03.capsnet.py" --epochs 3
    python "03.capsnet.py" --limit 2000 --epochs 2
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import mnist_common as mc


def squash(s, dim=-1, eps=1e-8):
    """Non-linear 'squashing': ||v|| in [0,1), direction preserved."""
    sq_norm = (s ** 2).sum(dim=dim, keepdim=True)
    scale = sq_norm / (1.0 + sq_norm)
    return scale * s / torch.sqrt(sq_norm + eps)


class PrimaryCaps(nn.Module):
    def __init__(self, in_ch=256, caps_dim=8, n_maps=32, kernel=9, stride=2):
        super().__init__()
        self.caps_dim = caps_dim
        self.conv = nn.Conv2d(in_ch, n_maps * caps_dim, kernel_size=kernel, stride=stride)

    def forward(self, x):
        out = self.conv(x)                       # (B, n_maps*caps_dim, 6, 6)
        B = out.size(0)
        out = out.view(B, -1, self.caps_dim)     # (B, 1152, 8)
        return squash(out, dim=-1)


class DigitCaps(nn.Module):
    def __init__(self, n_in=1152, in_dim=8, n_out=10, out_dim=16, routing_iters=3):
        super().__init__()
        self.n_out = n_out
        self.routing_iters = routing_iters
        # Transformation matrices W: (1, n_in, n_out, out_dim, in_dim)
        self.W = nn.Parameter(0.01 * torch.randn(1, n_in, n_out, out_dim, in_dim))

    def forward(self, u):
        B = u.size(0)
        # u: (B, n_in, in_dim) -> u_hat predictions: (B, n_in, n_out, out_dim)
        u = u.unsqueeze(2).unsqueeze(-1)                     # (B, n_in, 1, in_dim, 1)
        u_hat = torch.matmul(self.W, u).squeeze(-1)          # (B, n_in, n_out, out_dim)

        # Routing by agreement (logits b detached from u_hat gradient path).
        b = torch.zeros(B, u_hat.size(1), self.n_out, device=u.device)
        u_hat_detached = u_hat.detach()
        for it in range(self.routing_iters):
            c = F.softmax(b, dim=2).unsqueeze(-1)            # coupling coeffs
            src = u_hat if it == self.routing_iters - 1 else u_hat_detached
            s = (c * src).sum(dim=1)                         # (B, n_out, out_dim)
            v = squash(s, dim=-1)
            if it < self.routing_iters - 1:
                b = b + (u_hat_detached * v.unsqueeze(1)).sum(-1)
        return v                                             # (B, n_out, out_dim)


class CapsNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9)        # 28 -> 20
        self.primary = PrimaryCaps()
        self.digits = DigitCaps(n_out=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.primary(x)
        v = self.digits(x)                                   # (B, 10, 16)
        # Class scores = capsule lengths; argmax decodes the prediction.
        return v.norm(dim=-1)                                # (B, 10)


class MarginLoss(nn.Module):
    """Margin loss from the paper (m+ = 0.9, m- = 0.1, lambda = 0.5)."""

    def __init__(self, m_pos=0.9, m_neg=0.1, lam=0.5, num_classes=10):
        super().__init__()
        self.m_pos, self.m_neg, self.lam, self.k = m_pos, m_neg, lam, num_classes

    def forward(self, lengths, target):
        t = F.one_hot(target, self.k).float()
        pos = t * F.relu(self.m_pos - lengths) ** 2
        neg = self.lam * (1 - t) * F.relu(lengths - self.m_neg) ** 2
        return (pos + neg).sum(dim=1).mean()


def main():
    args = mc.build_argparser("CapsNet on MNIST", epochs=3).parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    model = CapsNet()

    mc.train(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr,
             device=device, criterion=MarginLoss())

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name="CapsNet",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
