"""
14. NASNet-A / B / C  (Zoph, Vasudevan, Shlens & Le, 2018 — "Learning Transferable Architectures")
=================================================================================================

NASNet doesn't hand-design a network — it *searches* for two reusable building
blocks (a **Normal Cell** that preserves resolution and a **Reduction Cell** that
halves it) on a small proxy task, then stacks copies of them. The cells combine
several separable-conv and pooling branches with element-wise addition, and a
cell's output is the concatenation of its branch results.

    --variant a / b / c

A practical honesty note: the search produced three top architectures (A, B, C),
but only **NASNet-A**'s exact cell wiring was published in detail; B and C are
usually cited only by their accuracy/scale. So here all three share a faithful
NASNet-A-style cell (stacked separable convs + pooling branches, two cell inputs,
concatenated output) and differ by the published *scaling* knobs — the number of
Normal cells per stage (N) and the filter count — which is the part you can vary
meaningfully on MNIST anyway.

Run:
    python "14.nasnet.py" --variant a --epochs 5
    python "14.nasnet.py" --variant b --limit 2000
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import mnist_common as mc

# variant -> (filters, normal_cells_per_stage)
VARIANTS = {"a": (32, 2), "b": (24, 3), "c": (48, 2)}


class SepConv(nn.Module):
    """NASNet 'separable conv': ReLU -> dw/pw -> BN, stacked twice (paper-faithful)."""

    def __init__(self, in_ch, out_ch, kernel, stride=1):
        super().__init__()
        pad = kernel // 2
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),   # input is shared across branches — must not modify in place
            nn.Conv2d(in_ch, in_ch, kernel, stride=stride, padding=pad, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel, stride=1, padding=pad, groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.op(x)


def _align(x, ref_ch, ref_hw, proj):
    """Match a tensor to (ref_ch channels, ref_hw spatial) for two-input cells."""
    if x.shape[-1] != ref_hw:
        x = F.adaptive_avg_pool2d(x, ref_hw)
    return proj(x)


class Cell(nn.Module):
    """A NASNet-A-style cell. stride=2 makes it a Reduction Cell."""

    def __init__(self, prev_ch, cur_ch, filters, stride=1):
        super().__init__()
        self.stride = stride
        self.proj_prev = nn.Sequential(nn.Conv2d(prev_ch, filters, 1, bias=False),
                                       nn.BatchNorm2d(filters))
        self.proj_cur = nn.Sequential(nn.Conv2d(cur_ch, filters, 1, bias=False),
                                      nn.BatchNorm2d(filters))
        f = filters
        self.sep3_cur = SepConv(f, f, 3, stride)
        self.sep5_prev = SepConv(f, f, 5, stride)
        self.sep5_cur = SepConv(f, f, 5, stride)
        self.sep3_prev = SepConv(f, f, 3, stride)
        self.sep3_cur2 = SepConv(f, f, 3, stride)
        self.out_ch = 5 * filters

    def forward(self, prev, cur):
        cur_hw = cur.shape[-1]
        cur = self.proj_cur(cur)
        prev = _align(prev, None, cur_hw, self.proj_prev)
        if self.stride == 2:
            # Reduction cell: pooling branches also downsample.
            pooled_cur = F.avg_pool2d(cur, 3, 2, 1)
            pooled_prev = F.avg_pool2d(prev, 3, 2, 1)
        else:
            pooled_cur = F.avg_pool2d(cur, 3, 1, 1)
            pooled_prev = F.avg_pool2d(prev, 3, 1, 1)
        skip_prev = prev if self.stride == 1 else F.avg_pool2d(prev, 1, 2)
        skip_cur = cur if self.stride == 1 else F.avg_pool2d(cur, 1, 2)

        b0 = self.sep3_cur(cur) + self.sep5_prev(prev)
        b1 = self.sep5_cur(cur) + self.sep3_prev(prev)
        b2 = pooled_cur + skip_prev
        b3 = pooled_prev + pooled_prev
        b4 = self.sep3_cur2(cur) + skip_cur
        return torch.cat([b0, b1, b2, b3, b4], dim=1)


class NASNet(nn.Module):
    def __init__(self, filters, n_normal, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, filters, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(filters))
        prev_ch = cur_ch = filters
        self.cells = nn.ModuleList()
        self.is_reduction = []

        def add(reduction):
            nonlocal prev_ch, cur_ch
            stride = 2 if reduction else 1
            cell = Cell(prev_ch, cur_ch, filters * (2 if reduction else 1), stride)
            self.cells.append(cell)
            self.is_reduction.append(reduction)
            prev_ch, cur_ch = cur_ch, cell.out_ch

        # Two stages of Normal cells separated/followed by Reduction cells.
        for _ in range(n_normal):
            add(False)
        add(True)
        for _ in range(n_normal):
            add(False)
        add(True)

        self.head = nn.Sequential(nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
                                  nn.Flatten(), nn.Linear(cur_ch, num_classes))

    def forward(self, x):
        x = self.stem(x)
        prev, cur = x, x
        for cell in self.cells:
            prev, cur = cur, cell(prev, cur)
        return self.head(cur)


def main():
    p = mc.build_argparser("NASNet on MNIST")
    p.add_argument("--variant", choices=list(VARIANTS), default="a")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)
    filters, n_normal = VARIANTS[args.variant]
    model = NASNet(filters, n_normal)

    mc.train(model, train_loader, test_loader, epochs=args.epochs,
             lr=args.lr, device=device)

    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name=f"NASNet-{args.variant.upper()}",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
