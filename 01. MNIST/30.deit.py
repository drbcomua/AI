"""
30. DeiT — Data-efficient image Transformer  (Touvron et al., 2021)
===================================================================

A vanilla ViT (`16.vit.py`) needs enormous datasets because it has no
built-in notion of locality. DeiT makes transformers trainable on modest data
with one architectural addition: a **distillation token**.

Alongside the usual [class] token, a second learnable [distill] token is appended
to the sequence. It passes through the same transformer but is trained to predict
the output of a *teacher* network (here a small CNN trained on the fly) — the
"hard distillation" recipe from the paper. At inference the two heads are
averaged. The student effectively inherits the CNN's spatial inductive bias
through the teacher's labels, so it learns far more sample-efficiently.

This script trains a tiny CNN teacher for a few epochs, then distills a DeiT
student from it. Use `--no-distill` to instead supervise both tokens with the
ground-truth labels (i.e. a plain dual-token ViT) for comparison.

Run:
    python "30.deit.py" --epochs 10
    python "30.deit.py" --no-distill --limit 4000
"""

import os

import torch
import torch.nn as nn

import mnist_common as mc


# --------------------------------------------------------------------------- #
# Teacher: a small, fast CNN (provides distillation targets)
# --------------------------------------------------------------------------- #
class TeacherCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(inplace=True), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# Student: ViT with an extra distillation token
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class DeiT(nn.Module):
    def __init__(self, img_size=28, patch=7, dim=64, depth=6, heads=4,
                 mlp_ratio=2.0, num_classes=10, dropout=0.1):
        super().__init__()
        n = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, dim))     # the DeiT addition
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 2, dim))  # +2: cls and dist tokens
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads, mlp_ratio, dropout)
                                      for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.head_dist = nn.Linear(dim, num_classes)
        for pmt in (self.pos_embed, self.cls_token, self.dist_token):
            nn.init.trunc_normal_(pmt, std=0.02)

    def _tokens(self, x):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        dist = self.dist_token.expand(B, -1, -1)
        x = torch.cat([cls, dist, x], dim=1) + self.pos_embed
        x = self.norm(self.blocks(self.dropout(x)))
        return self.head(x[:, 0]), self.head_dist(x[:, 1])

    def forward(self, x):
        cls_out, dist_out = self._tokens(x)
        return (cls_out + dist_out) / 2            # inference: average the two heads

    def forward_train(self, x):
        return self._tokens(x)                     # training: keep heads separate


def train_distill(student, teacher, train_loader, test_loader, *, epochs, lr, device, distill):
    student.to(device); teacher.to(device); teacher.eval()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    mode = "hard distillation from CNN teacher" if distill else "dual-token, label-supervised"
    print(f"Device: {device} | student params: {n_params:,} | mode: {mode}")
    print("-" * 64)
    for epoch in range(1, epochs + 1):
        student.train()
        running = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if distill:
                with torch.no_grad():
                    teacher_labels = teacher(x).argmax(1)
            else:
                teacher_labels = y
            cls_out, dist_out = student.forward_train(x)
            loss = 0.5 * ce(cls_out, y) + 0.5 * ce(dist_out, teacher_labels)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * y.size(0)
            correct += ((cls_out + dist_out).argmax(1) == y).sum().item()
            total += y.size(0)
        test_acc = mc._accuracy(student, test_loader, device)
        print(f"Epoch {epoch:2d}/{epochs} | loss {running / total:.4f} | "
              f"train_acc {correct / total:.4f} | test_acc {test_acc:.4f}")
    print("-" * 64)


def main():
    p = mc.build_argparser("DeiT (distilled ViT) on MNIST", epochs=10)
    p.add_argument("--no-distill", action="store_true",
                   help="supervise both tokens with labels instead of a CNN teacher")
    p.add_argument("--teacher-epochs", type=int, default=2)
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader = mc.get_dataloaders(args.batch_size, args.limit)

    teacher = TeacherCNN()
    if not args.no_distill:
        print("Training CNN teacher...")
        mc.train(teacher, train_loader, test_loader, epochs=args.teacher_epochs,
                 lr=1e-3, device=device)

    student = DeiT()
    train_distill(student, teacher, train_loader, test_loader, epochs=args.epochs,
                  lr=args.lr, device=device, distill=not args.no_distill)

    y_true, y_pred, y_prob = mc.evaluate(student, test_loader, device)
    mc.report(y_true, y_pred, y_prob, model_name="DeiT",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
