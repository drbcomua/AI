"""
05. SphereFace & CosFace — the angular-margin lineage before ArcFace
====================================================================

ArcFace (04) is the top of a family of margin-based softmax losses that all
L2-normalize features and class weights and then push a margin into the angle.
This script covers the two it descends from, selectable via --variant:

    --variant sphereface   A-Softmax (Liu et al., 2017): MULTIPLICATIVE angular
                           margin, target logit ~ cos(m*theta). Strongest margin
                           but trickiest to optimize (needs the monotonic psi
                           function so cos(m*theta) stays decreasing).
    --variant cosface      Large Margin Cosine Loss (Wang et al., 2018): ADDITIVE
                           COSINE margin, target logit = cos(theta) - m. Simple and
                           very stable.

Together with ArcFace (additive ANGULAR margin, cos(theta + m)) these trace the
progression  SphereFace -> CosFace -> ArcFace.

Run:
    python "05.margin-softmax.py" --variant cosface --epochs 15
    python "05.margin-softmax.py" --variant sphereface --epochs 15
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import face_common as mc


class MarginProduct(nn.Module):
    def __init__(self, in_features, out_features, variant, s=30.0, m_cos=0.35, m_sphere=2):
        super().__init__()
        self.variant, self.s, self.m_cos, self.m_sphere = variant, s, m_cos, m_sphere
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        if self.variant == "cosface":
            phi = cosine - self.m_cos                          # additive cosine margin
        else:                                                   # sphereface: multiplicative angular margin
            theta = torch.acos(cosine)
            m = self.m_sphere
            k = torch.floor(m * theta / math.pi)
            phi = ((-1) ** k) * torch.cos(m * theta) - 2 * k    # monotonic psi(theta)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        return self.s * (one_hot * phi + (1 - one_hot) * cosine)


def main():
    p = mc.build_argparser("SphereFace / CosFace", epochs=15)
    args = p.parse_args()
    variant = args.variant or "cosface"
    device = mc.get_device(args.device)

    images, labels = mc.load_lfw()
    train_img, train_lbl, test_img, test_lbl = mc.split_lfw_identities(images, labels)
    num_classes = int(train_lbl.max().item() + 1)

    model = mc.FaceEmbeddingNet(embedding_dim=128).to(device)
    metric_fc = MarginProduct(128, num_classes, variant).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(metric_fc.parameters()),
                                 lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(train_img, train_lbl), batch_size=args.batch_size, shuffle=True)

    print(f"Training {variant.upper()} on {num_classes} identities | device: {device}")
    print("-" * 64)
    for epoch in range(1, args.epochs + 1):
        model.train(); metric_fc.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(metric_fc(model(x), y), y)
            loss.backward(); optimizer.step()
            total += loss.item() * x.size(0)
        print(f"Epoch {epoch:2d}/{args.epochs} | loss {total / len(train_img):.4f}")
    print("-" * 64)

    mc.verify_and_report(model, test_img, test_lbl, device, args.batch_size,
                         variant.capitalize(), os.path.dirname(os.path.abspath(__file__)), args.no_figure)


if __name__ == "__main__":
    main()
