"""
06. Center Loss (Wen et al., 2016)
==================================

The bridge from plain softmax to the margin losses. Ordinary softmax separates
classes but doesn't make features *compact*: embeddings of one identity can spread
widely. Center Loss adds a second term that pulls each embedding toward a learned
**center** for its class, while softmax keeps the classes apart:

    L = L_softmax + (lambda / 2) * || x_i - c_{y_i} ||^2

The class centers c are learnable and updated alongside the network. The result is
intra-class compactness + inter-class separation — exactly the property that the
later angular-margin losses (SphereFace/CosFace/ArcFace) enforce more directly.

Run:
    python "06.center-loss.py" --epochs 15
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import face_common as mc


class CenterLoss(nn.Module):
    def __init__(self, num_classes, dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, dim))

    def forward(self, x, labels):
        c = self.centers[labels]
        return ((x - c) ** 2).sum(dim=1).mean()


def main():
    args = mc.build_argparser("Center Loss", epochs=15).parse_args()
    device = mc.get_device(args.device)
    lam = 0.01

    images, labels = mc.load_lfw()
    train_img, train_lbl, test_img, test_lbl = mc.split_lfw_identities(images, labels)
    num_classes = int(train_lbl.max().item() + 1)

    model = mc.FaceEmbeddingNet(embedding_dim=128).to(device)
    classifier = nn.Linear(128, num_classes).to(device)
    center_loss = CenterLoss(num_classes, 128).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()) + list(center_loss.parameters()),
        lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(train_img, train_lbl), batch_size=args.batch_size, shuffle=True)

    print(f"Training Center Loss (softmax + lambda*center) on {num_classes} identities | device: {device}")
    print("-" * 64)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            emb = model(x)
            loss = ce(classifier(emb), y) + lam * center_loss(emb, y)
            loss.backward(); optimizer.step()
            total += loss.item() * x.size(0)
        print(f"Epoch {epoch:2d}/{args.epochs} | loss {total / len(train_img):.4f}")
    print("-" * 64)

    mc.verify_and_report(model, test_img, test_lbl, device, args.batch_size,
                         "Center Loss", os.path.dirname(os.path.abspath(__file__)), args.no_figure)


if __name__ == "__main__":
    main()
