"""
09. Face Identification — rank-1 accuracy & CMC curve
=====================================================

The other core face task. Verification (scripts 02-08) answers a 1:1 question —
"are these two the same person?". **Identification** is 1:N — "who, among a gallery
of known people, is this probe?". You enroll one image per identity into a gallery,
then for each probe rank the gallery by embedding distance.

Metrics:
    * rank-1 accuracy: how often the nearest gallery face is the right identity.
    * CMC (Cumulative Match Characteristic): rank-k accuracy as k grows — the curve
      rises to 100% as the gallery list lengthens.

The embedding here is trained with a CosFace head (any of the metric-learning
models in this folder would work); the script's focus is the identification
*evaluation*, which reuses `face_common.evaluate_identification`.

Run:
    python "09.identification.py" --epochs 15
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import face_common as mc


class CosFaceHead(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        return self.s * (cosine - one_hot * self.m)


def main():
    args = mc.build_argparser("Face Identification (CMC / rank-1)", epochs=15).parse_args()
    device = mc.get_device(args.device)

    images, labels = mc.load_lfw()
    train_img, train_lbl, test_img, test_lbl = mc.split_lfw_identities(images, labels)
    num_classes = int(train_lbl.max().item() + 1)

    model = mc.FaceEmbeddingNet(embedding_dim=128).to(device)
    head = CosFaceHead(128, num_classes).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()),
                                 lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(train_img, train_lbl), batch_size=args.batch_size, shuffle=True)

    print(f"Training a CosFace embedding for identification | device: {device}")
    print("-" * 64)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = ce(head(model(x), y), y)
            loss.backward(); optimizer.step()
            total += loss.item() * x.size(0)
        print(f"Epoch {epoch:2d}/{args.epochs} | loss {total / len(train_img):.4f}")
    print("-" * 64)

    # Identification evaluation on unseen test identities
    emb = mc.embed_all(model, test_img, device, args.batch_size)
    cmc = mc.evaluate_identification(emb, test_lbl, max_rank=10)
    print(f"Rank-1 accuracy: {cmc[0] * 100:.2f}%  |  Rank-5: {cmc[min(4, len(cmc)-1)] * 100:.2f}%")

    save_dir = os.path.dirname(os.path.abspath(__file__))
    if not args.no_figure:
        mc.plot_cmc(cmc, os.path.join(save_dir, "identification_cmc.png"), "CosFace Identification")
    # also report 1:1 verification AUC for comparison
    mc.verify_and_report(model, test_img, test_lbl, device, args.batch_size,
                         "Identification", save_dir, args.no_figure)


if __name__ == "__main__":
    main()
