"""
11. Proxy-Anchor Loss (Kim et al., 2020)
========================================

Proxy-based metric learning: keep one learnable **proxy** vector per identity and
compare each embedding to the proxies instead of to other samples. This avoids the
combinatorial explosion (and hard-mining) of pair/triplet methods — every batch
sees all classes through their proxies — while still using true sample-to-sample
relations via the proxies as anchors.

    L = 1/|P+| * sum_{p in P+} softplus( logsumexp_{x in pos(p)}( -alpha (s(x,p) - delta) ) )
      + 1/|P|  * sum_{p}       softplus( logsumexp_{x in neg(p)}(  alpha (s(x,p) + delta) ) )

where s(x,p) is cosine similarity, P+ are proxies with a positive in the batch.

Key insights / educational takeaways:
    * Proxies give triplet-like gradients at softmax-like speed and stability.
    * Convergence is fast even with random batches (no mining needed).

Run:
    python "11.proxy-anchor.py" --epochs 20
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import face_common as mc


class ProxyAnchorLoss(nn.Module):
    def __init__(self, num_classes, dim, alpha=32.0, delta=0.1):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(num_classes, dim))
        self.alpha, self.delta, self.num_classes = alpha, delta, num_classes

    def forward(self, z, labels):
        proxies = F.normalize(self.proxies, dim=1)
        sim = z @ proxies.t()                                  # [B, C] cosine
        pos = torch.zeros_like(sim)
        pos.scatter_(1, labels.view(-1, 1), 1.0)               # [B, C] positive mask
        neg = 1 - pos

        with_pos = pos.sum(0) > 0                               # proxies present in this batch
        pos_logit = -self.alpha * (sim - self.delta)
        neg_logit = self.alpha * (sim + self.delta)
        # masked logsumexp over the batch dimension for each proxy
        pos_lse = torch.logsumexp(pos_logit.masked_fill(pos == 0, -1e9), dim=0)
        neg_lse = torch.logsumexp(neg_logit.masked_fill(neg == 0, -1e9), dim=0)
        pos_term = F.softplus(pos_lse)[with_pos].sum() / with_pos.sum().clamp(min=1)
        neg_term = F.softplus(neg_lse).sum() / self.num_classes
        return pos_term + neg_term


def main():
    args = mc.build_argparser("Proxy-Anchor Loss", epochs=20).parse_args()
    device = mc.get_device(args.device)

    images, labels = mc.load_lfw()
    train_img, train_lbl, test_img, test_lbl = mc.split_lfw_identities(images, labels)
    num_classes = int(train_lbl.max().item() + 1)

    model = mc.FaceEmbeddingNet(embedding_dim=128).to(device)
    criterion = ProxyAnchorLoss(num_classes, 128).to(device)
    # proxies usually want a larger learning rate than the backbone
    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": args.lr},
        {"params": criterion.parameters(), "lr": args.lr * 100},
    ], weight_decay=1e-4)
    loader = DataLoader(TensorDataset(train_img, train_lbl), batch_size=args.batch_size, shuffle=True)

    print(f"Training Proxy-Anchor on {num_classes} identities | device: {device}")
    print("-" * 64)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
            total += loss.item() * x.size(0)
        print(f"Epoch {epoch:2d}/{args.epochs} | loss {total / len(train_img):.4f}")
    print("-" * 64)

    mc.verify_and_report(model, test_img, test_lbl, device, args.batch_size,
                         "Proxy-Anchor", os.path.dirname(os.path.abspath(__file__)), args.no_figure)


if __name__ == "__main__":
    main()
