"""
08. Circle Loss (Sun et al., 2020)
==================================

A unified view of pair-based and class-based deep metric learning. Triplet and
softmax-margin losses optimize (s_p - s_n) with *equal, fixed* penalty on every
similarity. Circle Loss instead re-weights each similarity by *how far it is from
its optimum*: a within-class similarity that is already high is nudged little,
while one that is far from 1 is pushed hard (and symmetrically for negatives). The
decision boundary becomes a circle rather than a line, giving more definite
convergence.

    alpha_p = relu(1 + m - s_p),  alpha_n = relu(s_n + m)
    L = softplus( logsumexp_n( gamma * alpha_n * (s_n - m) )
                + logsumexp_p( -gamma * alpha_p * (s_p - (1 - m)) ) )

Computed over in-batch positive/negative similarities ("P identities x K images").

Run:
    python "08.circle-loss.py" --epochs 20
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import face_common as mc


def circle_loss(z, labels, m=0.25, gamma=64.0):
    sim = z @ z.t()                                            # cosine (z is L2-normalized)
    n = sim.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    same = (labels[:, None] == labels[None, :]) & ~eye         # positive pairs (exclude self)
    diff = labels[:, None] != labels[None, :]                  # negative pairs

    ap = torch.relu(1 + m - sim).detach()                      # per-similarity adaptive weights
    an = torch.relu(sim + m).detach()
    # masked_fill (not boolean indexing) keeps the backward MPS-safe; vectorized over anchors
    logit_p = (-gamma * ap * (sim - (1 - m))).masked_fill(~same, float("-inf"))
    logit_n = (gamma * an * (sim - m)).masked_fill(~diff, float("-inf"))
    per_anchor = F.softplus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1))
    valid = ((same.sum(1) > 0) & (diff.sum(1) > 0)).float()    # anchors with both pos & neg
    return (per_anchor * valid).sum() / valid.sum().clamp(min=1)


def main():
    args = mc.build_argparser("Circle Loss", epochs=20).parse_args()
    device = mc.get_device(args.device)
    if device.type == "mps":   # Gram-matrix loss hits an MPS BatchNorm-backward bug; CPU is fast here
        print("Note: in-batch Gram loss triggers an MPS autograd bug; falling back to CPU.")
        device = torch.device("cpu")
    P, K, steps = 16, 4, 60

    images, labels = mc.load_lfw()
    train_img, train_lbl, test_img, test_lbl = mc.split_lfw_identities(images, labels)

    model = mc.FaceEmbeddingNet(embedding_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"Training Circle Loss ({P} ids x {K} imgs/batch) | device: {device}")
    print("-" * 64)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in mc.iterate_pk_batches(train_img, train_lbl, P, K, steps):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = circle_loss(model(x), y)
            loss.backward(); optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch:2d}/{args.epochs} | loss {total / steps:.4f}")
    print("-" * 64)

    mc.verify_and_report(model, test_img, test_lbl, device, args.batch_size,
                         "Circle Loss", os.path.dirname(os.path.abspath(__file__)), args.no_figure)


if __name__ == "__main__":
    main()
