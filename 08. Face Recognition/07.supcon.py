"""
07. Supervised Contrastive Loss (Khosla et al., 2020)
=====================================================

The modern, batch-level successor to pairwise Siamese / triplet learning. Within a
batch containing several images per identity, every same-identity image is a
positive and every other image a negative — so each anchor is pulled toward *all*
its positives at once and pushed from *all* negatives, with no explicit pair or
triplet mining:

    L_i = -1/|P(i)| * sum_{p in P(i)} log( exp(z_i·z_p / tau)
                                           / sum_{a != i} exp(z_i·z_a / tau) )

Batches are sampled "P identities x K images" so positives always exist.

Key insights / educational takeaways:
    * Many positives + many negatives per step gives far richer gradients than a
      single triplet, and removes the hard-mining headache.
    * The temperature tau controls how sharply the embedding space is concentrated.

Run:
    python "07.supcon.py" --epochs 20
"""

import os
import torch
import torch.nn as nn
import face_common as mc


def supcon_loss(z, labels, tau=0.1):
    n = z.size(0)
    sim = (z @ z.t()) / tau
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()    # numerical stability
    eye = torch.eye(n, device=z.device)
    exp_sim = torch.exp(sim) * (1 - eye)                        # exclude self
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    same = (labels[:, None] == labels[None, :]).float() * (1 - eye)
    pos_per = same.sum(dim=1)
    mean_log_prob = (same * log_prob).sum(dim=1) / pos_per.clamp(min=1)
    # masked mean over anchors that have positives (avoid boolean indexing -> MPS-safe)
    valid = (pos_per > 0).float()
    return -(mean_log_prob * valid).sum() / valid.sum().clamp(min=1)


def main():
    args = mc.build_argparser("Supervised Contrastive Loss", epochs=20).parse_args()
    device = mc.get_device(args.device)
    if device.type == "mps":   # Gram-matrix loss hits an MPS BatchNorm-backward bug; CPU is fast here
        print("Note: in-batch Gram loss triggers an MPS autograd bug; falling back to CPU.")
        device = torch.device("cpu")
    P, K, steps = 16, 4, 60                                     # P identities x K images per batch

    images, labels = mc.load_lfw()
    train_img, train_lbl, test_img, test_lbl = mc.split_lfw_identities(images, labels)

    model = mc.FaceEmbeddingNet(embedding_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"Training SupCon ({P} ids x {K} imgs/batch) | device: {device}")
    print("-" * 64)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in mc.iterate_pk_batches(train_img, train_lbl, P, K, steps):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = supcon_loss(model(x), y)
            loss.backward(); optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch:2d}/{args.epochs} | loss {total / steps:.4f}")
    print("-" * 64)

    mc.verify_and_report(model, test_img, test_lbl, device, args.batch_size,
                         "SupCon", os.path.dirname(os.path.abspath(__file__)), args.no_figure)


if __name__ == "__main__":
    main()
