"""
03. Triplet Network
===================

Three-way comparison metric learning utilizing Triplet Loss and online/offline negative mining (Schroff et al., FaceNet, 2015).

Triplet Loss Formulation:
    L = max(0, ||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + margin)
    where A is Anchor (target face), P is Positive (same identity), and N is Negative (different identity).

Key insights / educational takeaways:
    * Triplet learning directly optimizes relative distances, pushing negatives further from the anchor than positives.
    * Hard negative mining is critical because easy negatives produce 0 gradients, causing model learning to stall.

Run:
    python "03.triplet-net.py" --epochs 15
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import face_common as mc


class TripletLoss(nn.Module):
    """Semi-hard/Hard margin Triplet Loss."""
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # L2 distances squared
        d_pos = torch.sum(torch.pow(anchor - positive, 2), dim=1)
        d_neg = torch.sum(torch.pow(anchor - negative, 2), dim=1)
        loss = torch.clamp(d_pos - d_neg + self.margin, min=0.0)
        return loss.mean()


def mine_triplets(images, labels, model, device, num_triplets=4000):
    """Perform semi-hard/hard offline triplet mining.

    Extracts embeddings, then for each identity pair, chooses informative negatives.
    """
    model.eval()
    B = len(images)

    # Extract all embeddings to find hard negatives
    embeddings = []
    with torch.no_grad():
        for i in range(0, B, 128):
            xb = images[i:i+128].to(device)
            embeddings.append(model(xb).cpu())
    embeddings = torch.cat(embeddings, dim=0) # [B, D]

    labels_np = labels.numpy()
    unique_labels = np.unique(labels_np)
    label_to_indices = {l: np.where(labels_np == l)[0] for l in unique_labels}
    valid_labels = [l for l in unique_labels if len(label_to_indices[l]) >= 2]

    anchors = []
    positives = []
    negatives = []

    # Precompute pairwise distances in CPU
    # dist[i, j] = Euclidean distance between embedding i and j
    dist_matrix = torch.cdist(embeddings, embeddings, p=2).numpy()

    for _ in range(num_triplets):
        # Pick anchor identity
        l_ap = random.choice(valid_labels)
        idx_a, idx_p = np.random.choice(label_to_indices[l_ap], size=2, replace=False)

        # Distance to positive
        d_ap = dist_matrix[idx_a, idx_p]

        # Candidates for negatives (different identities)
        neg_labels = [l for l in unique_labels if l != l_ap]
        l_n = random.choice(neg_labels)

        # For the selected negative class, find the node that maximizes loss (hard negative)
        neg_indices = label_to_indices[l_n]
        d_ans = dist_matrix[idx_a, neg_indices]

        # Select the 'hardest' negative in the group (smallest distance to anchor)
        hardest_idx = neg_indices[np.argmin(d_ans)]

        anchors.append(images[idx_a])
        positives.append(images[idx_p])
        negatives.append(images[hardest_idx])

    return (
        torch.stack(anchors),
        torch.stack(positives),
        torch.stack(negatives)
    )


def main():
    p = mc.build_argparser("Triplet Network Face Verifier", epochs=15)
    args = p.parse_args()

    device = mc.get_device(args.device)

    # Load and split LFW by identities (Ensures open-set test evaluation)
    images, labels = mc.load_lfw()
    train_img, train_lbl, test_img, test_lbl = mc.split_lfw_identities(images, labels)

    model = mc.FaceEmbeddingNet(embedding_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = TripletLoss(margin=0.5)

    print("Training Triplet Network with Offline Hard Negative Mining...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        # Generate mined triplets dynamically at the start of each epoch
        anchors, positives, negatives = mine_triplets(train_img, train_lbl, model, device, num_triplets=4000)

        dataset = TensorDataset(anchors, positives, negatives)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        model.train()
        epoch_loss = 0.0
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)

            optimizer.zero_grad()
            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)

            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * a.size(0)

        epoch_loss_avg = epoch_loss / len(dataset)
        print(f"Epoch {epoch:2d}/{args.epochs} | loss: {epoch_loss_avg:.4f}")

    print("-" * 64)

    # Evaluation on unseen test identities
    print("Generating unseen verification test pairs...")
    test_p1, test_p2, test_pair_lbls = mc.generate_verification_pairs(test_img, test_lbl, num_pairs=1000)

    model.eval()
    distances = []
    with torch.no_grad():
        for i in range(0, len(test_p1), args.batch_size):
            x1 = test_p1[i:i+args.batch_size].to(device)
            x2 = test_p2[i:i+args.batch_size].to(device)
            emb1 = model(x1)
            emb2 = model(x2)
            d = torch.norm(emb1 - emb2, p=2, dim=1)
            distances.extend(d.cpu().numpy())

    distances = np.array(distances)

    save_dir = os.path.dirname(os.path.abspath(__file__))

    if not args.no_figure:
        # Plot ROC curve
        roc_path = os.path.join(save_dir, "triplet_net_roc.png")
        mc.plot_verification_roc(test_pair_lbls.numpy(), distances, roc_path, "Triplet Network")

        # Plot embeddings cluster projections
        test_embeddings = []
        with torch.no_grad():
            for i in range(0, len(test_img), args.batch_size):
                xb = test_img[i:i+args.batch_size].to(device)
                test_embeddings.append(model(xb).cpu())
        test_embeddings = torch.cat(test_embeddings, dim=0)

        tsne_path = os.path.join(save_dir, "triplet_net_tsne.png")
        mc.plot_tsne_embeddings(
            test_embeddings, test_lbl, tsne_path,
            title="t-SNE Projection of Triplet Network Face Embeddings"
        )


if __name__ == "__main__":
    main()
