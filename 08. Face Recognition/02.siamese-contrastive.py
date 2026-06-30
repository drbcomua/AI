"""
02. Siamese Network with Contrastive Loss
=========================================

Twin weight-sharing networks optimized via Contrastive Loss (Chopra et al., 2005).

Contrastive Loss Formulation:
    L = (1 - y) * 0.5 * d^2 + y * 0.5 * max(0, margin - d)^2
    where d = ||f(x1) - f(x2)||_2, y = 0 if same person, y = 1 if different people.

Key insights / educational takeaways:
    * The twin networks share identical weights to project both face images into a shared metric embedding space.
    * Contrastive loss pulls matching pairs together while pushing non-matching pairs apart up to a margin threshold.

Run:
    python "02.siamese-contrastive.py" --epochs 15
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import face_common as mc


class ContrastiveLoss(nn.Module):
    """Contrastive loss function for pairwise embeddings."""
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, x1, x2, y):
        # Euclidean distance
        d = torch.norm(x1 - x2, p=2, dim=1)
        # y = 0 for matching (same), y = 1 for non-matching (different)
        loss = (1.0 - y.float()) * 0.5 * torch.pow(d, 2) + \
               y.float() * 0.5 * torch.pow(torch.clamp(self.margin - d, min=0.0), 2)
        return loss.mean()


def main():
    p = mc.build_argparser("Siamese Network with Contrastive Loss", epochs=15)
    args = p.parse_args()

    device = mc.get_device(args.device)

    # Load and split LFW by identities (Ensures open-set test evaluation)
    images, labels = mc.load_lfw()
    train_img, train_lbl, test_img, test_lbl = mc.split_lfw_identities(images, labels)

    model = mc.FaceEmbeddingNet(embedding_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = ContrastiveLoss(margin=1.0)

    print("Training Siamese Network with Contrastive Loss...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Generate fresh training pairs dynamically each epoch for data augmentation
        # Let's generate 4000 pairs per epoch (matching and non-matching)
        p1, p2, pair_lbls = mc.generate_verification_pairs(train_img, train_lbl, num_pairs=4000)

        # In generate_verification_pairs: 1 = same, 0 = different
        # In Contrastive Loss: 0 = same (matching), 1 = different (non-matching)
        # So we convert: y = 1 - pair_lbls
        loss_lbls = 1 - pair_lbls

        dataset = TensorDataset(p1, p2, loss_lbls)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        epoch_loss = 0.0
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            optimizer.zero_grad()
            emb1 = model(x1)
            emb2 = model(x2)
            loss = criterion(emb1, emb2, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x1.size(0)

        epoch_loss_avg = epoch_loss / len(dataset)
        print(f"Epoch {epoch:2d}/{args.epochs} | loss: {epoch_loss_avg:.4f}")

    print("-" * 64)

    # Evaluation on unseen test identities
    print("Generating unseen verification test pairs...")
    test_p1, test_p2, test_pair_lbls = mc.generate_verification_pairs(test_img, test_lbl, num_pairs=1000)

    model.eval()
    distances = []
    with torch.no_grad():
        # Process in batches to avoid GPU memory overflow
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
        roc_path = os.path.join(save_dir, "siamese_contrastive_roc.png")
        mc.plot_verification_roc(test_pair_lbls.numpy(), distances, roc_path, "Siamese Contrastive")

        # Plot embeddings cluster projections
        # Extract embeddings of test images
        test_embeddings = []
        with torch.no_grad():
            for i in range(0, len(test_img), args.batch_size):
                xb = test_img[i:i+args.batch_size].to(device)
                test_embeddings.append(model(xb).cpu())
        test_embeddings = torch.cat(test_embeddings, dim=0)

        tsne_path = os.path.join(save_dir, "siamese_contrastive_tsne.png")
        mc.plot_tsne_embeddings(
            test_embeddings, test_lbl, tsne_path,
            title="t-SNE Projection of Siamese Contrastive Face Embeddings"
        )


if __name__ == "__main__":
    main()
