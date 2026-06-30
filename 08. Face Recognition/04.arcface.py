"""
04. ArcFace
===========

Additive Angular Margin Loss for hyperspherical classification-based metric learning (Deng et al., ArcFace, 2019).

ArcFace Loss Formulation:
    L = -log( exp( s * cos(theta_y + m) ) / ( exp( s * cos(theta_y + m) ) + sum_{j != y} exp( s * cos(theta_j) ) ) )
    where theta_j is the angle between weight vector W_j and embedding x, m is the angular margin, and s is the scale factor.

Key insights / educational takeaways:
    * Avoids the combinatorial expansion issues of pairs/triplets during batch assembly.
    * Adding margin in the angular space directly maximizes geodesic class separation on the hypersphere, yielding highly discriminative open-set face embeddings.

Run:
    python "04.arcface.py" --epochs 15
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import face_common as mc


class ArcMarginProduct(nn.Module):
    """ArcFace additive angular margin classification layer."""
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50, easy_margin: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

    def forward(self, input, label):
        # 1. Normalize weights and input features
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # 2. Calculate sine: sin(theta) = sqrt(1 - cos^2(theta))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + 1e-8)
        # 3. cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 4. Generate one-hot class mask
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        # 5. Apply angular margin only to target class index
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


def main():
    p = mc.build_argparser("ArcFace Metric Learning", epochs=15)
    args = p.parse_args()

    device = mc.get_device(args.device)

    # Load and split LFW by identities (Ensures open-set test evaluation)
    images, labels = mc.load_lfw()
    train_img, train_lbl, test_img, test_lbl = mc.split_lfw_identities(images, labels)

    # Train labels were remapped to 0..num_train_classes-1 in split_lfw_identities
    num_train_classes = int(train_lbl.max().item() + 1)

    model = mc.FaceEmbeddingNet(embedding_dim=128).to(device)
    # ArcFace classification product
    metric_fc = ArcMarginProduct(in_features=128, out_features=num_train_classes, s=30.0, m=0.50).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(metric_fc.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Training ArcFace on {num_train_classes} training identities...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable backbone params: {n_params:,}")
    print("-" * 64)

    dataset = TensorDataset(train_img, train_lbl)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        metric_fc.train()
        epoch_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            embeddings = model(x)
            # Pass embeddings and labels to the angular margin classifier
            logits = metric_fc(embeddings, y)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)

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
        roc_path = os.path.join(save_dir, "arcface_roc.png")
        mc.plot_verification_roc(test_pair_lbls.numpy(), distances, roc_path, "ArcFace")

        # Plot embeddings cluster projections
        test_embeddings = []
        with torch.no_grad():
            for i in range(0, len(test_img), args.batch_size):
                xb = test_img[i:i+args.batch_size].to(device)
                test_embeddings.append(model(xb).cpu())
        test_embeddings = torch.cat(test_embeddings, dim=0)

        tsne_path = os.path.join(save_dir, "arcface_tsne.png")
        mc.plot_tsne_embeddings(
            test_embeddings, test_lbl, tsne_path,
            title="t-SNE Projection of ArcFace Face Embeddings"
        )


if __name__ == "__main__":
    main()
