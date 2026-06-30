"""
13. Deep Graph Infomax (DGI) — self-supervised node representations (Velickovic et al., 2019)
=============================================================================================

Unsupervised representation learning on graphs by *mutual information maximization*.
No labels are used to train the encoder; instead it learns embeddings that agree
with a global summary of the graph, while a corrupted graph's embeddings do not:

    positive: H     = Encoder(X, A)          (real graph)
    negative: H~    = Encoder(shuffle(X), A) (features row-shuffled = corruption)
    summary : s     = sigmoid(mean(H))
    discriminator D(h, s) = sigmoid(h^T W s); train D to score H high, H~ low.

After this contrastive pretraining the encoder is frozen and a simple linear
classifier is fit on the embeddings — and it rivals supervised GCN on Cora.

Run:
    python "13.dgi.py" --epochs 200
"""

import os
import torch
import torch.nn as nn
import gnn_common as mc


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden)
        self.prelu = nn.PReLU()

    def forward(self, x, adj):
        return self.prelu(adj @ self.lin(x))


class DGI(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden)
        self.W = nn.Bilinear(hidden, hidden, 1)              # discriminator

    def forward(self, x, adj):
        h_pos = self.encoder(x, adj)
        x_shuf = x[torch.randperm(x.size(0), device=x.device)]   # corruption
        h_neg = self.encoder(x_shuf, adj)
        s = torch.sigmoid(h_pos.mean(dim=0, keepdim=True))       # global summary
        s_exp = s.expand_as(h_pos)
        pos = self.W(h_pos, s_exp).squeeze(-1)
        neg = self.W(h_neg, s_exp).squeeze(-1)
        return pos, neg, h_pos


def main():
    args = mc.build_argparser("Deep Graph Infomax (DGI)", epochs=200).parse_args()
    device = mc.get_device(args.device)
    features, adj, labels, splits = mc.load_cora(limit=args.limit)
    features, adj, labels = features.to(device), adj.to(device), labels.to(device)
    train_mask = splits["train_mask"].cpu().numpy()
    test_mask = splits["test_mask"].cpu().numpy()

    model = DGI(features.size(1), 256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    bce = nn.BCEWithLogitsLoss()
    print(f"Self-supervised DGI pretraining on Cora | params: {sum(q.numel() for q in model.parameters()):,}")
    print("-" * 64)
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad()
        pos, neg, _ = model(features, adj)
        loss = bce(pos, torch.ones_like(pos)) + bce(neg, torch.zeros_like(neg))
        loss.backward(); optimizer.step()
        if epoch % max(1, args.epochs // 10) == 0:
            print(f"Epoch {epoch:3d}/{args.epochs} | infomax_loss {loss.item():.4f}")
    print("-" * 64)

    # Freeze embeddings, fit a linear classifier (the standard DGI linear-probe protocol)
    model.eval()
    with torch.no_grad():
        emb = model.encoder(features, adj)
    emb_np = emb.cpu().numpy(); labels_np = labels.cpu().numpy()
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=500).fit(emb_np[train_mask], labels_np[train_mask])
        test_acc = clf.score(emb_np[test_mask], labels_np[test_mask])
        print(f"DGI linear-probe Test Accuracy: {test_acc:.4f}")
    except Exception as e:
        print(f"(skipping linear probe: {e})")

    save_dir = os.path.dirname(os.path.abspath(__file__))
    mc.plot_tsne_embeddings(emb, labels, os.path.join(save_dir, "dgi_cora_tsne.png"))


if __name__ == "__main__":
    main()
