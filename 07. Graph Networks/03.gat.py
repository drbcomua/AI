"""
03. Graph Attention Network (GAT)
=================================

Anisotropic message passing with self-attention (Veličković et al., 2017).

Mathematical Formulation:
    Alpha_(i, j) = Softmax_j( LeakyReLU( a^T * [W*h_i || W*h_j] ) )
    h_i^(l+1) = Concat_k( Sigmoid( Sum_j Alpha_(i, j)^k * W^k * h_j ) )  (for multi-head)

Key insights / educational takeaways:
    * Graph Attention allows dynamically weighting neighborhood edges, unlike GCN's static degree weighting.
    * Multi-head attention stabilizes the learning process and reduces noisy attention spikes.

Run:
    python "03.gat.py" --epochs 200
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gnn_common as mc


class GATConv(nn.Module):
    """Multi-head Graph Attention Layer in pure PyTorch using batched matrix operations."""
    def __init__(self, in_features: int, out_features: int, heads: int = 8, concat: bool = True, dropout: float = 0.6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat

        # Projection weights
        self.linear = nn.Linear(in_features, heads * out_features, bias=False)

        # Learnable attention vectors: [heads, out_features, 1]
        self.attn_src = nn.Parameter(torch.zeros(heads, out_features, 1))
        self.attn_dst = nn.Parameter(torch.zeros(heads, out_features, 1))

        # Initialization
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x shape: [N, in_features], adj shape: [N, N]
        N = x.size(0)
        # Linear projection: [N, heads * out_features] -> [N, heads, out_features]
        h = self.linear(x).view(N, self.heads, self.out_features)

        # Permute to head-first batches: [heads, N, out_features]
        h_trans = h.permute(1, 0, 2)

        # Query projections: [heads, N, 1]
        s = torch.bmm(h_trans, self.attn_src)
        d = torch.bmm(h_trans, self.attn_dst)

        # Compute pairwise logits: [heads, N, N]
        e = s + d.transpose(1, 2)
        e = F.leaky_relu(e, negative_slope=0.2)

        # Neighborhood mask: set non-edge attention coefficients to -inf
        mask = (adj == 0).unsqueeze(0) # [1, N, N]
        e = e.masked_fill(mask, float('-inf'))

        # Softmax over neighbor choices
        alpha = torch.softmax(e, dim=-1)
        alpha = self.dropout(alpha)

        # Aggregate: [heads, N, N] @ [heads, N, out_features] -> [heads, N, out_features]
        out = torch.bmm(alpha, h_trans)
        out = out.permute(1, 0, 2) # [N, heads, out_features]

        if self.concat:
            return out.contiguous().view(N, self.heads * self.out_features)
        else:
            return out.mean(dim=1)


class GAT(nn.Module):
    """2-layer Graph Attention Network."""
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, heads: int = 8, dropout: float = 0.6):
        super().__init__()
        # First layer: Multi-head attention (e.g. 8 heads of 8 features -> 64 dimensions output)
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        # Second layer: Single-head classification average
        self.conv2 = GATConv(hidden_dim * heads, num_classes, heads=1, concat=False, dropout=dropout)

    def forward(self, x, adj):
        h = F.elu(self.conv1(x, adj))
        logits = self.conv2(h, adj)
        return logits, h


def main():
    p = mc.build_argparser("Graph Attention Network (GAT)", epochs=200)
    args = p.parse_args()

    device = mc.get_device(args.device)

    # Load Cora
    features, adj, labels, splits = mc.load_cora(limit=args.limit)

    in_dim = features.size(1)
    num_classes = int(labels.max().item() + 1)

    # Kipf/Veličković GAT parameters: 8 heads, 8 hidden features
    model = GAT(in_dim, 8, num_classes, heads=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Move tensors to device
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    train_mask = splits["train_mask"].to(device)
    val_mask = splits["val_mask"].to(device)
    test_mask = splits["test_mask"].to(device)

    print("Training Graph Attention Network (GAT)...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        out, _ = model(features, adj)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits, _ = model(features, adj)
            val_loss = criterion(logits[val_mask], labels[val_mask])
            preds = logits.argmax(dim=-1)
            val_acc = (preds[val_mask] == labels[val_mask]).float().mean().item()

        if epoch % max(1, args.epochs // 10) == 0:
            print(f"Epoch {epoch:3d}/{args.epochs} | train_loss {loss.item():.4f} | val_loss {val_loss.item():.4f} | val_acc {val_acc:.4f}")

    print("-" * 64)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        logits, h_emb = model(features, adj)
        preds = logits.argmax(dim=-1)
        test_acc = (preds[test_mask] == labels[test_mask]).float().mean().item()
    print(f"GAT Test Accuracy: {test_acc:.4f}")

    # Plot and save t-SNE of hidden node embeddings
    save_dir = os.path.dirname(os.path.abspath(__file__))
    tsne_path = os.path.join(save_dir, "gat_cora_tsne.png")
    mc.plot_tsne_embeddings(h_emb, labels, tsne_path)


if __name__ == "__main__":
    main()
