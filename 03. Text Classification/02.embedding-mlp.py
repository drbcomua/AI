"""
02. Embedding MLP Classifier
============================

Continuous vector space representations paired with multi-layer perceptrons.

Architecture Diagram / Layout:
    Input [Batch, Seq_Len] 
       -> Embedding Layer [Batch, Seq_Len, Embedding_Dim]
       -> Global Average Pooling [Batch, Embedding_Dim]
       -> Dense Layer [Batch, Hidden] -> ReLU
       -> Dropout
       -> Dense Layer [Batch, 1] (outputs logits)

Run:
    python "02.embedding-mlp.py" --epochs 5
    python "02.embedding-mlp.py" --limit 1000 --epochs 2
"""

import os
import torch
import torch.nn as nn
import tc_common as mc


class EmbeddingMLP(nn.Module):
    """Dense embedding mapping with global pooling and dense regression head."""
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x shape: [Batch, Seq_Len]
        embedded = self.embedding(x) # [Batch, Seq_Len, Embedding_Dim]
        # Average pooling along the sequence length (dimension 1)
        pooled = embedded.mean(dim=1) # [Batch, Embedding_Dim]
        return self.fc_net(pooled)


def main():
    p = mc.build_argparser("Embedding MLP Sentiment Classifier")
    args = p.parse_args()

    vocab_size = 5000
    max_len = 150
    device = mc.get_device(args.device)

    print("Preparing IMDB dataloaders...")
    train_loader, test_loader, tokenizer = mc.get_imdb_dataloaders(
        batch_size=args.batch_size, limit=args.limit, vocab_size=vocab_size, max_len=max_len
    )

    model = EmbeddingMLP(vocab_size=len(tokenizer.word2idx), embedding_dim=128, hidden_dim=64)

    # Train
    mc.train_classifier(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)

    # Evaluate
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x).squeeze(-1)
            preds = (torch.sigmoid(out) >= 0.5).int().cpu().numpy()
            y_true.append(y.numpy())
            y_pred.append(preds)

    import numpy as np
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Report
    mc.report_classification(
        y_true, y_pred, model_name="Embedding-MLP",
        save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__))
    )


if __name__ == "__main__":
    main()
