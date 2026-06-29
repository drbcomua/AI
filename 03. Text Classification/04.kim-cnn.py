"""
04. CNN for Text Classification (Kim CNN)
=========================================

1D Convolutional Neural Network for sentence classification (Kim, 2014).

Architecture Diagram / Layout:
    Input [Batch, Seq_Len]
       -> Embedding [Batch, Seq_Len, Embedding_Dim]
       -> Transpose [Batch, Embedding_Dim, Seq_Len]
       -> Parallel Convolutions:
            * Conv1d (kernel=3) -> ReLU -> MaxPool [Batch, Num_Filters]
            * Conv1d (kernel=4) -> ReLU -> MaxPool [Batch, Num_Filters]
            * Conv1d (kernel=5) -> ReLU -> MaxPool [Batch, Num_Filters]
       -> Concatenate filters [Batch, Num_Filters * 3]
       -> Dropout
       -> Fully Connected [Batch, 1]

Key insights / educational takeaways:
    * Parallel convolutional filters extract local n-gram patterns of different lengths.
    * Max-pooling-over-time captures the most salient semantic feature within the entire sequence, making the model invariant to position.

Run:
    python "04.kim-cnn.py" --epochs 5
    python "04.kim-cnn.py" --limit 1000 --epochs 2
"""

import os
import torch
import torch.nn as nn
import tc_common as mc


class KimCNN(nn.Module):
    """Parallel Conv1D feature extractors for text classification."""
    def __init__(self, vocab_size: int, embedding_dim: int = 128, num_filters: int = 100,
                 filter_sizes: list[int] = [3, 4, 5], dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Parallel Conv1d layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 1)

    def forward(self, x):
        # x shape: [Batch, Seq_Len]
        embedded = self.embedding(x) # [Batch, Seq_Len, Embedding_Dim]
        # Conv1d expects [Batch, Channels, Seq_Len]
        embedded = embedded.transpose(1, 2)

        pooled_outputs = []
        for conv in self.convs:
            c_out = conv(embedded) # [Batch, Num_Filters, Seq_Len - fs + 1]
            activated = torch.relu(c_out)
            # Max pooling over the temporal dimension (last dimension)
            pooled, _ = torch.max(activated, dim=2) # [Batch, Num_Filters]
            pooled_outputs.append(pooled)

        # Concatenate outputs along the filter channel dimension
        flat = torch.cat(pooled_outputs, dim=1) # [Batch, Num_Filters * len(filter_sizes)]
        out = self.dropout(flat)
        return self.fc(out)


def main():
    p = mc.build_argparser("Kim CNN Sentiment Classifier")
    args = p.parse_args()

    vocab_size = 5000
    max_len = 150
    device = mc.get_device(args.device)

    train_loader, test_loader, tokenizer = mc.get_imdb_dataloaders(
        batch_size=args.batch_size, limit=args.limit, vocab_size=vocab_size, max_len=max_len
    )

    model = KimCNN(vocab_size=len(tokenizer.word2idx), filter_sizes=[3, 4, 5], num_filters=100)
    model_name = "Kim-CNN"

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
        y_true, y_pred, model_name=model_name,
        save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__))
    )


if __name__ == "__main__":
    main()
