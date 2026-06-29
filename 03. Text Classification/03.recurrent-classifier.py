"""
03. Recurrent Sentiment Classifier
==================================

Recurrent neural network models for text sentiment classification.

Architecture Diagram / Layout:
    Input [Batch, Seq_Len]
       -> Embedding Layer [Batch, Seq_Len, Embedding_Dim]
       -> Bidirectional LSTM / GRU [Batch, Seq_Len, Hidden_Dim * 2]
       -> Gather Final Hidden States [Batch, Hidden_Dim * 2]
       -> Fully Connected Layer [Batch, 1]

Key insights / educational takeaways:
    * Bidirectional RNNs process context in both forward and backward directions, capturing surrounding semantic clues.
    * Extracting the final sequence step represents the global contextual state.

Run:
    python "03.recurrent-classifier.py" --variant lstm --epochs 5
    python "03.recurrent-classifier.py" --variant gru --epochs 5
"""

import os
import torch
import torch.nn as nn
import tc_common as mc


class RecurrentClassifier(nn.Module):
    """Sequence-aware classifier utilizing bidirectional LSTMs or GRUs."""
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 64, cell_type: str = "lstm"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.cell_type = cell_type.lower()

        if self.cell_type == "lstm":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,
                               bidirectional=True, batch_first=True)
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=1,
                              bidirectional=True, batch_first=True)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

        # bidirectional doubles hidden_dim size
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x shape: [Batch, Seq_Len]
        embedded = self.embedding(x) # [Batch, Seq_Len, Embedding_Dim]

        # RNN Forward pass
        out, _ = self.rnn(embedded) # out shape: [Batch, Seq_Len, Hidden_Dim * 2]

        # Gather final hidden state (last sequence step outputs)
        last_step = out[:, -1, :] # [Batch, Hidden_Dim * 2]
        return self.fc(last_step)


def main():
    p = mc.build_argparser("Recurrent Sentiment Classifiers")
    args = p.parse_args()

    vocab_size = 5000
    max_len = 150
    device = mc.get_device(args.device)

    train_loader, test_loader, tokenizer = mc.get_imdb_dataloaders(
        batch_size=args.batch_size, limit=args.limit, vocab_size=vocab_size, max_len=max_len
    )

    variant = args.variant or "lstm"
    if variant not in ["lstm", "gru"]:
        variant = "lstm"

    model = RecurrentClassifier(vocab_size=len(tokenizer.word2idx), cell_type=variant)
    model_name = f"Recurrent-{variant.upper()}"

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
