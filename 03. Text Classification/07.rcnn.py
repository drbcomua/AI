"""
07. RCNN — Recurrent Convolutional Neural Network (Lai et al., 2015)
===================================================================

RCNN fuses the two preceding ideas. A bidirectional RNN gives every word a
*context-aware* representation (what came before and after), which is concatenated
with the word's own embedding. A shared linear "semantic" layer plus a tanh acts
like a 1-word convolution over these enriched vectors, and a global max-pool-over-
time picks the most salient features — combining RNN context with CNN-style
pooling, with no fixed filter window.

Architecture Diagram / Layout:
    Input [Batch, Seq_Len]
       -> Embedding e [Batch, Seq_Len, Emb]
       -> BiLSTM -> context [Batch, Seq_Len, 2*Hidden]
       -> concat([context, e]) [Batch, Seq_Len, 2*Hidden + Emb]
       -> Linear -> tanh  (latent semantic vector per word) [Batch, Seq_Len, K]
       -> Max-pool over time [Batch, K]
       -> Linear [Batch, 1]

Key insights / educational takeaways:
    * The recurrent context window is unbounded (unlike CNN's fixed kernel), while
      max-pooling-over-time keeps the position-invariance that makes CNNs strong.
    * A clean illustration that RNN and CNN inductive biases are complementary.

Run:
    python "07.rcnn.py" --epochs 5
    python "07.rcnn.py" --limit 1000 --epochs 2
"""

import os
import numpy as np
import torch
import torch.nn as nn
import tc_common as mc


class RCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden: int = 128,
                 latent: int = 128, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden, batch_first=True, bidirectional=True)
        self.semantic = nn.Linear(2 * hidden + embedding_dim, latent)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(latent, 1)

    def forward(self, x):                                   # x: [B, L]
        emb = self.embedding(x)                             # [B, L, E]
        context, _ = self.lstm(emb)                         # [B, L, 2H]
        combined = torch.cat([context, emb], dim=2)         # [B, L, 2H+E]
        latent = torch.tanh(self.semantic(combined))        # [B, L, K]
        pooled, _ = latent.max(dim=1)                       # max-over-time [B, K]
        return self.fc(self.dropout(pooled))


def main():
    args = mc.build_argparser("RCNN Sentiment Classifier").parse_args()
    vocab_size, max_len = 5000, 150
    device = mc.get_device(args.device)

    train_loader, test_loader, tokenizer = mc.get_imdb_dataloaders(
        batch_size=args.batch_size, limit=args.limit, vocab_size=vocab_size, max_len=max_len)

    model = RCNN(len(tokenizer.word2idx))
    mc.train_classifier(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(device)).squeeze(-1)
            y_pred.append((torch.sigmoid(out) >= 0.5).int().cpu().numpy())
            y_true.append(y.numpy())
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

    mc.report_classification(y_true, y_pred, model_name="RCNN",
                             save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
