"""
06. Attention-Pooled BiLSTM (Bahdanau-style additive attention)
===============================================================

The recurrent classifier in 03 summarizes a review with its *last* hidden state,
forcing one vector to remember everything. This model instead keeps every
timestep's hidden state and learns an **attention** distribution over them — a
weighted average where the network decides which words matter most. It's the
conceptual bridge from RNNs to the all-attention Transformer, and the weights are
interpretable.

Architecture Diagram / Layout:
    Input [Batch, Seq_Len]
       -> Embedding [Batch, Seq_Len, Emb]
       -> BiLSTM -> hidden states H [Batch, Seq_Len, 2*Hidden]
       -> Additive attention: score = v^T tanh(W H); alpha = softmax(score)
          (padding positions masked to -inf before softmax)
       -> Context = sum(alpha * H) [Batch, 2*Hidden]
       -> Linear [Batch, 1]

Key insights / educational takeaways:
    * Attention removes the recurrent bottleneck of a single summary vector.
    * The learned alpha weights highlight the words driving the prediction.

Run:
    python "06.attention-bilstm.py" --epochs 5
    python "06.attention-bilstm.py" --limit 1000 --epochs 2
"""

import os
import numpy as np
import torch
import torch.nn as nn
import tc_common as mc


class AttentionBiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden: int = 128,
                 attn_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden, batch_first=True, bidirectional=True)
        self.attn_w = nn.Linear(2 * hidden, attn_dim)
        self.attn_v = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, 1)

    def forward(self, x):                                   # x: [B, L]
        mask = (x == 0)                                    # padding
        H, _ = self.lstm(self.embedding(x))                # [B, L, 2H]
        scores = self.attn_v(torch.tanh(self.attn_w(H))).squeeze(-1)   # [B, L]
        scores = scores.masked_fill(mask, float("-inf"))   # ignore padding
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, L, 1]
        context = (alpha * H).sum(dim=1)                    # [B, 2H]
        return self.fc(self.dropout(context))


def main():
    args = mc.build_argparser("Attention-Pooled BiLSTM Sentiment Classifier").parse_args()
    vocab_size, max_len = 5000, 150
    device = mc.get_device(args.device)

    train_loader, test_loader, tokenizer = mc.get_imdb_dataloaders(
        batch_size=args.batch_size, limit=args.limit, vocab_size=vocab_size, max_len=max_len)

    model = AttentionBiLSTM(len(tokenizer.word2idx))
    mc.train_classifier(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(device)).squeeze(-1)
            y_pred.append((torch.sigmoid(out) >= 0.5).int().cpu().numpy())
            y_true.append(y.numpy())
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

    mc.report_classification(y_true, y_pred, model_name="Attention-BiLSTM",
                             save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
