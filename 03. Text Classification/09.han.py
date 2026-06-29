"""
09. Hierarchical Attention Network (Yang et al., 2016)
======================================================

HAN mirrors the structure of a document: words form sentences, sentences form a
document. It applies attention twice — once to pool words into a sentence vector,
again to pool sentences into a document vector — with a BiGRU encoder at each
level. The two-level attention is interpretable (which words, and which
sentences, mattered) and is well-suited to longer texts like full reviews.

Architecture Diagram / Layout:
    Input [Batch, Sentences, Words]
       word level:  Embedding -> BiGRU -> word attention  => sentence vectors [B, S, 2*Hw]
       sent level:  BiGRU -> sentence attention            => document vector [B, 2*Hs]
       -> Linear [Batch, 1]

Key insights / educational takeaways:
    * Structuring the model like the data (document -> sentences -> words) plus
      attention at each level both improves accuracy and yields a readable
      explanation of the prediction.
    * Padding is masked at both the word and sentence levels.

Run:
    python "09.han.py" --epochs 5
    python "09.han.py" --limit 1000 --epochs 2
"""

import os
import numpy as np
import torch
import torch.nn as nn
import tc_common as mc


class HAN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, word_hidden: int = 64,
                 sent_hidden: int = 64, attn_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_gru = nn.GRU(embedding_dim, word_hidden, batch_first=True, bidirectional=True)
        self.word_w = nn.Linear(2 * word_hidden, attn_dim)
        self.word_v = nn.Linear(attn_dim, 1, bias=False)
        self.sent_gru = nn.GRU(2 * word_hidden, sent_hidden, batch_first=True, bidirectional=True)
        self.sent_w = nn.Linear(2 * sent_hidden, attn_dim)
        self.sent_v = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * sent_hidden, 1)

    @staticmethod
    def _attend(H, mask, w, v):
        """Additive attention pooling over dim=1 with a padding mask."""
        scores = v(torch.tanh(w(H))).squeeze(-1)           # [N, L]
        scores = scores.masked_fill(mask, float("-inf"))
        alpha = torch.softmax(scores, dim=1)
        alpha = torch.nan_to_num(alpha)                    # fully-masked rows -> 0
        return (alpha.unsqueeze(-1) * H).sum(dim=1)        # [N, dim]

    def forward(self, x):                                   # x: [B, S, T]
        B, S, T = x.shape
        word_mask = (x == 0).view(B * S, T)
        emb = self.embedding(x.view(B * S, T))             # [B*S, T, E]
        Hw, _ = self.word_gru(emb)                         # [B*S, T, 2Hw]
        sent_vec = self._attend(Hw, word_mask, self.word_w, self.word_v)   # [B*S, 2Hw]
        sent_vec = sent_vec.view(B, S, -1)                 # [B, S, 2Hw]

        sent_mask = (x == 0).all(dim=2)                    # [B, S] fully-padded sentences
        Hs, _ = self.sent_gru(sent_vec)                    # [B, S, 2Hs]
        doc = self._attend(Hs, sent_mask, self.sent_w, self.sent_v)        # [B, 2Hs]
        return self.fc(self.dropout(doc))


def main():
    args = mc.build_argparser("Hierarchical Attention Network Classifier").parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader, tokenizer = mc.get_imdb_hierarchical_dataloaders(
        batch_size=args.batch_size, limit=args.limit)

    model = HAN(len(tokenizer.word2idx))
    mc.train_classifier(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(device)).squeeze(-1)
            y_pred.append((torch.sigmoid(out) >= 0.5).int().cpu().numpy())
            y_true.append(y.numpy())
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

    mc.report_classification(y_true, y_pred, model_name="HAN",
                             save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
