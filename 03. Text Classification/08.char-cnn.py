"""
08. Character-level CNN (Zhang, Zhao & LeCun, 2015)
===================================================

Instead of treating words as the atomic unit, this model reads the *raw character
stream* and lets convolutions learn morphology, spelling, and n-gram cues from
scratch. There is no word vocabulary at all — robust to typos, rare words, and
made-up slang, at the cost of much longer sequences.

Architecture Diagram / Layout:
    Input [Batch, Char_Len]  (default 400 chars; alphabet of ~50 symbols)
       -> Char Embedding [Batch, Char_Len, Emb] -> transpose to [Batch, Emb, Char_Len]
       -> Conv1d(k=7) -> ReLU -> MaxPool(3)
       -> Conv1d(k=7) -> ReLU -> MaxPool(3)
       -> Conv1d(k=3) -> ReLU
       -> Conv1d(k=3) -> ReLU -> Adaptive Max-pool
       -> FC -> ReLU -> Dropout -> FC [Batch, 1]

Key insights / educational takeaways:
    * Language can be modeled with no notion of "words" — character convolutions
      suffice, trading vocabulary for sequence length.
    * Compact, faithful version of the paper (smaller widths/length for speed).

Run:
    python "08.char-cnn.py" --epochs 5
    python "08.char-cnn.py" --limit 1000 --epochs 2
"""

import os
import numpy as np
import torch
import torch.nn as nn
import tc_common as mc


class CharCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 16, num_filters: int = 128,
                 dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.features = nn.Sequential(
            nn.Conv1d(embedding_dim, num_filters, 7, padding=3), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(num_filters, num_filters, 7, padding=3), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(num_filters, num_filters, 3, padding=1), nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, 3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):                                   # x: [B, Char_Len]
        e = self.embedding(x).transpose(1, 2)              # [B, Emb, Char_Len]
        return self.classifier(self.features(e))


def main():
    p = mc.build_argparser("Character-level CNN Sentiment Classifier")
    p.add_argument("--char-len", type=int, default=400, help="characters per review")
    args = p.parse_args()
    device = mc.get_device(args.device)

    train_loader, test_loader, char_tok = mc.get_imdb_char_dataloaders(
        batch_size=args.batch_size, limit=args.limit, max_len=args.char_len)

    model = CharCNN(char_tok.vocab_size)
    mc.train_classifier(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(device)).squeeze(-1)
            y_pred.append((torch.sigmoid(out) >= 0.5).int().cpu().numpy())
            y_true.append(y.numpy())
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

    mc.report_classification(y_true, y_pred, model_name="Char-CNN",
                             save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
