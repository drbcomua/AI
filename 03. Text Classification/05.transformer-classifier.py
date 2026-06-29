"""
05. Transformer Encoder Classifier (Vaswani et al., 2017)
========================================================

The architecture that reshaped NLP, applied here to sentiment: a stack of
self-attention encoder layers reads the whole review at once, letting every word
attend to every other word (no recurrence, fully parallel). A learnable [CLS]
token aggregates the sequence into a single vector for classification — exactly
the recipe BERT later scaled up with pretraining.

Architecture Diagram / Layout:
    Input [Batch, Seq_Len]
       -> Embedding [Batch, Seq_Len, d_model] (+ prepend learnable [CLS])
       -> Positional Encoding
       -> N x Transformer Encoder Layers (multi-head self-attention + FFN)
          (padding positions masked out of attention)
       -> Take [CLS] hidden state [Batch, d_model]
       -> Linear [Batch, 1]

Key insights / educational takeaways:
    * Self-attention models long-range dependencies in O(1) path length, unlike
      RNNs — any two words interact directly.
    * The padding mask keeps attention from "reading" <pad> tokens.
    * From scratch on small data it rivals (not crushes) the CNN/RNN here; its
      real power shows once pretrained (see 10.bert.py).

Run:
    python "05.transformer-classifier.py" --epochs 5
    python "05.transformer-classifier.py" --limit 1000 --epochs 2
"""

import math
import os
import numpy as np

# Enable MPS fallback for missing operations (like grid_sampler_2d_backward)
# This MUST be set before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import tc_common as mc


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.2,
                 max_len: int = 150):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model, max_len + 1)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_feedforward,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):                                   # x: [B, L]
        pad_mask = (x == 0)                                # True where padding
        emb = self.embedding(x)                            # [B, L, d]
        cls = self.cls_token.expand(x.size(0), -1, -1)     # [B, 1, d]
        h = self.pos_encoder(torch.cat([cls, emb], dim=1))
        # CLS is never masked; concat a False column for it.
        cls_mask = torch.zeros(x.size(0), 1, dtype=torch.bool, device=x.device)
        mask = torch.cat([cls_mask, pad_mask], dim=1)
        out = self.encoder(h, src_key_padding_mask=mask)
        return self.fc(out[:, 0])                          # [CLS] -> logit


def main():
    args = mc.build_argparser("Transformer Encoder Sentiment Classifier").parse_args()
    vocab_size, max_len = 5000, 150
    device = mc.get_device(args.device)

    train_loader, test_loader, tokenizer = mc.get_imdb_dataloaders(
        batch_size=args.batch_size, limit=args.limit, vocab_size=vocab_size, max_len=max_len)

    model = TransformerClassifier(len(tokenizer.word2idx), max_len=max_len)
    mc.train_classifier(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(device)).squeeze(-1)
            y_pred.append((torch.sigmoid(out) >= 0.5).int().cpu().numpy())
            y_true.append(y.numpy())
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

    mc.report_classification(y_true, y_pred, model_name="Transformer",
                             save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
