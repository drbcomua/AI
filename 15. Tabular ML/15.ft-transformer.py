"""
15. FT-Transformer & Tabular ResNet (Gorishniy et al., 2021)
============================================================

"Revisiting Deep Learning Models for Tabular Data" proposed two strong, simple
baselines that remain the deep-tabular models to beat. This script implements
both, selectable with `--variant`:

  ft-transformer : *Feature Tokenizer + Transformer*. Every feature (numeric or
                   categorical) is turned into its own d-dimensional token; a
                   learnable [CLS] token is prepended; a standard Transformer
                   encoder lets features attend to one another; the final [CLS]
                   embedding is read out through a linear head. Attention over
                   feature tokens is the key idea.
  resnet         : the same paper's tabular ResNet — an MLP with pre-norm
                   residual blocks. Much cheaper, and a genuinely tough baseline.

Architecture Diagram / Layout (ft-transformer):
    x [N, F]
      -> FeatureTokenizer:  token_f = x_f * W_f + b_f            [N, F, d]
      -> prepend [CLS]                                           [N, F+1, d]
      -> L x TransformerEncoderLayer (MHSA + FFN, pre-norm)      [N, F+1, d]
      -> take CLS token -> LayerNorm -> Linear                   -> logits [N, C]

Architecture Diagram / Layout (resnet):
    x -> Linear(F -> d) -> [ x + Drop(Lin(ReLU(Lin(BN(x))))) ] * B -> Norm -> Linear -> logits

Simplification (documented): all features are treated as numeric (standardized);
a full FT-Transformer would tokenize categoricals with embeddings — see script
12 for that mechanism.

Key insights / educational takeaways:
    * Per-feature tokenization lets self-attention model interactions between
      features explicitly, which is what gives the Transformer its edge over a
      plain MLP on some tabular problems.
    * The ResNet variant shows most of the benefit often comes from good
      normalization + residual connections, not attention — always benchmark the
      cheap variant before paying for the expensive one.

Run:
    python "15.ft-transformer.py" --variant ft-transformer --dataset covtype --epochs 30
    python "15.ft-transformer.py" --variant resnet --dataset covtype
    python "15.ft-transformer.py" --limit 2000 --epochs 2        # fast smoke test
"""

import os
import torch
import torch.nn as nn
import tabular_common as mc


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight + self.bias      # (B, F, d)
        cls = self.cls.expand(x.size(0), -1, -1)                # (B, 1, d)
        return torch.cat([cls, tokens], dim=1)                  # (B, F+1, d)


class FTTransformer(nn.Module):
    def __init__(self, n_features, n_classes, d=64, n_layers=3, n_heads=8):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=2 * d,
            dropout=0.1, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, n_classes))

    def forward(self, x):
        tokens = self.encoder(self.tokenizer(x))
        return self.head(tokens[:, 0])                          # CLS readout


class ResBlock(nn.Module):
    def __init__(self, d, hidden, dropout=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(d)
        self.lin1 = nn.Linear(d, hidden)
        self.lin2 = nn.Linear(hidden, d)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.lin2(self.drop(self.act(self.lin1(self.bn(x)))))
        return x + h


class TabResNet(nn.Module):
    def __init__(self, n_features, n_classes, d=128, n_blocks=3):
        super().__init__()
        self.first = nn.Linear(n_features, d)
        self.blocks = nn.ModuleList([ResBlock(d, 2 * d) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, n_classes))

    def forward(self, x):
        h = self.first(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h)


def main():
    p = mc.build_argparser("FT-Transformer / Tabular ResNet", lr=1e-3)
    p.add_argument("--variant", choices=["ft-transformer", "resnet"],
                   default="ft-transformer")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=3)
    args = p.parse_args()
    mc.set_seed(args.seed)
    device = mc.get_device(args.device)

    X_train, X_test, y_train, y_test, feature_names, class_names = \
        mc.load_classification_dataset(args.dataset)
    if args.limit is not None:
        X_train, y_train = X_train[:args.limit], y_train[:args.limit]
    Xtr, Xte = mc.standardize(X_train, X_test)

    train_loader, test_loader = mc.get_dataloaders_from_arrays(
        Xtr, y_train, Xte, y_test, batch_size=args.batch_size)

    if args.variant == "ft-transformer":
        model = FTTransformer(Xtr.shape[1], len(class_names), d=args.d_model,
                              n_layers=args.n_layers)
    else:
        model = TabResNet(Xtr.shape[1], len(class_names), d=2 * args.d_model,
                          n_blocks=args.n_layers)

    mc.train(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr,
             device=device)
    y_true, y_pred, y_prob = mc.evaluate(model, test_loader, device)
    nice_name = "FT-Transformer" if args.variant == "ft-transformer" else "Tabular-ResNet"
    mc.report_classification(y_true, y_pred, y_prob, class_names=class_names,
                             model_name=nice_name,
                             save_dir=None if args.no_figure else
                             os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
