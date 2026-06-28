"""
11. iTransformer (Liu et al., 2024 — "Inverted Transformers Are Effective for Time Series")
==========================================================================================

iTransformer makes one deceptively simple change to the vanilla transformer:
**invert the axes**. Instead of treating each *timestep* as a token (and mixing
variables inside the embedding), it treats each *variable's entire series* as a
token and applies attention *across variables*.

Architecture Diagram / Layout:
    Input [B, W, F] -> transpose -> [B, F, W]
       -> Linear(W -> d) embeds each variate's series as a token  [B, F, d]
       -> Transformer Encoder (attention ACROSS the F variate tokens)
       -> Linear(d -> 1) per variate -> [B, F] -> Linear(F -> 1) -> [B, 1]

Key insights / educational takeaways:
    * Attention now models *correlations between variables* (e.g. pressure vs.
      temperature vs. humidity), which is exactly what a multivariate forecast
      needs — try it on `--dataset jena_full` (14 variables) to see it shine.
    * Each variate token sees the whole lookback at once, so the time axis is
      handled by a plain linear embedding rather than positional attention.

Run:
    python "11.itransformer.py" --dataset jena_full --epochs 3
    python "11.itransformer.py" --dataset jena --epochs 5
"""

import os
import torch.nn as nn
import ts_common as mc


class ITransformer(nn.Module):
    def __init__(self, num_features: int, seq_len: int = 24, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(seq_len, d_model)          # each variate's series -> token
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, 1)                 # per-variate forecast
        self.head = nn.Linear(num_features, 1)            # combine variate forecasts

    def forward(self, x):                                 # [B, W, F]
        tokens = self.embed(x.permute(0, 2, 1))           # [B, F, d_model]
        enc = self.encoder(tokens)                        # attention across F variates
        per_variate = self.proj(enc).squeeze(-1)          # [B, F]
        return self.head(per_variate)                     # [B, 1]


def main():
    args = mc.build_argparser("iTransformer Forecaster").parse_args()
    W = 24
    device = mc.get_device(args.device)

    train_loader, test_loader, num_features, mean, std = mc.get_dataloaders(
        args.dataset, args.batch_size, args.limit, W=W)

    model = ITransformer(num_features, W)
    mc.train(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)
    y_true, y_pred = mc.evaluate(model, test_loader, device=device)
    mc.report(y_true, y_pred, mean, std, dataset_name=args.dataset, model_name="iTransformer",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
