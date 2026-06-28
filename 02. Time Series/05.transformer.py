"""
05. Time-Series Transformers (Vanilla & PatchTST)
=================================================

Attention-based sequence models for time-series forecasting.

Architecture Diagram / Layout (Vanilla):
    Input [Batch, W, Features] -> Linear Projection [Batch, W, d_model]
       -> Add Positional Encoding [Batch, W, d_model]
       -> Transformer Encoder Layers [Batch, W, d_model]
       -> Extract Last Step [Batch, d_model]
       -> Linear [Batch, d_model -> 1] -> Output [Batch, 1]

Architecture Diagram / Layout (PatchTST):
    Input [Batch, W, Features] -> Slice Patches [Batch, Num_Patches, Patch_Len * Features]
       -> Linear Projection [Batch, Num_Patches, d_model]
       -> Add Positional Encoding [Batch, Num_Patches, d_model]
       -> Transformer Encoder Layers [Batch, Num_Patches, d_model]
       -> Average Pool [Batch, d_model]
       -> Linear [Batch, d_model -> 1] -> Output [Batch, 1]

Key insights / educational takeaways:
    * Vanilla step-level transformers calculate attention over individual sequence steps, which is computationally expensive and can lose temporal context.
    * PatchTST groups adjacent time-steps into overlapping "patches", projecting them to token vectors. This reduces sequence length, dampens noise, and models local correlations.

Run:
    python "05.transformer.py" --dataset jena --variant vanilla --epochs 5
    python "05.transformer.py" --dataset spy --variant patch --epochs 5
"""

import os
import math
import torch
import torch.nn as nn
import ts_common as mc


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence modeling."""
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # [1, max_len, d_model]

    def forward(self, x):
        # x shape: [Batch, Seq_Len, d_model]
        return x + self.pe[:, :x.size(1)]


class VanillaTransformerForecaster(nn.Module):
    """Vanilla step-level transformer forecaster."""
    def __init__(self, num_features: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [Batch, W, Features]
        x_proj = self.proj(x) # [Batch, W, d_model]
        x_pe = self.pos_encoder(x_proj)
        # Apply self-attention
        out = self.transformer(x_pe) # [Batch, W, d_model]
        # Gather last encoded sequence step
        last_step = out[:, -1, :] # [Batch, d_model]
        return self.fc(last_step)


class PatchTSTForecaster(nn.Module):
    """Simplified PatchTST transformer forecaster."""
    def __init__(self, num_features: int, seq_len: int = 24, patch_len: int = 8, stride: int = 4,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.seq_len = seq_len

        # Number of patches calculation
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Project patch dimensions to d_model
        self.proj = nn.Linear(patch_len * num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [Batch, W, Features]
        batch_size = x.size(0)

        # Slice patches along the sequence dimension (W)
        patches = []
        for i in range(0, self.seq_len - self.patch_len + 1, self.stride):
            # Extract window segment: [Batch, patch_len, Features]
            patch = x[:, i : i + self.patch_len, :]
            # Flatten to [Batch, patch_len * Features]
            patches.append(patch.flatten(start_dim=1))

        # Stack patches: [Batch, num_patches, patch_len * Features]
        patches = torch.stack(patches, dim=1)

        # Project to transformer hidden dimensions
        x_proj = self.proj(patches) # [Batch, num_patches, d_model]
        x_pe = self.pos_encoder(x_proj)

        # Pass through transformer encoder
        out = self.transformer(x_pe) # [Batch, num_patches, d_model]

        # Average pool across patch token representations
        pooled = out.mean(dim=1) # [Batch, d_model]
        return self.fc(pooled)


def main():
    p = mc.build_argparser("Time-Series Transformer Forecasters")
    args = p.parse_args()

    W = 24
    device = mc.get_device(args.device)

    train_loader, test_loader, num_features, mean, std = mc.get_dataloaders(
        args.dataset, args.batch_size, args.limit, W=W
    )

    variant = args.variant or "vanilla"

    if variant == "vanilla":
        model = VanillaTransformerForecaster(num_features=num_features)
        model_name = "VanillaTransformer"
    elif variant == "patch":
        model = PatchTSTForecaster(num_features=num_features, seq_len=W, patch_len=8, stride=4)
        model_name = "PatchTST"
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Train
    mc.train(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)

    # Evaluate
    y_true, y_pred = mc.evaluate(model, test_loader, device=device)

    # Report
    mc.report(y_true, y_pred, mean, std, dataset_name=args.dataset,
              model_name=model_name,
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
