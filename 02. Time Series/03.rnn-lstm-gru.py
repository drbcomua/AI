"""
03. Recurrent Forecasters (RNN, LSTM, GRU)
===========================================

Sequence-aware recurrent neural network models for time-series forecasting.

Architecture Diagram / Layout:
    Input [Batch, W, Features] 
       -> RNN/LSTM/GRU [Batch, W, Hidden]
       -> Gather Last Step Output [Batch, Hidden]
       -> Linear [Batch, Hidden -> 1]
       -> Output [Batch, 1]

Key insights / educational takeaways:
    * Vanilla RNNs suffer from vanishing and exploding gradients when processing long sequences.
    * LSTMs solve this by introducing an internal cell memory state and gating functions.
    * GRUs simplify the gating by combining the hidden and cell state, maintaining comparable performance with fewer parameters.

Run:
    python "03.rnn-lstm-gru.py" --dataset jena --variant lstm --epochs 5
    python "03.rnn-lstm-gru.py" --dataset spy --variant rnn --epochs 5
"""

import os
import torch
import torch.nn as nn
import ts_common as mc


class RecurrentForecaster(nn.Module):
    """Unified recurrent network for time-series forecasting supporting RNN, LSTM, and GRU."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, cell_type: str = "lstm"):
        super().__init__()
        self.cell_type = cell_type.lower()

        if self.cell_type == "rnn":
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        elif self.cell_type == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [Batch, W, Features]
        out, _ = self.rnn(x) # out: [Batch, W, Hidden]
        # We take the output of the last timestep
        last_step = out[:, -1, :] # [Batch, Hidden]
        return self.fc(last_step)


def main():
    p = mc.build_argparser("Recurrent time-series models")
    args = p.parse_args()

    W = 24
    device = mc.get_device(args.device)

    train_loader, test_loader, num_features, mean, std = mc.get_dataloaders(
        args.dataset, args.batch_size, args.limit, W=W
    )

    variant = args.variant or "lstm"
    if variant not in ["rnn", "lstm", "gru"]:
        variant = "lstm"

    model = RecurrentForecaster(input_dim=num_features, hidden_dim=64, num_layers=1, cell_type=variant)
    model_name = variant.upper()

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
