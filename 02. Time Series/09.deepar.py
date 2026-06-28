"""
09. DeepAR (Salinas et al., 2020 — Amazon, "Probabilistic Forecasting with Autoregressive RNNs")
================================================================================================

Every other model in this folder predicts a single number. DeepAR predicts a
*probability distribution*: an LSTM reads the history and emits the parameters
(mean and standard deviation) of a Gaussian over the next value, trained by
maximizing likelihood. You get calibrated uncertainty, not just a point estimate.

Architecture Diagram / Layout:
    Input [B, W, F] -> LSTM -> last hidden state [B, H]
                    -> Linear -> mu     [B, 1]
                    -> Linear -> sigma  [B, 1]  (softplus, kept positive)
    Loss = Gaussian negative log-likelihood of the true value under (mu, sigma).

Key insights / educational takeaways:
    * Optimizing likelihood (not MSE) makes the model report *how confident* it is;
      sigma widens on noisy regimes (very visible on SPY) and narrows on the
      smooth, predictable Jena signal.
    * The point forecast used for the metrics below is simply the predicted mean.

Run:
    python "09.deepar.py" --dataset jena --epochs 5
    python "09.deepar.py" --dataset spy --epochs 5
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import ts_common as mc


class DeepAR(nn.Module):
    def __init__(self, num_features: int, hidden: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(num_features, hidden, num_layers, batch_first=True, dropout=0.1)
        self.mu_head = nn.Linear(hidden, 1)
        self.sigma_head = nn.Linear(hidden, 1)

    def distribution(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        mu = self.mu_head(h)
        sigma = F.softplus(self.sigma_head(h)) + 1e-3     # keep strictly positive
        return mu, sigma

    def forward(self, x):                                 # point forecast = predicted mean
        return self.distribution(x)[0]


def gaussian_nll(mu, sigma, y):
    y = y.unsqueeze(-1)
    return (0.5 * math.log(2 * math.pi) + torch.log(sigma)
            + 0.5 * ((y - mu) / sigma) ** 2).mean()


def train_deepar(model, train_loader, test_loader, *, epochs, lr, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Device: {device} | trainable params: "
          f"{sum(p.numel() for p in model.parameters()):,}  | loss: Gaussian NLL")
    print("-" * 64)
    for epoch in range(1, epochs + 1):
        model.train()
        tot = run = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            mu, sigma = model.distribution(x)
            loss = gaussian_nll(mu, sigma, y)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * y.size(0); tot += y.size(0)
        # validation NLL + MSE of the mean
        model.eval()
        v_nll = v_mse = vt = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                mu, sigma = model.distribution(x)
                v_nll += gaussian_nll(mu, sigma, y).item() * y.size(0)
                v_mse += ((mu.squeeze(-1) - y) ** 2).sum().item()
                vt += y.size(0)
        print(f"Epoch {epoch:2d}/{epochs} | train_nll {run/tot:.5f} | "
              f"test_nll {v_nll/vt:.5f} | test_mse {v_mse/vt:.5f}")
    print("-" * 64)


def main():
    args = mc.build_argparser("DeepAR Probabilistic Forecaster").parse_args()
    W = 24
    device = mc.get_device(args.device)

    train_loader, test_loader, num_features, mean, std = mc.get_dataloaders(
        args.dataset, args.batch_size, args.limit, W=W)

    model = DeepAR(num_features)
    train_deepar(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)
    y_true, y_pred = mc.evaluate(model, test_loader, device=device)
    mc.report(y_true, y_pred, mean, std, dataset_name=args.dataset, model_name="DeepAR",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
