"""
13. Mamba (Gu & Dao, 2023 — "Linear-Time Sequence Modeling with Selective State Spaces")
=======================================================================================

Mamba is the state-space-model (SSM) challenger to the transformer. A classical
SSM, h_t = A h_{t-1} + B x_t ; y_t = C h_t, is linear-time but content-agnostic.
Mamba makes the SSM **selective**: B, C, and the timestep Δ are produced *from the
input*, so the model can choose what to remember or forget at each step — the
expressiveness of attention at the cost of a recurrence (here a simple scan).

Architecture Diagram / Layout:
    Input [B, W, F] -> Linear embed [B, W, d]
       MambaBlock: in_proj -> (x, gate)
                   causal depthwise Conv1d -> SiLU
                   input-dependent Δ, B, C ; discretize A,B ; selective scan
                   y = scan_output * SiLU(gate) -> out_proj   (+ residual)
       -> last step [B, d] -> Linear -> [B, 1]

Key insights / educational takeaways:
    * Selectivity is what lifts SSMs to transformer-level quality while keeping
      O(W) cost — promising for very long sequences.
    * The scan here is an explicit Python loop over W=24 steps (clear, not fast);
      real Mamba uses a hardware-aware parallel scan.

Run:
    python "13.mamba.py" --dataset jena --epochs 5
    python "13.mamba.py" --dataset spy --epochs 5
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import ts_common as mc


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = expand * d_model
        self.dt_rank = max(1, d_model // 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        # A is parameterized in log space and stays negative (stable decay)
        A = torch.arange(1, d_state + 1, dtype=torch.float).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.d_state = d_state

    def forward(self, x):                                 # [B, L, d_model]
        B, L, _ = x.shape
        x_in, gate = self.in_proj(x).chunk(2, dim=-1)     # each [B, L, d_inner]
        # causal depthwise conv
        xc = self.conv1d(x_in.transpose(1, 2))[..., :L].transpose(1, 2)
        xc = F.silu(xc)
        # input-dependent SSM parameters
        dbl = self.x_proj(xc)
        dt, Bm, Cm = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))              # [B, L, d_inner]
        A = -torch.exp(self.A_log)                        # [d_inner, d_state]
        deltaA = torch.exp(delta.unsqueeze(-1) * A)       # [B, L, d_inner, d_state]
        deltaB_x = delta.unsqueeze(-1) * Bm.unsqueeze(2) * xc.unsqueeze(-1)
        # selective scan
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB_x[:, t]
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))     # [B, d_inner]
        y = torch.stack(ys, dim=1) + xc * self.D
        y = y * F.silu(gate)
        return self.out_proj(y)


class Mamba(nn.Module):
    def __init__(self, num_features: int, d_model: int = 64, n_blocks: int = 2):
        super().__init__()
        self.embed = nn.Linear(num_features, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model) for _ in range(n_blocks)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):                                 # [B, W, F]
        x = self.embed(x)
        for block, norm in zip(self.blocks, self.norms):
            x = x + block(norm(x))                        # pre-norm residual
        return self.head(x[:, -1, :])                     # [B, 1]


def main():
    args = mc.build_argparser("Mamba (Selective SSM) Forecaster").parse_args()
    W = 24
    device = mc.get_device(args.device)

    train_loader, test_loader, num_features, mean, std = mc.get_dataloaders(
        args.dataset, args.batch_size, args.limit, W=W)

    model = Mamba(num_features)
    mc.train(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)
    y_true, y_pred = mc.evaluate(model, test_loader, device=device)
    mc.report(y_true, y_pred, mean, std, dataset_name=args.dataset, model_name="Mamba",
              save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
