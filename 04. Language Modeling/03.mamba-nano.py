"""
03. Nano selective State Space Model (Nano-Mamba)
=================================================

A scaled-down, character-level selective State Space Model (SSM) / Mamba generator.

Architecture Diagram / Layout:
    Input [Batch, Seq_Len]
       -> Token Embedding [Batch, Seq_Len, d_model]
       -> Stack of Mamba Blocks:
            * LayerNorm
            * Project Input to [Batch, Seq_Len, d_inner]
            * Causal 1D Convolution [Batch, Seq_Len, d_inner]
            * Selective SSM (Scan loop along W):
                - Delta = Softplus(Linear(x_t))
                - B = Linear(x_t), C = Linear(x_t)
                - State update: h_t = exp(Delta * A) * h_t-1 + (Delta * B) * x_t
                - Output: y_t = h_t * C
            * Gated Connection (Multiplication with SiLU projection)
            * Project Back [Batch, Seq_Len, d_model]
            * Residual addition
       -> Final LayerNorm
       -> Linear Language Modeling Head [Batch, Seq_Len, Vocab_Size]

Key insights / educational takeaways:
    * Selective State Space Models (SSMs) select which information to remember or ignore based on the input sequence content.
    * Mamba provides linear O(N) complexity over sequence length, combining the execution speed of recurrence with the training parallelizability of Transformers.

Run:
    python "03.mamba-nano.py" --epochs 5
    python "03.mamba-nano.py" --limit 50000 --epochs 2
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lm_common as mc


class SelectiveSSM(nn.Module):
    """Selective State Space Model scan simulation in PyTorch."""
    def __init__(self, d_inner: int, d_state: int = 8):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        # Learnable parameters: A is initialized to negative values for stability
        # A shape: [d_inner, d_state]
        self.A = nn.Parameter(-torch.log(torch.arange(1.0, d_state + 1.0).unsqueeze(0).repeat(d_inner, 1)))

        # Projection weights for selectivity: B_t, C_t, and Delta_t are input-dependent
        self.proj_delta = nn.Linear(d_inner, d_inner)
        self.proj_B = nn.Linear(d_inner, d_state)
        self.proj_C = nn.Linear(d_inner, d_state)

        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x):
        # x shape: [B, W, D] (Batch, Seq_Len, d_inner)
        B, W, D = x.size()
        device = x.device

        # Project parameters for the selective scan
        delta = F.softplus(self.proj_delta(x)) # [B, W, D]
        B_val = self.proj_B(x) # [B, W, N] (N = d_state)
        C_val = self.proj_C(x) # [B, W, N]

        # Scan loop: compute hidden state step-by-step
        h = torch.zeros(B, D, self.d_state, device=device) # [B, D, N]
        y_out = []

        for t in range(W):
            x_t = x[:, t, :] # [B, D]
            delta_t = delta[:, t, :] # [B, D]
            B_t = B_val[:, t, :] # [B, N]
            C_t = C_val[:, t, :] # [B, N]

            # Discretize A: exp(delta_t * A) -> [B, D, N]
            A_bar = torch.exp(delta_t.unsqueeze(-1) * self.A)

            # Discretize B: (delta_t * B_t) -> [B, D, N]
            B_bar = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)

            # State transition: h_t = A_bar * h_t-1 + B_bar * x_t
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)

            # Output: y_t = Sum_over_N(h_t * C_t) -> [B, D]
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)
            y_out.append(y_t)

        # Stack sequence results: [B, W, D]
        y = torch.stack(y_out, dim=1)
        # Direct input-to-output skip projection (D)
        return y + x * self.D


class MambaBlock(nn.Module):
    """Mamba Block with selective SSM, causal convolution, and gated residual paths."""
    def __init__(self, d_model: int, d_state: int = 8, expand: int = 2, kernel_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # Causal convolution to model local dependency
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=kernel_size,
            padding=kernel_size - 1
        )
        self.kernel_size = kernel_size

        self.ssm = SelectiveSSM(self.d_inner, d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        # x shape: [B, W, d_model]
        B, W, C = x.size()

        # Input projections
        proj_out = self.in_proj(x) # [B, W, 2 * d_inner]
        x_branch, z_branch = proj_out.chunk(2, dim=-1) # Each is [B, W, d_inner]

        # Convolve along time dim: Conv1d expects [B, Channels, W]
        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :W] # Causal crop
        x_conv = x_conv.transpose(1, 2)

        x_ssm = self.ssm(F.silu(x_conv))

        # Gate multiplication using SiLU on the second branch
        gated = x_ssm * F.silu(z_branch)

        return self.out_proj(gated)


class MambaNano(nn.Module):
    """Nano selective State Space Model language model."""
    def __init__(self, vocab_size: int, d_model: int = 64, d_state: int = 8,
                 num_layers: int = 3, max_seq_len: int = 64):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state) for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x shape: [B, T]
        h = self.token_embedding(x) # [B, T, d_model]

        for block in self.blocks:
            h = h + block(h) # Residual additions

        h = self.ln_f(h)
        return self.lm_head(h) # [B, T, Vocab_Size]


def main():
    p = mc.build_argparser("Nano-Mamba Language Model")
    args = p.parse_args()

    W = 64 # sequence window size
    device = mc.get_device(args.device)

    print(f"Loading Shakespeare dataset (limit={args.limit})...")
    train_loader, test_loader, tokenizer = mc.get_shakespeare_dataloaders(
        seq_len=W, batch_size=args.batch_size, limit=args.limit
    )

    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary Size (unique chars): {vocab_size}")

    model = MambaNano(vocab_size=vocab_size, d_model=64, d_state=8, num_layers=3)

    # Train
    mc.train_language_model(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)

    # Generate samples
    seed = "Before we proceed any further, hear me speak."
    mc.generate_text(model, start_str=seed, tokenizer=tokenizer, gen_len=150, temperature=0.8, device=device)


if __name__ == "__main__":
    main()
