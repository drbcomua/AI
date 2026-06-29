"""
07. RWKV — Receptance Weighted Key Value (Peng et al., 2023)
===========================================================

RWKV is an "attention-free Transformer": it trains in parallel like a Transformer
but runs as an O(1)-per-step RNN at inference. It replaces self-attention with a
**WKV** operator — a linear-attention recurrence with a learned per-channel time
decay — so context is summarized in a fixed-size state instead of an O(T^2)
attention matrix. It is the bridge between your Char-RNN and nano-GPT.

Architecture Diagram / Layout:
    Block = Time-Mixing  +  Channel-Mixing  (each pre-LN + residual)
      Time-Mixing : token-shift -> R,K,V -> WKV recurrence (decay w, bonus u)
                    -> output = sigmoid(R) * WKV
      Channel-Mix : token-shift -> sigmoid(R) * Value(square-relu(Key))

Key insights / educational takeaways:
    * The WKV state replaces the attention matrix: linear time, constant memory.
    * "Token shift" (mixing each token with the previous one) is a cheap, powerful
      inductive bias; no positional embeddings are needed.
    * The WKV is computed here in parallel "training mode" (one batched T x T op,
      max-stabilized) — mathematically identical to the sequential recurrence but
      fast on GPU/MPS. At inference it can instead run as a constant-memory RNN.

Performance note:
    RWKV's WKV uses a per-channel decay, so (unlike attention) it has no matmul
    shortcut — it is the most compute-heavy model in this folder. On Apple Silicon
    it is actually faster on CPU than on MPS for this small size. Use --limit for
    quick runs.

Run:
    python "07.rwkv.py" --limit 50000 --epochs 2          # quick run
    python "07.rwkv.py" --device cpu                      # full run (CPU > MPS here)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lm_common as mc


def token_shift(x):
    """Shift the sequence right by one (each position sees the previous token)."""
    return F.pad(x, (0, 0, 1, 0))[:, :-1, :]


class TimeMix(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.time_decay = nn.Parameter(torch.zeros(d_model))     # w = -exp(.) <= 0
        self.time_first = nn.Parameter(torch.zeros(d_model))     # u (bonus for current token)
        self.mix_k = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.mix_v = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.mix_r = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):                                   # [B, T, d]
        xx = token_shift(x)
        k = self.key(x * self.mix_k + xx * (1 - self.mix_k))
        v = self.value(x * self.mix_v + xx * (1 - self.mix_v))
        r = torch.sigmoid(self.receptance(x * self.mix_r + xx * (1 - self.mix_r)))

        B, T, d = x.shape
        w = -torch.exp(self.time_decay)                    # [d] negative decay
        u = self.time_first
        # WKV in parallel "training mode": equivalent to the recurrence but computed
        # as one batched T x T operation (no Python loop), so it's fast on GPU/MPS.
        # weight of key i for query t: exp((t-1-i)*w + k_i) for i<t; exp(u + k_t) at i=t.
        idx = torch.arange(T, device=x.device)
        rel = (idx.view(T, 1) - 1 - idx.view(1, T)).float()            # (t-1-i)
        bias = rel.unsqueeze(-1) * w.view(1, 1, d)                     # [T, T, d] decay (i<t)
        diag = (idx.view(T, 1) == idx.view(1, T)).unsqueeze(-1)
        future = (idx.view(1, T) > idx.view(T, 1)).unsqueeze(-1)       # i > t (not allowed)
        bias = torch.where(diag, u.view(1, 1, d).expand(T, T, d), bias)
        bias = bias.masked_fill(future, float("-inf"))
        logits = bias.unsqueeze(0) + k.unsqueeze(1)                    # [B, T, T, d], key index = dim 2
        m = logits.max(dim=2, keepdim=True).values                    # stabilize exponentials
        e = torch.exp(logits - m)
        wkv = (e * v.unsqueeze(1)).sum(dim=2) / e.sum(dim=2)           # [B, T, d]
        return self.output(r * wkv)


class ChannelMix(nn.Module):
    def __init__(self, d_model, mult=4):
        super().__init__()
        self.mix_k = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.mix_r = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.key = nn.Linear(d_model, mult * d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(mult * d_model, d_model, bias=False)

    def forward(self, x):
        xx = token_shift(x)
        k = self.key(x * self.mix_k + xx * (1 - self.mix_k))
        r = torch.sigmoid(self.receptance(x * self.mix_r + xx * (1 - self.mix_r)))
        return r * self.value(torch.square(torch.relu(k)))


class RWKVBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.time_mix = TimeMix(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.channel_mix = ChannelMix(d_model)

    def forward(self, x):
        x = x + self.time_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x


class RWKV(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.ln0 = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([RWKVBlock(d_model) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):                                   # [B, T]
        h = self.ln0(self.embedding(x))
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))                     # [B, T, vocab]


def main():
    args = mc.build_argparser("RWKV Language Model").parse_args()
    W = 64
    device = mc.get_device(args.device)

    train_loader, test_loader, tokenizer = mc.get_shakespeare_dataloaders(
        seq_len=W, batch_size=args.batch_size, limit=args.limit)
    print(f"Vocabulary Size (unique chars): {tokenizer.vocab_size}")

    model = RWKV(tokenizer.vocab_size)
    mc.train_language_model(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)

    mc.generate_text(model, start_str="Before we proceed any further, hear me speak.",
                     tokenizer=tokenizer, gen_len=150, temperature=0.8, device=device)


if __name__ == "__main__":
    main()
