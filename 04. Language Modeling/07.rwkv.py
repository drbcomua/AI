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
    * A numerically stable scan (max-tracking) keeps the exponentials from blowing
      up — the same trick as a log-sum-exp.

Run:
    python "07.rwkv.py" --epochs 5
    python "07.rwkv.py" --limit 50000 --epochs 2
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
        # numerically stable WKV recurrence (aa/bb = running num/den, pp = running max)
        aa = torch.zeros(B, d, device=x.device)
        bb = torch.zeros(B, d, device=x.device)
        pp = torch.full((B, d), -1e38, device=x.device)
        out = torch.empty(B, T, d, device=x.device)
        for t in range(T):
            kt, vt = k[:, t], v[:, t]
            ww = u + kt                                    # include current token (bonus u)
            qq = torch.maximum(pp, ww)
            e1, e2 = torch.exp(pp - qq), torch.exp(ww - qq)
            out[:, t] = (e1 * aa + e2 * vt) / (e1 * bb + e2)
            ww = pp + w                                     # decay the running state
            qq = torch.maximum(ww, kt)
            e1, e2 = torch.exp(ww - qq), torch.exp(kt - qq)
            aa = e1 * aa + e2 * vt
            bb = e1 * bb + e2
            pp = qq
        return self.output(r * out)


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
