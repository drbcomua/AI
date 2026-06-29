"""
06. Mixture-of-Experts GPT (Shazeer et al., 2017; Fedus et al., 2021 — Switch Transformer)
=========================================================================================

A decoder-only Transformer where each block's feed-forward layer is replaced by a
**Mixture of Experts**: a pool of independent MLP "experts" plus a small router
that, per token, picks the top-k experts to run. This is *conditional computation*
— total parameters grow with the number of experts, but the FLOPs per token stay
fixed (only k experts fire). It's the mechanism behind today's largest LLMs.

Architecture Diagram / Layout:
    GPT block = Causal Self-Attention  +  MoE feed-forward
      MoE: router logits = Linear(d, E)  (+ noise during training)
           pick top-k experts; weight = softmax(top-k logits)
           output = sum_k weight_k * Expert_{idx_k}(token)

Key insights / educational takeaways:
    * Decouples capacity (many experts) from per-token cost (only k run).
    * Noisy top-k gating spreads tokens across experts and limits "expert collapse"
      without a separate load-balancing loss (kept out to fit the shared trainer).

Run:
    python "06.moe-gpt.py" --epochs 5
    python "06.moe-gpt.py" --limit 50000 --epochs 2
"""

import os
import torch
import torch.nn as nn
import lm_common as mc


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=64, nhead=4, dropout=0.1):
        super().__init__()
        self.nhead, self.head_dim = nhead, d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        ctx = torch.softmax(scores, dim=-1) @ v
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(ctx)


class Expert(nn.Module):
    def __init__(self, d_model, mult=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, mult * d_model), nn.ReLU(),
                                 nn.Linear(mult * d_model, d_model))

    def forward(self, x):
        return self.net(x)


class MoE(nn.Module):
    def __init__(self, d_model, num_experts=4, top_k=2, noise=1.0):
        super().__init__()
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k, self.noise = top_k, noise

    def forward(self, x):                                  # [B, T, d]
        B, T, d = x.shape
        flat = x.reshape(-1, d)                            # [N, d]
        logits = self.gate(flat)
        if self.training:                                 # noisy top-k gating
            logits = logits + torch.randn_like(logits) * self.noise
        top_val, top_idx = logits.topk(self.top_k, dim=-1)
        weights = torch.softmax(top_val, dim=-1)          # [N, k]

        out = torch.zeros_like(flat)
        for slot in range(self.top_k):
            idx = top_idx[:, slot]                         # [N]
            w = weights[:, slot:slot + 1]
            for e, expert in enumerate(self.experts):
                sel = idx == e
                if sel.any():
                    out[sel] += w[sel] * expert(flat[sel])
        return out.view(B, T, d)


class MoEBlock(nn.Module):
    def __init__(self, d_model, nhead, num_experts, top_k, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoE(d_model, num_experts, top_k)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x


class MoEGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=3,
                 num_experts=4, top_k=2, max_seq_len=64):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [MoEBlock(d_model, nhead, num_experts, top_k) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):                                  # [B, T]
        B, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))                    # [B, T, vocab]


def main():
    args = mc.build_argparser("Mixture-of-Experts GPT Language Model").parse_args()
    W = 64
    device = mc.get_device(args.device)

    train_loader, test_loader, tokenizer = mc.get_shakespeare_dataloaders(
        seq_len=W, batch_size=args.batch_size, limit=args.limit)
    print(f"Vocabulary Size (unique chars): {tokenizer.vocab_size}")

    model = MoEGPT(tokenizer.vocab_size, num_experts=4, top_k=2, max_seq_len=W)
    mc.train_language_model(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)

    mc.generate_text(model, start_str="Before we proceed any further, hear me speak.",
                     tokenizer=tokenizer, gen_len=150, temperature=0.8, device=device)


if __name__ == "__main__":
    main()
