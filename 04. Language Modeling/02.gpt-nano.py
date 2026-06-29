"""
02. Nano decoder-only Transformer (Nano-GPT)
============================================

A scaled-down, character-level decoder-only Transformer generator.

Architecture Diagram / Layout:
    Input [Batch, Seq_Len]
       -> Token Embedding [Batch, Seq_Len, d_model]
       -> Learnable Position Embedding [Batch, Seq_Len, d_model]
       -> Stack of GPT Block Layers:
            * LayerNorm
            * Causal Self-Attention [Batch, Seq_Len, d_model]
            * Residual addition
            * LayerNorm
            * FeedForward Linear MLP [Batch, Seq_Len, d_model]
            * Residual addition
       -> Final LayerNorm
       -> Linear Language Modeling Head [Batch, Seq_Len, Vocab_Size]

Key insights / educational takeaways:
    * Causal self-attention masks out future steps, ensuring token prediction only depends on historical contexts.
    * Positional embeddings are required because attention is order-invariant (bag-of-words) without them.

Run:
    python "02.gpt-nano.py" --epochs 5
    python "02.gpt-nano.py" --limit 50000 --epochs 2
"""

import os
import torch
import torch.nn as nn
import lm_common as mc


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking."""
    def __init__(self, d_model: int = 64, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead

        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [B, T, C] (Batch, Seq_Len, d_model)
        B, T, C = x.size()

        # Query, Key, Value projections
        qkv = self.qkv_proj(x) # [B, T, 3 * C]
        q, k, v = qkv.chunk(3, dim=-1) # Each is [B, T, C]

        # Reshape to project heads: [B, nhead, T, head_dim]
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # Dot-product attention: [B, nhead, T, T]
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask: create boolean mask where True elements are upper triangular indices
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(1), float('-inf'))

        probs = torch.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)

        # Output context: [B, nhead, T, head_dim]
        context = probs @ v
        # Re-assemble heads: [B, T, C]
        context = context.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.out_proj(context))


class GPTBlock(nn.Module):
    """A standard Transformer decoder block with attention & feed-forward."""
    def __init__(self, d_model: int = 64, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-LN residual setup
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTNano(nn.Module):
    """Nano-GPT Language Model."""
    def __init__(self, vocab_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 3, max_seq_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.Sequential(*[
            GPTBlock(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x shape: [B, T]
        B, T = x.size()
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max sequence len {self.max_seq_len}"

        tok_emb = self.token_embedding(x) # [B, T, d_model]
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0) # [1, T]
        pos_emb = self.pos_embedding(pos) # [1, T, d_model]

        h = tok_emb + pos_emb # [B, T, d_model]
        h = self.blocks(h)
        h = self.ln_f(h)
        return self.lm_head(h) # [B, T, Vocab_Size]


def main():
    p = mc.build_argparser("Nano-GPT Language Model")
    args = p.parse_args()

    W = 64 # sequence length window
    device = mc.get_device(args.device)

    print(f"Loading Shakespeare dataset (limit={args.limit})...")
    train_loader, test_loader, tokenizer = mc.get_shakespeare_dataloaders(
        seq_len=W, batch_size=args.batch_size, limit=args.limit
    )

    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary Size (unique chars): {vocab_size}")

    model = GPTNano(vocab_size=vocab_size, d_model=64, nhead=4, num_layers=3, max_seq_len=W)

    # Train
    mc.train_language_model(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)

    # Generate samples
    seed = "Before we proceed any further, hear me speak."
    mc.generate_text(model, start_str=seed, tokenizer=tokenizer, gen_len=150, temperature=0.8, device=device)


if __name__ == "__main__":
    main()
