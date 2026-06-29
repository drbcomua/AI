"""
05. WaveNet — Gated Causal Dilated CNN (van den Oord et al., 2016; Dauphin et al., 2017)
=======================================================================================

The convolutional answer to sequence modeling. Stacks of *causal* 1D convolutions
(each output sees only the past) with exponentially growing *dilation* give a very
large receptive field in few layers, while training fully in parallel — no
recurrence, no attention. Each layer uses a gated activation (tanh * sigmoid, the
GLU idea) and residual + skip connections.

Architecture Diagram / Layout:
    Input [Batch, Seq_Len] -> Embedding -> [Batch, d, Seq_Len]
       Residual blocks (dilation 1,2,4,8,16,32):
            tanh(CausalConv) * sigmoid(CausalConv)  -> 1x1 -> (+ residual), (+ skip)
       Sum(skips) -> ReLU -> 1x1 -> ReLU -> 1x1 -> [Batch, Seq_Len, Vocab]

Key insights / educational takeaways:
    * Dilation doubles the receptive field per layer, so 6 layers see 64 steps.
    * Causal (left-only) padding is what prevents the model from peeking ahead.
    * Parallel training like a Transformer, but with a fixed (not learned) mixing
      pattern — a useful contrast in inductive bias.

Run:
    python "05.wavenet.py" --epochs 5
    python "05.wavenet.py" --limit 50000 --epochs 2
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lm_common as mc


class CausalConv1d(nn.Module):
    """1D convolution padded on the left only, so output t depends on inputs <= t."""
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class GatedBlock(nn.Module):
    def __init__(self, d_model, kernel, dilation):
        super().__init__()
        self.filt = CausalConv1d(d_model, d_model, kernel, dilation)
        self.gate = CausalConv1d(d_model, d_model, kernel, dilation)
        self.res = nn.Conv1d(d_model, d_model, 1)
        self.skip = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x):
        z = torch.tanh(self.filt(x)) * torch.sigmoid(self.gate(x))   # gated activation
        return x + self.res(z), self.skip(z)


class WaveNet(nn.Module):
    def __init__(self, vocab_size, d_model=64, kernel=2, dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.input_conv = nn.Conv1d(d_model, d_model, 1)
        self.blocks = nn.ModuleList([GatedBlock(d_model, kernel, d) for d in dilations])
        self.head = nn.Sequential(
            nn.ReLU(), nn.Conv1d(d_model, d_model, 1),
            nn.ReLU(), nn.Conv1d(d_model, vocab_size, 1),
        )

    def forward(self, x):                                  # x: [B, T]
        h = self.embedding(x).transpose(1, 2)             # [B, d, T]
        h = self.input_conv(h)
        skips = 0
        for block in self.blocks:
            h, s = block(h)
            skips = skips + s
        out = self.head(skips)                            # [B, vocab, T]
        return out.transpose(1, 2).contiguous()           # [B, T, vocab]


def main():
    args = mc.build_argparser("WaveNet (Gated Causal CNN) Language Model").parse_args()
    W = 64
    device = mc.get_device(args.device)

    train_loader, test_loader, tokenizer = mc.get_shakespeare_dataloaders(
        seq_len=W, batch_size=args.batch_size, limit=args.limit)
    print(f"Vocabulary Size (unique chars): {tokenizer.vocab_size}")

    model = WaveNet(tokenizer.vocab_size)
    mc.train_language_model(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)

    mc.generate_text(model, start_str="Before we proceed any further, hear me speak.",
                     tokenizer=tokenizer, gen_len=150, temperature=0.8, device=device)


if __name__ == "__main__":
    main()
