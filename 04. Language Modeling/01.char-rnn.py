"""
01. Character-Level Language Model (Char-RNN)
==============================================

Recurrent neural network language model for character-level text generation.

Architecture Diagram / Layout:
    Input [Batch, Seq_Len]
       -> Embedding [Batch, Seq_Len, Embedding_Dim]
       -> LSTM Layers [Batch, Seq_Len, Hidden_Dim]
       -> Linear Head [Batch, Seq_Len, Vocab_Size]
       -> CrossEntropyLoss (shifts labels by 1)

Key insights / educational takeaways:
    * Character-level modeling removes out-of-vocabulary (OOV) tokens because the vocabulary is just the alphabet + punctuation.
    * The LSTM learns grammar, names, line structures, and word spellings character-by-character.

Run:
    python "01.char-rnn.py" --epochs 5
    python "01.char-rnn.py" --limit 50000 --epochs 2
"""

import os
import torch
import torch.nn as nn
import lm_common as mc


class CharRNN(nn.Module):
    """LSTM-based character-level language model."""
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x shape: [Batch, Seq_Len]
        embedded = self.embedding(x) # [Batch, Seq_Len, Embedding_Dim]
        out, _ = self.lstm(embedded) # out shape: [Batch, Seq_Len, Hidden_Dim]
        logits = self.fc(out) # [Batch, Seq_Len, Vocab_Size]
        return logits


def main():
    p = mc.build_argparser("Character-Level RNN Language Model")
    args = p.parse_args()

    W = 64 # Sequence window length
    device = mc.get_device(args.device)

    print(f"Loading Shakespeare dataset (limit={args.limit})...")
    train_loader, test_loader, tokenizer = mc.get_shakespeare_dataloaders(
        seq_len=W, batch_size=args.batch_size, limit=args.limit
    )

    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary Size (unique chars): {vocab_size}")

    model = CharRNN(vocab_size=vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2)

    # Train
    mc.train_language_model(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device)

    # Generate samples
    seed = "Before we proceed any further, hear me speak."
    mc.generate_text(model, start_str=seed, tokenizer=tokenizer, gen_len=150, temperature=0.8, device=device)


if __name__ == "__main__":
    main()
