"""
04. N-gram Language Model (classical, count-based)
==================================================

The pre-neural baseline for language modeling: estimate P(next char | previous
n-1 chars) by simply *counting* how often each continuation appears in the
training text. No gradients, no embeddings — just frequencies with smoothing and
back-off. It establishes how much of "language" is capturable by local statistics
alone, the bar the neural models must clear.

Architecture / Method:
    * Count all k-grams for k = 1..n over the training text.
    * P(c | context) = add-k smoothed frequency at the highest order whose context
      was actually seen; otherwise back off to a shorter context (down to unigram).
    * Perplexity = exp(mean negative log-likelihood) on held-out text.
    * Generation samples from the back-off distribution with temperature.

Key insights / educational takeaways:
    * Local character statistics already produce Shakespeare-flavored gibberish.
    * Raising the order lowers perplexity but explodes the context table and
      generalizes worse — the curse of dimensionality the neural models dodge.

Run:
    python "04.ngram.py" --order 5
    python "04.ngram.py" --order 3 --limit 50000
"""

import math
from collections import Counter, defaultdict
import numpy as np
import lm_common as mc


def build_counts(ids, order):
    """counts[k][context_tuple] -> Counter of next-id, for k = 1..order."""
    counts = {k: defaultdict(Counter) for k in range(1, order + 1)}
    for i in range(len(ids)):
        for k in range(1, order + 1):
            if i >= k - 1:
                ctx = tuple(ids[i - (k - 1):i])
                counts[k][ctx][ids[i]] += 1
    return counts


def dist(counts, history, order, vocab_size, k_smooth=0.1):
    """Back-off probability vector over the vocabulary given prior ids `history`."""
    for k in range(order, 0, -1):
        ctx = tuple(history[-(k - 1):]) if k > 1 else ()
        c = counts[k].get(ctx)
        if c:
            total = sum(c.values())
            probs = np.full(vocab_size, k_smooth, dtype=np.float64)
            for idx, cnt in c.items():
                probs[idx] += cnt
            return probs / (total + k_smooth * vocab_size)
    return np.full(vocab_size, 1.0 / vocab_size)


def main():
    p = mc.build_argparser("N-gram Language Model")
    p.add_argument("--order", type=int, default=5, help="n-gram order (context = order-1 chars)")
    p.add_argument("--gen-len", type=int, default=150)
    args = p.parse_args()

    text = mc.load_shakespeare()
    if args.limit is not None:
        text = text[:args.limit]
    tok = mc.CharTokenizer()
    tok.fit(text)
    ids = tok.encode(text)
    V = tok.vocab_size

    split = int(len(ids) * 0.9)
    train_ids, test_ids = ids[:split], ids[split:]

    print(f"Order: {args.order} | vocab: {V} | train chars: {len(train_ids):,}")
    print("-" * 64)
    counts = build_counts(train_ids, args.order)
    n_contexts = sum(len(counts[k]) for k in counts)
    print(f"Stored contexts (all orders): {n_contexts:,}")

    # Perplexity on held-out text
    nll = 0.0
    n = 0
    for i in range(1, len(test_ids)):
        probs = dist(counts, test_ids[:i], args.order, V)
        nll += -math.log(max(probs[test_ids[i]], 1e-12))
        n += 1
    ppl = math.exp(nll / n)
    print(f"Test perplexity      : {ppl:.2f}")
    print("-" * 64)

    # Generation with temperature
    seed = "Before we proceed any further, hear me speak."
    print(f"\n--- GENERATING TEXT (Seed: '{seed}') ---")
    generated = tok.encode(seed)
    temperature = 0.8
    for _ in range(args.gen_len):
        probs = dist(counts, generated, args.order, V)
        probs = probs ** (1.0 / temperature)
        probs = probs / probs.sum()
        generated.append(int(np.random.choice(V, p=probs)))
    print(tok.decode(generated))
    print("-------------------------------------------\n")


if __name__ == "__main__":
    main()
