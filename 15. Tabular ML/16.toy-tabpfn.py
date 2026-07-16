"""
16. Toy TabPFN — a Prior-Fitted Network (Hollmann et al., 2023)
==============================================================

The headline modern development in tabular ML: *in-context learning* for tables.
A Prior-Fitted Network (PFN) is a transformer meta-trained on a huge number of
**synthetic** classification tasks. Once trained, it classifies a brand-new real
dataset in a **single forward pass with zero gradient steps** — you feed it the
labeled training rows (the "support" / context) and the unlabeled test rows (the
"queries") together, and it reads off predictions by attending from queries to
support. Learning has been amortized into the network weights ahead of time.

This is a *toy* demonstration of the mechanism, not a TabPFN replica: the real
TabPFN uses a carefully designed structural-causal-model prior, handles variable
feature counts, and is trained for a very long time. Here the prior is a batch
of random two-layer MLPs (smooth random decision boundaries), the feature/class
counts are fixed to the target dataset, and the context is capped for speed.

Architecture Diagram / Layout:
    Meta-training (on synthetic tasks, gradient descent):
        sample a random-MLP task -> (support X,y) + (query X,y)
        token_i = Linear(x_i) + Embed(y_i)      (queries use an "unknown" label)
        seq = [support tokens ... | query tokens ...]
        TransformerEncoder(seq) -> head over query positions -> CE vs query y
    Inference (on the REAL dataset, NO gradients):
        support = real training rows (labeled), query = real test rows
        one forward pass -> argmax over query outputs = predictions

Key insights / educational takeaways:
    * The weights encode a *learning algorithm*, not a fitted model: the same
      frozen network classifies any dataset shaped like its prior, instantly.
    * Attention is the mechanism — queries retrieve label information from the
      support set exactly like a differentiable, learned nearest-neighbor.
    * Trade-off: inference cost is O((n_support + n_query)^2) attention, so the
      context is capped (real TabPFN targets <~1000 training rows).

Run:
    python "16.toy-tabpfn.py" --dataset wine --epochs 20
    python "16.toy-tabpfn.py" --dataset covtype --epochs 20 --context-cap 512
    python "16.toy-tabpfn.py" --limit 2000 --epochs 2        # fast smoke test
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tabular_common as mc


class ToyPFN(nn.Module):
    def __init__(self, n_features, n_classes, d=64, n_layers=3, n_heads=8):
        super().__init__()
        self.n_classes = n_classes
        self.unknown = n_classes                 # extra label id for queries
        self.x_enc = nn.Linear(n_features, d)
        self.y_enc = nn.Embedding(n_classes + 1, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=2 * d,
            dropout=0.0, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x_sup, y_sup, x_qry):
        B, S, _ = x_sup.shape
        Q = x_qry.size(1)
        sup = self.x_enc(x_sup) + self.y_enc(y_sup)
        unk = torch.full((B, Q), self.unknown, device=x_qry.device, dtype=torch.long)
        qry = self.x_enc(x_qry) + self.y_enc(unk)
        out = self.encoder(torch.cat([sup, qry], dim=1))
        return self.head(out[:, S:])             # (B, Q, C)


def sample_synthetic_tasks(batch, n_features, n_classes, n_support, n_query,
                           device, hidden=16):
    """A random-MLP prior: each task is a random 2-layer net's argmax labels."""
    n = n_support + n_query
    x = torch.randn(batch, n, n_features, device=device)
    W1 = torch.randn(batch, n_features, hidden, device=device) / (n_features ** 0.5)
    W2 = torch.randn(batch, hidden, n_classes, device=device) / (hidden ** 0.5)
    h = torch.tanh(torch.einsum("bnf,bfh->bnh", x, W1))
    logits = torch.einsum("bnh,bhc->bnc", h, W2)
    y = logits.argmax(-1)
    return (x[:, :n_support], y[:, :n_support],
            x[:, n_support:], y[:, n_support:])


def meta_train(model, n_features, n_classes, *, steps, device, batch=32,
               n_support=128, n_query=32, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    log_every = max(1, steps // 10)
    for step in range(1, steps + 1):
        x_sup, y_sup, x_qry, y_qry = sample_synthetic_tasks(
            batch, n_features, n_classes, n_support, n_query, device)
        logits = model(x_sup, y_sup, x_qry)
        loss = F.cross_entropy(logits.reshape(-1, n_classes), y_qry.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % log_every == 0:
            acc = (logits.argmax(-1) == y_qry).float().mean().item()
            print(f"  meta-step {step:4d}/{steps} | synthetic loss {loss.item():.4f} "
                  f"| synthetic query acc {acc:.4f}")


@torch.no_grad()
def pfn_predict(model, X_sup, y_sup, X_qry, device, chunk=256):
    """One frozen forward pass per query chunk; support is shared context."""
    model.eval()
    xs = torch.from_numpy(X_sup).float().unsqueeze(0).to(device)
    ys = torch.from_numpy(y_sup).long().unsqueeze(0).to(device)
    preds, probs = [], []
    for i in range(0, len(X_qry), chunk):
        xq = torch.from_numpy(X_qry[i:i + chunk]).float().unsqueeze(0).to(device)
        logits = model(xs, ys, xq)[0]            # (q, C)
        p = torch.softmax(logits, dim=-1)
        preds.append(p.argmax(-1).cpu().numpy())
        probs.append(p.cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def main():
    p = mc.build_argparser("Toy TabPFN (Prior-Fitted Network)", lr=1e-3)
    p.add_argument("--context-cap", type=int, default=1000,
                   help="max support rows fed as in-context examples at inference")
    p.add_argument("--d-model", type=int, default=64)
    args = p.parse_args()
    mc.set_seed(args.seed)
    device = mc.get_device(args.device)

    X_train, X_test, y_train, y_test, feature_names, class_names = \
        mc.load_classification_dataset(args.dataset)
    if args.limit is not None:
        X_train, y_train = X_train[:args.limit], y_train[:args.limit]
    Xtr, Xte = mc.standardize(X_train, X_test)
    n_features, n_classes = Xtr.shape[1], len(class_names)

    model = ToyPFN(n_features, n_classes, d=args.d_model).to(device)
    print(f"Device: {device} | trainable params: {mc.count_parameters(model):,}")

    # Meta-train on synthetic tasks; ~100 gradient steps per requested "epoch".
    steps = args.epochs * 100
    print(f"Meta-training on synthetic random-MLP tasks for {steps} steps...")
    meta_train(model, n_features, n_classes, steps=steps, device=device, lr=args.lr)

    # In-context inference on the REAL dataset (zero gradient steps).
    if len(Xtr) > args.context_cap:
        idx = np.random.RandomState(args.seed).choice(
            len(Xtr), args.context_cap, replace=False)
        Xsup, ysup = Xtr[idx], y_train[idx]
    else:
        Xsup, ysup = Xtr, y_train
    print(f"In-context prediction: {len(Xsup)} support rows -> {len(Xte)} queries "
          f"(no gradient steps)")
    y_pred, y_prob = pfn_predict(model, Xsup, ysup, Xte, device)

    mc.report_classification(y_test, y_pred, y_prob, class_names=class_names,
                             model_name="Toy-TabPFN",
                             save_dir=None if args.no_figure else
                             os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
