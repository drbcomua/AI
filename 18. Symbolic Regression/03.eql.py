"""
03. Equation Learner — EQL (Martius & Lampert, 2016)
====================================================

Differentiable symbolic regression: keep the neural training machinery, but
replace the network's fixed activations with the **primitive functions of the
target algebra**, then use L1 sparsity + hard thresholding to prune the network
down to a formula you can read off the surviving weights.

Where an MLP layer is  y = ReLU(W x + b)  with one nonlinearity per node, an EQL
layer routes a linear pre-activation z = W x + b through a *menu* of primitives —
identity, sin, cos — and, crucially, **multiplication units** that multiply two
components of z.  Multiplication is what lets the network build polynomial and
product terms (x², x·y) that MLPs can only approximate:

    x -> [ W1 x + b1 = z ] -> split z into
             id   :  z_a                    (pass-through)
             sin  :  sin(z_b)
             cos  :  cos(z_c)
             mult :  z_d · z_e              (a learned product)
         -> concat -> next EQL layer -> ... -> Linear -> y_hat

    Stacking L EQL layers reaches depth-L compositions: one layer gives x²
    (= x·x), two layers give x³ (= x·x²), sin(x²), etc.

Phased training (the EQL recipe):
    Phase 1  — fit with an L1 penalty on all weights (encourages many to shrink).
    Sparsify — zero every weight below a threshold and FREEZE that mask.
    Phase 2  — keep training only the survivors, so the remaining terms refit
               cleanly without the pruned clutter.
Then walk the sparse weights layer by layer to print the closed-form expression.

Architectural note: the paper stacks several EQL layers and omits division to
avoid poles; this compact version uses 2 layers with {id, sin, cos, mult} units
(no division), which is enough for the polynomial/trig/product problems here and
keeps the weight-to-expression readout legible.  It is parameter-matched (by
printed count) to the MLP (04) and KAN (05).

Educational takeaway:
    EQL does symbolic regression by **gradient descent over expression
    structure** — the smooth, GPU-friendly cousin of GP's discrete search (01).
    It recovers structure when the truth is expressible in its primitive menu and
    the L1 prune finds it; it inherits the usual gradient pitfalls (local minima,
    sensitivity to the sparsity threshold) when it does not.

Run:
    python "03.eql.py" --problem nguyen1 --epochs 400
    python "03.eql.py" --problem kinetic --epochs 400
    python "03.eql.py" --problem nguyen1 --limit 200 --epochs 30   # smoke test
"""

import os
import numpy as np
import torch
import torch.nn as nn
import sr_common as mc


class EQLLayer(nn.Module):
    """One EQL block: linear pre-activation -> {id, sin, cos, mult} primitives.

    pre-activation width = n_id + n_sin + n_cos + 2*n_mult   (mult needs 2 inputs)
    output width          = n_id + n_sin + n_cos +   n_mult
    A frozen 0/1 `mask` buffer implements the phase-2 hard sparsity.
    """
    def __init__(self, in_dim, n_id=2, n_sin=2, n_cos=1, n_mult=2):
        super().__init__()
        self.n_id, self.n_sin, self.n_cos, self.n_mult = n_id, n_sin, n_cos, n_mult
        self.pre_dim = n_id + n_sin + n_cos + 2 * n_mult
        self.out_dim = n_id + n_sin + n_cos + n_mult
        self.weight = nn.Parameter(torch.empty(self.pre_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(self.pre_dim))
        nn.init.normal_(self.weight, std=0.5)
        self.register_buffer("mask", torch.ones_like(self.weight))

    def _pre(self, x):
        return torch.nn.functional.linear(x, self.weight * self.mask, self.bias)

    def forward(self, x):
        z = self._pre(x)
        i, s, c, m = self.n_id, self.n_sin, self.n_cos, self.n_mult
        parts = [z[:, :i],
                 torch.sin(z[:, i:i + s]),
                 torch.cos(z[:, i + s:i + s + c])]
        off = i + s + c
        if m:
            a = z[:, off:off + m]
            b = z[:, off + m:off + 2 * m]
            parts.append(torch.clamp(a * b, -1e4, 1e4))
        return torch.cat(parts, dim=1)

    # ---- symbolic readout ------------------------------------------------- #
    def expr_out(self, in_strs, threshold):
        """Given expression strings for each input, return strings for outputs."""
        W = (self.weight * self.mask).detach().cpu().numpy()
        b = self.bias.detach().cpu().numpy()
        z = [self._lin_str(W[r], b[r], in_strs, threshold) for r in range(self.pre_dim)]
        i, s, c, m = self.n_id, self.n_sin, self.n_cos, self.n_mult
        out = list(z[:i])
        out += [f"sin({z[i + k]})" for k in range(s)]
        out += [f"cos({z[i + s + k]})" for k in range(c)]
        off = i + s + c
        for k in range(m):
            out.append(f"({z[off + k]}) * ({z[off + m + k]})")
        return out

    @staticmethod
    def _lin_str(w, b, in_strs, threshold):
        terms = []
        for wj, sj in zip(w, in_strs):
            if abs(wj) > threshold:
                terms.append(f"{wj:.3g}*{sj}")
        if abs(b) > threshold:
            terms.append(f"{b:.3g}")
        return " + ".join(terms) if terms else "0"


class EQL(nn.Module):
    def __init__(self, in_features, n_layers=2, **layer_kw):
        super().__init__()
        self.layers = nn.ModuleList()
        d = in_features
        for _ in range(n_layers):
            layer = EQLLayer(d, **layer_kw)
            self.layers.append(layer)
            d = layer.out_dim
        self.out = nn.Linear(d, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x)

    def l1(self):
        reg = sum((layer.weight * layer.mask).abs().sum() for layer in self.layers)
        return reg + self.out.weight.abs().sum()

    def apply_mask(self, threshold):
        """Freeze a 0/1 mask on every weight below `threshold` (the prune step)."""
        with torch.no_grad():
            for layer in self.layers:
                layer.mask.copy_((layer.weight.abs() > threshold).float())

    def expression(self, var_names, threshold):
        strs = list(var_names)
        for layer in self.layers:
            strs = layer.expr_out(strs, threshold)
        w = self.out.weight.detach().cpu().numpy().reshape(-1)
        b = float(self.out.bias.detach().cpu().numpy().reshape(-1)[0])
        terms = [f"{wj:.3g}*({sj})" for wj, sj in zip(w, strs) if abs(wj) > threshold]
        if abs(b) > threshold:
            terms.append(f"{b:.3g}")
        return " + ".join(terms) if terms else "0"


def main():
    p = mc.build_argparser("Equation Learner (EQL) symbolic regression")
    p.add_argument("--n-layers", type=int, default=2, help="number of EQL blocks")
    p.add_argument("--l1", type=float, default=1e-3, help="L1 sparsity coefficient")
    p.add_argument("--threshold", type=float, default=0.1,
                   help="prune/readout weight threshold")
    args = p.parse_args()
    mc.set_seed(args.seed)
    device = mc.get_device(args.device)

    problem = mc.get_problem(args.problem)
    data = mc.apply_limit(mc.make_dataset(problem, noise_std=args.noise,
                                          seed=args.seed), args.limit)

    model = EQL(problem["n_vars"], n_layers=args.n_layers)
    l1 = args.l1
    extra = lambda mdl: l1 * mdl.l1()

    # Phase 1: fit with L1 sparsity.
    print("Phase 1: fit with L1 sparsity")
    mc.train_regression(model, data, epochs=args.epochs, lr=args.lr,
                        batch_size=args.batch_size, device=device, extra_loss=extra)
    # Sparsify: freeze a hard mask on small weights.
    model.apply_mask(args.threshold)
    active = int(sum(int(layer.mask.sum().item()) for layer in model.layers))
    print(f"Sparsify: kept {active} first-stage weights above {args.threshold}")
    # Phase 2: refit survivors with reduced L1.
    print("Phase 2: refit survivors (mask frozen)")
    l1 = args.l1 * 0.1
    mc.train_regression(model, data, epochs=max(1, args.epochs // 2), lr=args.lr * 0.5,
                        batch_size=args.batch_size, device=device, extra_loss=extra)

    model.to("cpu")
    expr = model.expression(problem["vars"], args.threshold)
    print(f"\nRecovered expression: y = {expr}")

    predict_fn = mc.torch_predict_fn(model, torch.device("cpu"))
    mc.report(problem, data, predict_fn, model_name="EQL",
              expr=expr, save_dir=None if args.no_figure else _here())


def _here():
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    main()
