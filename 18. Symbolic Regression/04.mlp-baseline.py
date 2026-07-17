"""
04. MLP Baseline — the black-box control (parameter-matched)
============================================================

A plain multilayer perceptron with ReLU activations and a scalar output.  It is
here as the **control** in the folder's central experiment: sized to the same
trainable-parameter count as the EQL (03) and KAN (05) networks, it shows what
raw capacity buys you when the architecture has *no* symbolic inductive bias.

    x [N, d] -> Linear(d, H) -> ReLU
              -> Linear(H, H) -> ReLU
              -> Linear(H, 1) -> y_hat [N, 1]

Educational takeaway (this script's whole point):
    The MLP is competitive on **interpolation RMSE** — a ReLU MLP is a universal
    approximator and fits the training region fine.  But it is a piecewise-linear
    surface with no notion of `sin`, `exp`, or `x^2`, so it:
        * FAILS extrapolation — outside the training range it continues as
          straight-line ramps, diverging from any curved truth, and
        * RECOVERS NOTHING — there is no expression to read out; it scores
          "not recoverable" by construction.

    The extrapolation figure is this script's required plot: compare its flat/
    linear continuation beyond the grey training band against the KAN (05), whose
    spline+SiLU edges keep tracking the true curve.  Same parameter budget,
    opposite behavior off-distribution — that contrast is the folder's thesis.

Run:
    python "04.mlp-baseline.py" --problem nguyen1 --epochs 400
    python "04.mlp-baseline.py" --problem gaussian --hidden 32
    python "04.mlp-baseline.py" --problem nguyen1 --limit 200 --epochs 20  # smoke
"""

import os
import torch
import torch.nn as nn
import sr_common as mc


class MLP(nn.Module):
    def __init__(self, in_features, hidden=24, depth=2):
        super().__init__()
        layers = [nn.Linear(in_features, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    p = mc.build_argparser("Plain MLP baseline (black-box control)")
    p.add_argument("--hidden", type=int, default=10,
                   help="hidden width (default 10 ≈ parameter-matches KAN/EQL "
                        "on 1-variable problems; compare printed param counts)")
    p.add_argument("--depth", type=int, default=2, help="number of hidden layers")
    args = p.parse_args()
    mc.set_seed(args.seed)
    device = mc.get_device(args.device)

    problem = mc.get_problem(args.problem)
    data = mc.apply_limit(mc.make_dataset(problem, noise_std=args.noise,
                                          seed=args.seed), args.limit)

    model = MLP(problem["n_vars"], hidden=args.hidden, depth=args.depth)
    mc.train_regression(model, data, epochs=args.epochs, lr=args.lr,
                        batch_size=args.batch_size, device=device)

    predict_fn = mc.torch_predict_fn(model, device)
    # expr=None -> scored as "black-box, not recoverable".
    mc.report(problem, data, predict_fn, model_name="MLP baseline",
              expr=None, save_dir=None if args.no_figure else _here())


def _here():
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    main()
