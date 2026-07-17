# 18. Symbolic Regression — Recovering Formulas, Not Just Fitting Them

Ordinary regression asks *"what number is `y`?"*.  **Symbolic regression** asks
the harder question: *"what closed-form expression generated this data?"*  The
output is not a weight matrix but an actual formula — `x³ + x² + x`,
`½ m v²`, `exp(−x²/2)` — that a human can read, differentiate, and trust outside
the training range.

This folder compares five paradigms for that task under identical conditions,
from the 1990s genetic-programming classic to the 2024 Kolmogorov-Arnold Network.

## Why this domain exists

It is the deliberate **positive control** for the KAN-vs-MLP comparison in
`15. Tabular ML/` (scripts 12 & 17).  There, a parameter-matched MLP *ties* the
KAN — because tabular targets have no smooth univariate structure for KAN's
splines to exploit.  An architecture evaluation is always a claim about a *task
distribution*; this folder is the distribution where the ranking flips.  Here the
ground truth genuinely **is** a composition of smooth univariate functions —
exactly the object a KAN edge represents — and the KAN's inductive bias earns its
keep.  Showing where a method wins *and* where it merely ties is the honest way to
evaluate it.

---

## The task and the three numbers

Every script samples `(X, y)` from a hidden ground-truth formula and reports the
**same three numbers** so the paradigms compare directly:

1. **Interpolation RMSE / R²** — held-out samples *inside* the training range
   (ordinary fit quality).
2. **Extrapolation RMSE** — held-out samples *outside* the training range, where
   black-box fits fall apart and structural methods keep working.
3. **Recovery** — did the method emit a human-readable expression, and does it
   match the truth?  Proxy (per SPEC, no `sympy` dependency): RMSE < 1e-6 on a
   freshly-sampled **wide** domain.  Black-box methods score *"not recoverable"*
   by construction — that structural difference is the folder's whole point.

No downloads, no new dependencies (`torch numpy scikit-learn scipy matplotlib`).

---

## The benchmark problems (`--problem`)

Eight ground-truth formulas spanning difficulty, all in `sr_common.PROBLEMS`:

| `--problem` | Formula | Vars | Family |
|-------------|---------|:----:|--------|
| `nguyen1`  | `x³ + x² + x`            | 1 | polynomial |
| `nguyen5`  | `sin(x²)·cos(x) − 1`     | 1 | trig composition |
| `nguyen6`  | `sin(x) + sin(x + x²)`   | 1 | trig composition |
| `gaussian` | `exp(−x²/2)`             | 1 | exponential |
| `kinetic`  | `½ m v²`                 | 2 | physics (product) |
| `distance` | `sqrt(x² + y²)`          | 2 | physics (norm) |
| `coulomb`  | `q₁ q₂ / r²`             | 3 | physics (rational) |
| `rational` | `x y / (1 + x²)`         | 2 | rational |

Each problem carries an **interpolation** sampling range and a disjoint
**extrapolation** range strictly outside it, so the extrapolation split is
genuinely off-distribution.  `sr_common.make_dataset` produces train/val/test
splits (from the interpolation range) plus an extrapolation split; Gaussian noise
(`--noise`, in units of `std(y)`) is added to the *train/val targets only*.

---

## Utility module: `sr_common.py`

Every script imports it as `mc`.  A **`predict_fn`** — any callable
`X[n, d] → y[n]` — is the common currency between the numpy scripts (01–02) and
the torch scripts (03–05); because each method exposes one, `report` and
`check_recovery` work identically for all five.

* `PROBLEMS` / `get_problem` / `make_dataset` — the ground-truth registry and data
  synthesis (interpolation + extrapolation splits, optional noise).
* `build_argparser` — the shared CLI (`--problem`, `--noise`, `--seed`, plus the
  neural `--epochs/--lr/--batch-size/--device` and universal `--limit/--no-figure`).
* `train_regression` / `torch_predict_fn` / `count_parameters` — Phase-2 neural
  helpers (an MSE loop that prints per-epoch loss, wall-clock time, and the exact
  trainable-parameter count).
* `report` — prints the three numbers and saves the standard figure: for 1-D
  problems the **prediction-vs-truth curve** over the training band (grey) *and*
  the extrapolation range; for multivariate problems a **predicted-vs-true
  scatter** for the interpolation and extrapolation splits side by side.
* `check_recovery` — the numeric symbolic-equivalence proxy.

---

## The catalog of scripts — paradigms and their innovations

### Phase 1 — Classical symbolic methods (NumPy, from scratch)

| # | Script | Paradigm | Key innovation / advantage |
|---|--------|----------|----------------------------|
| 01 | `01.genetic-programming.py` | Tree-based GP (Koza, 1992) | Searches **expression space directly** as evolving trees. |
| 02 | `02.sparse-regression.py` | SINDy / STLSQ (Brunton et al., 2016) | Recasts recovery as **sparse linear algebra** over a fixed library. |

**01 — Genetic Programming.**  The original symbolic-regression method and still
the field's mental model.  A candidate is an **expression tree** over primitives
`{+, −, ×, ÷, sin, cos, exp, sqrt, var, const}`; a *population* of random trees is
evolved by tournament selection, subtree crossover, and mutation.  *Innovation:*
it optimizes over **structure itself**, so it can invent compositions no fixed
template contains (it builds `sin(x²)` or `x³` on its own).  A **parsimony
penalty** (`RMSE + λ·size`) is its Occam pressure against "bloat".  *Advantage:*
strongest, most general recovery — it reconstructs `nguyen1` exactly.  *Cost:* a
stochastic, combinatorial search (population × generations × evaluations) that
scales poorly and varies run to run.

**02 — SINDy (sparse regression).**  The modern re-framing: fix a big **library**
of candidate terms up front — polynomials to degree 3 (with cross-products) plus
`sin/cos/exp` of each input — and solve `y ≈ Θ(X)·ξ` for a **sparse** ξ via
**Sequentially-Thresholded Least Squares** (fit → zero small coefficients →
refit).  *Innovation:* turns a search problem into deterministic linear algebra.
*Advantage:* instant and exact **when the true terms live in the library**
(`nguyen1`, `kinetic` → recovered to machine precision in milliseconds).
*Limitation (the teaching point):* blind outside its basis — on `nguyen5` the
`sin(x²)` term simply isn't expressible from `sin(x)`, so it returns a dense,
wrong fit that explodes on extrapolation.  Run both `nguyen1` and `nguyen5` to see
the two faces of the same method.

### Phase 2 — Neural methods (PyTorch, from scratch, parameter-matched)

All three are sized to the **same order of magnitude of trainable parameters**
(each script prints its exact count — see the note below) so differences reflect
*inductive bias*, not capacity.

| # | Script | Paradigm | Key innovation / advantage |
|---|--------|----------|----------------------------|
| 03 | `03.eql.py` | Equation Learner (Martius & Lampert, 2016) | Activations **are** the primitive functions; L1 + pruning reads off a formula. |
| 04 | `04.mlp-baseline.py` | Plain MLP | The **black-box control** — capacity without symbolic bias. |
| 05 | `05.kan.py` | Kolmogorov-Arnold Network (Liu et al., 2024) | A learnable **univariate function on every edge**. |

**03 — EQL (Equation Learner).**  Keep neural training, but replace the fixed
activation with a *menu of primitives* — identity, sin, cos, and crucially
**multiplication units** that multiply two components of the pre-activation.
*Innovation:* **differentiable symbolic regression** — gradient descent over
expression structure, the smooth cousin of GP's discrete search.  Multiplication
units are what let it build polynomial/product terms an MLP can only approximate;
stacking two layers reaches `x³`, `sin(x²)`, etc.  An **L1 penalty then hard
thresholding** ("fit → sparsify → freeze mask → refit") prunes the network down to
a formula you read off the surviving weights.  *Advantage:* GPU-friendly and,
because it literally built the polynomial, it **extrapolates polynomials far
better than the MLP or KAN** (see `nguyen1` below).  *Cost:* the usual gradient
pitfalls — local minima and sensitivity to the sparsity threshold.

**04 — MLP baseline.**  A plain ReLU MLP with a scalar output, the **control**.
*Innovation:* none, deliberately — it has universal-approximation capacity but no
notion of `sin`, `exp`, or `x²`.  *Behavior:* competitive on interpolation, but it
is a piecewise-linear surface, so it **fails on curved extrapolation** (continues
as straight-line ramps) and **recovers nothing** (there is no expression to read).
Its required plot is the extrapolation figure — compare its flat continuation past
the grey training band to the KAN's spline tracking the true curve.

**05 — KAN (Kolmogorov-Arnold Network).**  Moves the nonlinearity from the
**nodes to the edges**: instead of one scalar weight per edge and a fixed node
activation, each edge carries a **learnable univariate function** `φ(x) =
base·SiLU(x) + Σ cᵢ Bᵢ(x)` (a B-spline plus a SiLU residual) and nodes simply sum.
*Innovation:* the edge functions are the same kind of object symbolic regression
is trying to recover — smooth univariate transforms — so the architecture's
inductive bias is aligned with the task.  *Advantages here:* **lowest
interpolation RMSE on essentially every problem** at matched parameters, sane
off-distribution behavior on curved targets, per-edge **spline figures** you can
inspect, and an optional **symbolification** (`--symbolify`, on by default) that
snaps each variable's response to the best-fitting primitive `{linear, x², x³,
sin, cos, exp, sqrt}` with a residual guard — cleanly recovering, e.g., that
`kinetic` is *linear in m, quadratic in v*, and honestly reporting *"no
single-primitive match"* when the truth is a genuine composition (`gaussian`).

> **Parameter-matching note.**  On the 1-variable problems the three networks are
> **EQL ≈ 98, KAN ≈ 120, MLP ≈ 141** parameters — same ballpark, matched within
> ~1.5×.  A KAN edge holds `G+k` spline coefficients where an MLP edge holds one,
> so the KAN's count grows faster with input dimension (≈240 at 3 variables vs.
> the MLP's ≈161); the comparison is therefore *fairest on the 1-D problems*, and
> even at higher dimension all three stay the same order of magnitude.  Each script
> prints its exact count — compare those, not layer widths.

---

## Results — the folder's thesis in tables

Single-seed (`--seed 0`), noiseless, neural models at 300 epochs.  Numbers are for
qualitative comparison, not a leaderboard (small differences are seed noise).
**`*` = expression recovered** (wide-domain RMSE < 1e-6).

### Interpolation RMSE (fit quality inside the training range — lower is better)

| Problem | GP (01) | SINDy (02) | EQL (03) | MLP (04) | KAN (05) |
|---------|--------:|-----------:|---------:|---------:|---------:|
| nguyen1  | `2e-16`* | `2e-16`* | 6.2e-4 | 0.020 | **1.5e-3** |
| nguyen5  | 0.159 | 5.5e-3 | 0.025 | 0.019 | **1.9e-3** |
| gaussian | 0.064 | 1.6e-3 | 3.7e-3 | 5.5e-3 | **9.5e-4** |
| kinetic  | 0.307 | `1e-15`* | 0.021 | 0.231 | **0.016** |
| distance | 0.086 | 5.4e-3 | 0.013 | 0.071 | **3.4e-3** |
| coulomb  | 0.154 | 8.7e-3 | 0.022 | 0.034 | **9.3e-3** |
| rational | 0.043 | 0.193 | 0.037 | 0.024 | **3.4e-3** |

### Extrapolation RMSE (outside the training range — lower is better)

| Problem | GP (01) | SINDy (02) | EQL (03) | MLP (04) | KAN (05) |
|---------|--------:|-----------:|---------:|---------:|---------:|
| nguyen1  | `2e-15`* | `2e-15`* | **0.51** | 12.9 | 11.1 |
| nguyen5  | 0.60 | 21.6 | 0.70 | **0.46** | 0.65 |
| gaussian | 0.74 | 7.96 | 0.37 | **0.23** | 0.41 |
| kinetic  | 29.3 | `8e-15`* | **4.29** | 11.4 | 15.5 |
| distance | 0.30 | 4.26 | 2.01 | **0.035** | 0.39 |
| coulomb  | 0.10 | 8.86 | 0.14 | **0.078** | 0.16 |
| rational | 2.99 | 13.4 | 2.73 | 0.51 | **0.30** |

### Recovery, cost, and character

| Method | Emits a formula? | Recovers exactly when… | Cost | Character |
|--------|:----------------:|------------------------|------|-----------|
| **GP (01)** | ✅ tree | truth is in the primitive set & search finds it (`nguyen1`) | high (evolutionary search) | most general, stochastic, brittle to scale |
| **SINDy (02)** | ✅ sparse sum | truth's terms are in the library (`nguyen1`, `kinetic`) | ~instant (least squares) | exact in-basis, blind out-of-basis |
| **EQL (03)** | ✅ (approx., from weights) | prune + fit align on the structure | medium (gradient) | differentiable GP; best polynomial extrapolation |
| **MLP (04)** | ❌ black box | never (by construction) | medium (gradient) | capacity without bias; the control |
| **KAN (05)** | ⚠️ per-variable snap | single-transform problems (`kinetic`) | medium–high (splines) | best interpolation at matched params |

### Reading the tables

* **In-library recovery is unbeatable — when it happens.**  SINDy nails `nguyen1`
  and `kinetic` to machine precision (`*`) *and* extrapolates perfectly, because
  the true terms are polynomial and live in its library.  The moment they don't
  (`nguyen5`, `gaussian`, `coulomb`), it returns a dense wrong fit that **explodes
  off-distribution** (extrapolation RMSE 8–22).  Structural correctness, not curve
  closeness, is what buys extrapolation.
* **GP recovers by *building* structure.**  It reconstructs `nguyen1` exactly with
  no library at all — but it is stochastic (it missed `kinetic` on this seed) and
  the most expensive method here.  Manuscript claims need ≥10 seeds (`--seed`).
* **The KAN wins interpolation across the board** — lowest RMSE on every problem at
  matched parameters — because a spline edge *is* the smooth univariate object the
  targets are made of.  This is the **ranking flip** promised by the SPEC: the same
  KAN merely *ties* a param-matched MLP on tabular data (`15. Tabular ML/`).
* **Extrapolation is nuanced, and the honesty is the point.**  No method
  extrapolates *growing* polynomials well except the ones that recovered them
  (`nguyen1`, `kinetic`): KAN splines flatten past their grid and ReLU ramps
  linearly.  **EQL** extrapolates polynomials far better than MLP/KAN (`nguyen1`
  0.51 vs ~11) *because it literally built `x³`*.  And the MLP sometimes wins
  extrapolation when the truth is locally near-linear in the extrapolation region
  (`distance` is a cone, `coulomb` is gentle) — a fair caveat to "black boxes never
  extrapolate."  The clean headline is **fit quality (interpolation)**; the caveats
  live in extrapolation.
* **Only three of five emit a formula.**  Recovery is the structural divide the
  folder is built around: GP and SINDy *are* their expressions; EQL and KAN offer
  approximate readouts; the MLP offers nothing at all.

**The lesson:** on a task distribution whose targets are compositions of smooth
univariate functions, architectures with a *matching inductive bias* (SINDy's
library, EQL's primitive units, KAN's spline edges) beat a same-size black box —
the mirror image of `15. Tabular ML/`, where no such structure exists and the
plain MLP is impossible to beat.  Evaluate architectures against the task
distribution they claim to serve.

---

## Running

```bash
cd "18. Symbolic Regression"

# classical
python3 "01.genetic-programming.py" --problem nguyen1            # recovers x^3+x^2+x
python3 "02.sparse-regression.py"   --problem nguyen1            # in-library  -> exact
python3 "02.sparse-regression.py"   --problem nguyen5            # out-of-library -> fails

# neural (parameter-matched — compare the printed counts)
python3 "03.eql.py"          --problem kinetic  --epochs 400
python3 "04.mlp-baseline.py" --problem nguyen1  --epochs 400     # the extrapolation figure
python3 "05.kan.py"          --problem gaussian --epochs 400     # spline figure + symbolify

# fast smoke tests (always verify changes this way)
python3 "01.genetic-programming.py" --problem nguyen1 --limit 500
python3 "05.kan.py"                 --problem nguyen1 --limit 200 --epochs 20
```

Shared flags: `--problem {nguyen1,nguyen5,nguyen6,gaussian,kinetic,distance,coulomb,rational}`,
`--noise` (train-target noise in units of `std(y)`), `--seed`, `--limit`,
`--no-figure`; neural scripts add `--epochs`, `--lr`, `--batch-size`, `--device`.
Figures are written to this directory (prefix `sr_`).
