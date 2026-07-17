"""
01. Genetic Programming — Tree-based Symbolic Regression (Koza, 1992)
====================================================================

The original symbolic-regression method and still the mental model for the whole
field.  A candidate solution is an **expression tree** built from a primitive
set; a population of random trees is *evolved* by Darwinian selection — fitter
trees breed, their subtrees recombine, and mutation injects novelty — until a
tree fits the data.  The output is not weights but an actual formula.

    Primitives (this script):
        functions : +  -  *  /(protected)  sin  cos  exp  sqrt(|.|)
        terminals : x0..x{d-1}  and ephemeral random constants

    An expression tree for  x^3 + x^2 + x :

                    (+)
                   /   \
                (+)     x
               /   \
            (*)     (*)
           /   \   /   \
          x    (*) x    x
              /  \
             x    x

Evolution loop (one generation):
    evaluate fitness -> tournament-select parents -> subtree crossover
    -> subtree / point mutation -> elitism -> next generation

Fitness = RMSE(tree, data) + parsimony_coeff * tree_size.  The parsimony term is
GP's Occam pressure: without it trees "bloat" into huge unreadable formulas that
memorize noise.  Protected operators (division guarded against /0, sqrt of |x|,
exp clipped) keep the numeric search well-defined.

Educational takeaway:
    GP searches *expression space directly* — a discrete, combinatorial search,
    not gradient descent.  It recovers exact structure on clean problems and is
    the reference for "strong recovery" here, but it is stochastic and scales
    poorly (population x generations x tree evaluations).  Contrast with script
    02 (SINDy), which turns the same goal into instant linear algebra when the
    true terms happen to live in a fixed library.

Run:
    python "01.genetic-programming.py" --problem nguyen1
    python "01.genetic-programming.py" --problem kinetic --generations 60
    python "01.genetic-programming.py" --problem nguyen1 --limit 500   # smoke test
"""

import os
import numpy as np
import sr_common as mc

# --------------------------------------------------------------------------- #
# Primitive set
# --------------------------------------------------------------------------- #
# Each function: (name, arity, numpy impl, pretty-printer).  Binary/unary impls
# are "protected" so no candidate ever produces NaN/inf during evolution.

def _pdiv(a, b):
    return np.divide(a, b, out=np.ones_like(a), where=np.abs(b) > 1e-6)


def _pexp(a):
    return np.exp(np.clip(a, -30.0, 30.0))


def _psqrt(a):
    return np.sqrt(np.abs(a))


FUNCS = {
    "add": (2, np.add, lambda a, b: f"({a} + {b})"),
    "sub": (2, np.subtract, lambda a, b: f"({a} - {b})"),
    "mul": (2, np.multiply, lambda a, b: f"({a} * {b})"),
    "div": (2, _pdiv, lambda a, b: f"({a} / {b})"),
    "sin": (1, np.sin, lambda a: f"sin({a})"),
    "cos": (1, np.cos, lambda a: f"cos({a})"),
    "exp": (1, _pexp, lambda a: f"exp({a})"),
    "sqrt": (1, _psqrt, lambda a: f"sqrt({a})"),
}
FUNC_NAMES = list(FUNCS)


class Node:
    """A node in an expression tree.

    kind is one of: 'func' (op in FUNCS), 'var' (idx), 'const' (value).
    """
    __slots__ = ("kind", "op", "idx", "value", "children")

    def __init__(self, kind, op=None, idx=None, value=None, children=None):
        self.kind = kind
        self.op = op
        self.idx = idx
        self.value = value
        self.children = children or []

    # ---- evaluation ------------------------------------------------------- #
    def eval(self, X):
        if self.kind == "var":
            return X[:, self.idx]
        if self.kind == "const":
            return np.full(X.shape[0], self.value)
        impl = FUNCS[self.op][1]
        args = [c.eval(X) for c in self.children]
        with np.errstate(all="ignore"):
            return impl(*args)

    # ---- pretty printing -------------------------------------------------- #
    def to_str(self, var_names):
        if self.kind == "var":
            return var_names[self.idx]
        if self.kind == "const":
            return f"{self.value:.4g}"
        printer = FUNCS[self.op][2]
        return printer(*[c.to_str(var_names) for c in self.children])

    # ---- structural helpers ---------------------------------------------- #
    def size(self):
        return 1 + sum(c.size() for c in self.children)

    def clone(self):
        return Node(self.kind, self.op, self.idx, self.value,
                    [c.clone() for c in self.children])

    def all_nodes(self):
        yield self
        for c in self.children:
            yield from c.all_nodes()


# --------------------------------------------------------------------------- #
# Random tree generation (ramped half-and-half)
# --------------------------------------------------------------------------- #
class GP:
    def __init__(self, n_vars, rng, pop_size=300, generations=40,
                 tournament=5, p_crossover=0.8, p_mutation=0.2,
                 max_depth=5, parsimony=0.002, const_range=(-2.0, 2.0)):
        self.n_vars = n_vars
        self.rng = rng
        self.pop_size = pop_size
        self.generations = generations
        self.tournament = tournament
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.max_depth = max_depth
        self.parsimony = parsimony
        self.const_range = const_range

    # ---- terminals & random subtrees -------------------------------------- #
    def _terminal(self):
        if self.rng.random() < 0.7:
            return Node("var", idx=int(self.rng.integers(self.n_vars)))
        lo, hi = self.const_range
        return Node("const", value=float(self.rng.uniform(lo, hi)))

    def random_tree(self, depth, method):
        """`method` in {'full', 'grow'} — Koza's ramped half-and-half."""
        if depth <= 0:
            return self._terminal()
        if method == "grow" and self.rng.random() < 0.3:
            return self._terminal()
        op = FUNC_NAMES[int(self.rng.integers(len(FUNC_NAMES)))]
        arity = FUNCS[op][0]
        children = [self.random_tree(depth - 1, method) for _ in range(arity)]
        return Node("func", op=op, children=children)

    def init_population(self):
        pop = []
        for i in range(self.pop_size):
            depth = 2 + int(self.rng.integers(self.max_depth - 1))
            method = "full" if i % 2 == 0 else "grow"
            pop.append(self.random_tree(depth, method))
        return pop

    # ---- fitness ---------------------------------------------------------- #
    def fitness(self, tree, X, y):
        with np.errstate(all="ignore"):
            pred = tree.eval(X)
        if not np.all(np.isfinite(pred)):
            return 1e9
        err = float(np.sqrt(np.mean((pred - y) ** 2)))
        return err + self.parsimony * tree.size()

    # ---- genetic operators ------------------------------------------------ #
    def _select(self, pop, fits):
        idx = self.rng.integers(0, len(pop), size=self.tournament)
        best = idx[np.argmin([fits[i] for i in idx])]
        return pop[best]

    def _random_point(self, tree):
        nodes = list(tree.all_nodes())
        # Bias toward internal (function) nodes 90% of the time (Koza).
        funcs = [n for n in nodes if n.kind == "func"]
        if funcs and self.rng.random() < 0.9:
            return funcs[int(self.rng.integers(len(funcs)))]
        return nodes[int(self.rng.integers(len(nodes)))]

    def crossover(self, a, b):
        child = a.clone()
        donor = b.clone()
        # Replace a random subtree of `child` with a random subtree of `donor`.
        target = self._random_point(child)
        source = self._random_point(donor)
        target.kind, target.op, target.idx, target.value, target.children = (
            source.kind, source.op, source.idx, source.value, source.children)
        return child

    def mutate(self, a):
        child = a.clone()
        node = self._random_point(child)
        if node.kind == "const" and self.rng.random() < 0.5:
            # point mutation: perturb a constant
            node.value += float(self.rng.normal(0, 0.5))
        else:
            # subtree mutation: grow a fresh subtree in place
            fresh = self.random_tree(1 + int(self.rng.integers(3)), "grow")
            node.kind, node.op, node.idx, node.value, node.children = (
                fresh.kind, fresh.op, fresh.idx, fresh.value, fresh.children)
        return child

    # ---- main loop -------------------------------------------------------- #
    def evolve(self, X, y, var_names, verbose=True):
        pop = self.init_population()
        best_tree, best_fit = None, float("inf")
        for gen in range(self.generations):
            fits = [self.fitness(t, X, y) for t in pop]
            order = np.argsort(fits)
            if fits[order[0]] < best_fit:
                best_fit = fits[order[0]]
                best_tree = pop[order[0]].clone()
            if verbose:
                rmse_only = self.fitness(best_tree, X, y) - self.parsimony * best_tree.size()
                print(f"gen {gen:3d} | best fitness {best_fit:.5f} | "
                      f"rmse {rmse_only:.5f} | size {best_tree.size():3d} | "
                      f"{best_tree.to_str(var_names)[:70]}")
            # Elitism: carry the top few unchanged.
            new_pop = [pop[order[i]].clone() for i in range(min(2, len(pop)))]
            while len(new_pop) < self.pop_size:
                if self.rng.random() < self.p_crossover:
                    child = self.crossover(self._select(pop, fits),
                                           self._select(pop, fits))
                else:
                    child = self._select(pop, fits).clone()
                if self.rng.random() < self.p_mutation:
                    child = self.mutate(child)
                # depth guard against bloat
                if child.size() <= 60:
                    new_pop.append(child)
            pop = new_pop
        return best_tree, best_fit


def simplify_constants(tree):
    """Fold any subtree with no variables into a single constant node."""
    if tree.kind != "func":
        return tree
    tree.children = [simplify_constants(c) for c in tree.children]
    if all(c.kind == "const" for c in tree.children):
        val = float(tree.eval(np.zeros((1, 1)))[0])
        return Node("const", value=val)
    return tree


def main():
    p = mc.build_argparser("Genetic Programming symbolic regression")
    p.add_argument("--pop-size", type=int, default=300, help="population size")
    p.add_argument("--generations", type=int, default=40, help="generations")
    p.add_argument("--parsimony", type=float, default=0.002,
                   help="parsimony coefficient (size penalty)")
    args = p.parse_args()
    mc.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    problem = mc.get_problem(args.problem)
    data = mc.apply_limit(mc.make_dataset(problem, noise_std=args.noise,
                                          seed=args.seed), args.limit)

    pop_size, generations = args.pop_size, args.generations
    if args.limit is not None:  # smoke-test budget: shrink pop x gens
        pop_size = min(pop_size, 60)
        generations = min(generations, 15)

    gp = GP(problem["n_vars"], rng, pop_size=pop_size, generations=generations,
            parsimony=args.parsimony)
    print(f"GP: pop={pop_size} gens={generations} "
          f"primitives={FUNC_NAMES} vars={problem['vars']}")
    best, _ = gp.evolve(data.X_train, data.y_train, problem["vars"])
    best = simplify_constants(best)
    expr = best.to_str(problem["vars"])
    print(f"\nFinal expression: {expr}")

    predict_fn = lambda X: best.eval(np.asarray(X, np.float64))
    mc.report(problem, data, predict_fn, model_name="Genetic Programming",
              expr=expr, save_dir=None if args.no_figure else _here())


def _here():
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    main()
