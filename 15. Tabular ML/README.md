# 15. Tabular ML — Classical & Deep Learning on Structured Data

This directory explores Machine Learning on structured **tabular** data — the
row/column databases that dominate real-world industry problems. It walks the
full arc of tabular modeling: from a single decision tree, through the
tree-ensemble methods that remain the practical champions, the classical
non-tree baselines, and finally the modern **deep-tabular** architectures
(embeddings, differentiable trees, attention, in-context learning, and KANs).

Every model is trained and evaluated under identical conditions so the
paradigms can be compared directly.

---

## Datasets

Two classification datasets (choose with `--dataset`) plus a regression track:

* **Wine** (`--dataset wine`, default) — 178 samples, 13 features, 3 cultivars.
  Tiny and near-linearly-separable; everything scores ~97-100%, so it is only a
  fast **smoke test**.
* **Forest Cover Type** (`--dataset covtype`) — 20k stratified rows (cached to
  `data/`), 54 features (10 quantitative + 4 wilderness + 40 soil one-hots),
  7 forest-cover classes. The dataset that actually **separates the models**.
* **California Housing** — 8 features, continuous target; the regression track
  exposed via `tabular_common.load_california_housing_dataset()` and the
  `report_regression` (RMSE / MAE / R²) helpers.

---

## Utility Module: `tabular_common.py`

Every script imports `tabular_common.py` as `mc`. It provides:

* Dataset loaders (Wine, Cover Type with SSL-safe download + cache, California
  Housing) and the `--dataset` dispatcher.
* `standardize()` (fit on train only) — mandatory for k-NN, SVM, and all neural
  scripts.
* The shared `build_argparser()` (tree flags *and* neural flags) and
  `get_device()`.
* Neural helpers: `get_dataloaders_from_arrays`, a `train()` loop that prints
  per-epoch accuracy **and wall-clock time**, `evaluate()`, `count_parameters()`.
* Research-grade reporting: `report_classification` (accuracy + Wilson CI +
  macro-F1 + kappa/MCC + confusion-matrix PNG) and `report_regression`.
* `plot_feature_importances()` for importance/attention-mask charts.

---

## The Catalog of Scripts

### Phase 1 — Foundations & classical baselines

| # | Script | Paradigm | Educational takeaway |
|---|--------|----------|----------------------|
| 01 | `01.decision-tree.py` | Single CART tree (Breiman, 1984) | How feature thresholds recursively partition the space; feature importance from impurity reduction. |
| 02 | `02.random-forest.py` | Bagging ensemble (Breiman, 2001) | Bootstrap + feature subsampling decorrelates trees; averaging cuts variance. |
| 03 | `03.gradient-boosting.py` | Boosting on residual gradients (Friedman, 2001) | Sequentially fitting residuals reduces bias; bagging-vs-boosting contrast. |
| 04 | `04.logistic-regression.py` | Regularized linear model | A tuned linear model is a strong baseline; L1 sparsity vs. L2 shrinkage (`--variant l2/l1/elasticnet`). |
| 05 | `05.knn.py` | k-Nearest Neighbors (Cover & Hart, 1967) | Instance-based / lazy learning; **why standardization is mandatory** (prints raw vs. scaled accuracy). |
| 06 | `06.naive-bayes.py` | Gaussian Naive Bayes | Generative vs. discriminative; conditional-independence assumption; ~free training. |
| 07 | `07.svm.py` | Support Vector Machine (Cortes & Vapnik, 1995) | Max-margin + kernel trick (`--variant linear/rbf`); the pre-GBDT SOTA that doesn't scale (O(n²)). |

### Phase 2 — Completing the ensemble story

| # | Script | Paradigm | Educational takeaway |
|---|--------|----------|----------------------|
| 08 | `08.extra-trees.py` | Extremely Randomized Trees (Geurts et al., 2006) | Random split *thresholds* on top of random features — the bias/variance dial past Random Forest. |
| 09 | `09.adaboost.py` | AdaBoost (Freund & Schapire, 1997) | Boosting by **reweighting samples** vs. script 03's **residual gradients** — same family, different derivation. |
| 10 | `10.hist-gradient-boosting.py` | Histogram GBDT (XGBoost/LightGBM paradigm) | Feature **binning** is why modern GBDTs are fast; the dependency-free stand-in for XGBoost/LightGBM/CatBoost. |
| 11 | `11.mini-xgboost.py` | **From-scratch** second-order boosting (Chen & Guestrin, 2016) | The educational centerpiece: Taylor-expanded loss, gain splits with Hessian denominators, λ/γ regularization, shrinkage — all in NumPy. |

### Phase 3 — Deep tabular models (PyTorch, from scratch)

| # | Script | Paradigm | Educational takeaway |
|---|--------|----------|----------------------|
| 12 | `12.mlp-embeddings.py` | MLP + entity embeddings (Guo & Berkhahn, 2016) | Learned embeddings vs. one-hots (`--variant embed/plain`); `--variant learnable-act` is the KAN activation ablation. |
| 13 | `13.node.py` | Neural Oblivious Decision Ensembles (Popov et al., 2019) | Differentiable oblivious trees — the bridge from trees to neural nets. |
| 14 | `14.tabnet.py` | TabNet (Arik & Pfister, 2019) | Sequential **sparsemax** attention for instance-wise feature selection; saves an aggregate-mask figure to compare against tree importances. |
| 15 | `15.ft-transformer.py` | FT-Transformer + Tabular ResNet (Gorishniy et al., 2021) | Per-feature tokenization + self-attention (`--variant ft-transformer`); the ResNet variant is the tough cheap baseline. |
| 16 | `16.toy-tabpfn.py` | Prior-Fitted Network (Hollmann et al., 2023) | **In-context learning** for tables: meta-train on synthetic tasks, then classify a real dataset in one forward pass with **zero gradient steps**. |
| 17 | `17.kan.py` | Kolmogorov-Arnold Network (Liu et al., 2024) | Learnable B-spline activations on **edges**; saves the per-feature learned-spline figure (its interpretability selling point). |

*Stretch (not yet implemented):* `18.tabm.py` — TabM (Gorishniy et al., 2024),
parameter-efficient MLP ensembling via batched heads.

---

## Trees vs. neural nets on tabular data

Test accuracy on **Forest Cover Type** (`--dataset covtype`, 16k train / 4k
test, identical split; classical/tree models at their script defaults, deep
models at 25 epochs, Toy-TabPFN meta-trained on synthetic tasks only). Numbers
are single-seed and meant for qualitative comparison, not a leaderboard (small
per-run differences are within seed noise).

| Model | Family | Test accuracy |
|-------|--------|--------------:|
| **Extra-Trees (08)** | tree ensemble | **0.847** |
| Tabular ResNet (15) | deep (MLP + residual) | 0.828 |
| MLP + one-hot (12, `plain`) | deep | 0.808 |
| MLP + embeddings (12, `embed`) | deep | 0.806 |
| HistGradientBoosting (10) | tree ensemble | 0.799 |
| KAN (17) | deep (spline edges) | 0.791 |
| k-NN, k=15 (05) | instance-based | 0.785 |
| FT-Transformer (15) | deep (attention) | 0.783 |
| Mini-XGBoost (11, 6k subset) | tree ensemble (from scratch) | 0.766 |
| NODE (13) | deep (differentiable trees) | 0.741 |
| Logistic Regression (04) | linear | 0.725 |
| AdaBoost, depth-3 (09) | tree ensemble | 0.622 |
| TabNet (14) | deep (attention) | 0.524 |
| Gaussian NB (06) | generative | 0.438 |
| Toy-TabPFN (16) | in-context, zero-grad | 0.412 |

Reading the table:

* **Tree ensembles win the top spot** (Extra-Trees, 0.85) — the recurring
  headline of tabular ML.
* **The cheap deep baseline beats the fancy one:** the paper's own Tabular
  ResNet (0.83) outscores its FT-Transformer (0.78) here, and a plain MLP (0.81)
  ties the embedding MLP — always benchmark the simple variant first.
* **KAN does not beat a same-size MLP** (0.79 vs. 0.81) while training several
  times slower per epoch — exactly the honest takeaway of Yu et al. (2024); its
  value is the per-feature interpretability figure, not accuracy.
* **TabNet and Toy-TabPFN lag** on this hard 7-class, 54-feature problem: TabNet
  is famously finicky (this compact version omits its sparsity loss and ghost
  BN), and the toy PFN's synthetic random-MLP prior + capped 1000-row context is
  a deliberately weak stand-in for real TabPFN — both are here to demonstrate the
  *mechanism*, and both do far better on the small Wine dataset.

**The recurring lesson of tabular ML:** tree ensembles (Phase 2) remain the
strongest and cheapest models on this kind of structured data, and a
parameter-matched MLP is a very hard-to-beat deep baseline. The deep-tabular
architectures are conceptually rich — attention masks, differentiable trees,
in-context learning, spline activations — but on standard tabular benchmarks
they typically **match** rather than beat well-tuned GBDTs (see Grinsztajn et
al. 2022; and for KANs specifically, Yu et al. 2024). Reach for the deep models
when you need what they *uniquely* offer: entity embeddings for
high-cardinality categoricals, in-context/zero-training prediction, or per-edge
interpretability.

---

## Running

```bash
cd "15. Tabular ML"
python3 "10.hist-gradient-boosting.py" --dataset covtype       # full run
python3 "17.kan.py" --limit 2000 --epochs 2                    # fast smoke test
python3 "12.mlp-embeddings.py" --variant plain --no-figure     # skip figures
```

All scripts share `--dataset {wine,covtype}`, `--limit N`, `--seed`, and
`--no-figure`; deep-tabular scripts add `--epochs`, `--batch-size`, and
`--device`. Figures are written to this directory.
