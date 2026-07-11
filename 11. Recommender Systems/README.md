# 11. Recommender Systems — Collaborative Filtering & Interaction Prediction

This directory explores Collaborative Filtering and Recommendation Systems, demonstrating how deep neural networks learn user preferences and item characteristics from sparse interaction histories.

Instead of predicting a pre-defined target label, these models learn to match users with items in a shared embedding space, predicting click-through rates (CTR) or rating scores.

All models are evaluated on the standardized **MovieLens-100k** dataset, representing movie ratings assigned by users.

---

## Two Evaluation Regimes

The folder covers two complementary framings of recommendation:

**A. Explicit rating regression** (`01`–`05`). Given a user ID $u$ and item ID $i$, predict the numeric rating $\hat{y}_{u,i} = f(e_u, e_i)$.
*   **Metrics:** MSE / MAE on held-out ratings (random 80/20 split).

**B. Implicit-feedback top-K ranking** (`06`–`08`). Binarize interactions (rating $\geq 4$ = positive) and rank the *whole catalog* for each user, judging how many held-out items land near the top.
*   **Metrics:** Recall@K and NDCG@K (K = 10, 20), computed by `rec_common.ranking_metrics_at_k`, with training items masked out.
*   **Note:** LightGCN and Mult-VAE use a per-user 20% random holdout; SASRec uses a **leave-one-out** split (single held-out final item). Because the number of relevant test items differs, SASRec's ranking numbers are *not* directly comparable to the other two — compare within a protocol, not across.

---

## Side Features (the `--features` flag)

The base task uses only `(user, item)` IDs. But Factorization-Machine-style
models exist precisely to exploit **side features**. The `--features` flag on
`03.deepfm.py`, `04.fm.py`, and `05.wide-and-deep.py` joins the MovieLens
metadata files onto every rating and switches each model to a multi-field input:

| Field | Source | Cardinality |
|---|---|---|
| user id | `u.data` | 943 |
| movie id | `u.data` | 1682 |
| gender | `u.user` | 2 |
| age group | `u.user` (bucketed) | 7 |
| occupation | `u.user` | 21 |
| release decade | `u.item` | ~9 |
| genres (multi-hot) | `u.item` | 19 |

Single-value fields are offset-encoded into one shared embedding table; the
multi-label genre field is mean-pooled. All of this lives in
`rec_common.load_movielens_features` + the shared `FeatureEmbedding` block, so
each model only implements how it *combines* fields (FM sum-of-squares, deep MLP,
wide crosses).

**Which models benefit?** Only the feature-interaction family (`03`–`05`).
Matrix Factorization and NCF have no feature slots (adding them turns MF into FM);
SASRec, LightGCN, and Mult-VAE are ID/graph/history models whose feature-aware
variants are separate architectures, so they deliberately keep the ID-only setup.

**Honest takeaway — features help most under sparsity.** On the *full* 100k
ratings, IDs already carry nearly all the signal, so side features give only a
marginal change (and DeepFM can slightly overfit the extra parameters). Restrict
the data (`--limit 15000`) to simulate a sparse / cold-start regime and the
benefit becomes clear:

| Model | Full data (ID → +features) | Sparse `--limit 15000` (ID → +features) |
|---|---|---|
| FM | 0.898 → 0.886 | **1.20 → 1.02** |
| DeepFM | 0.861 → 0.878 | 1.04 → 1.02 |
| Wide & Deep | 0.882 → 0.874 | 1.03 → 1.02 |

*(Test MSE; lower is better.)* This is the real lesson of side features: they are
insurance against sparse interaction histories and cold-start users/items, not a
free win when dense collaborative signal is already available.

---

## Utility Module: `rec_common.py`

Every script imports `rec_common.py` as `mc`. It is responsible for:
*   Downloading and parsing the MovieLens-100k rating matrix.
*   Mapping sparse User IDs and Movie IDs to continuous integer indices.
*   **Explicit regime:** `load_movielens` (rating tensors) for MSE/MAE models.
*   **Side-feature regime:** `load_movielens_features` (multi-field metadata) with the shared `FeatureEmbedding` block and `train_feature_regression` / `evaluate_feature_regression` helpers.
*   **Implicit regime:** `load_movielens_implicit` (per-user positive splits) and `load_movielens_sequences` (chronological histories), plus `ranking_metrics_at_k` / `print_ranking_metrics` for Recall@K and NDCG@K.
*   Plotting latent movie embedding clusters via t-SNE.

---

## The Catalog of Scripts

The scripts trace the evolution of collaborative filtering models. Scripts `04`
and `05` are appended by sequence number, but conceptually they are *precursors*
of DeepFM (`03`): FM (2010) is the shallow half DeepFM builds on, and Wide & Deep
(2016) is its immediate architectural ancestor.

### 01. Matrix Factorization (`01.matrix-factorization.py`)
*   **Description:** Classic latent factor collaborative filtering.
*   **Method:** Learns User and Item embedding weights. The rating prediction is computed as the dot product of the user and item embedding vectors plus user and item biases:
    $$\hat{y}_{u,i} = \mu + b_u + b_i + e_u^T e_i$$
*   **Educational Takeaway:** The mathematical foundation of collaborative filtering and representation alignment via linear dot-product layers.

### 02. Neural Collaborative Filtering (`02.ncf.py`)
*   **Description:** Non-linear collaborative filtering using neural networks (He et al., 2017).
*   **Method:** Bypasses linear dot products by concatenating user and item embeddings and feeding them through a Multi-Layer Perceptron (MLP) combined with a Generalized Matrix Factorization (GMF) layer.
*   **Educational Takeaway:** Learning non-linear, deep interactive relationships between users and items instead of simple linear dot products.

### 03. DeepFM (`03.deepfm.py`)
*   **Description:** Factorization-machine supported neural network (Guo et al., 2017).
*   **Method:** Combines a 1-order linear component, a 2-order FM component (which computes pairwise cross-feature dot products), and a deep MLP network.
*   **Educational Takeaway:** Understanding how to capture low-order and high-order feature crossings simultaneously to improve click-through rate (CTR) prediction.
*   **Side features:** supports `--features` (see the Side Features section).

### 04. Factorization Machines (`04.fm.py`)
*   **Description:** The general second-order Factorization Machine (Rendle, 2010).
*   **Method:** Models every pairwise field interaction through shared latent factors, computed in linear time with the sum-of-squares identity: $\frac{1}{2}\sum_f\left[(\sum_i v_{i,f})^2 - \sum_i v_{i,f}^2\right]$. With one-hot User/Item fields this reduces to the user-item interaction.
*   **Educational Takeaway:** The linear-time trick that lets FM scale to sparse high-cardinality features, and how Matrix Factorization with biases is just a single-field-pair special case of FM.
*   **Side features:** supports `--features` (see the Side Features section) — this is where FM most clearly beats MF.

### 05. Wide & Deep (`05.wide-and-deep.py`)
*   **Description:** Google's jointly-trained memorization + generalization framework (Cheng et al., 2016).
*   **Method:** A **wide** linear model over hashed user×item cross-product features (memorization) is summed with a **deep** embedding MLP (generalization) and trained end-to-end.
*   **Educational Takeaway:** Why memorization and generalization are complementary, the hashing trick for tractable cross features, and how DeepFM later replaces the hand-designed wide crosses with a learned FM component.
*   **Side features:** supports `--features` (see the Side Features section) — demographics/genre flow into the deep tower.

---

The models below switch to **implicit-feedback top-K ranking** (regime B), each representing a different modern paradigm: sequential, graph, and generative.

### 06. SASRec (`06.sasrec.py`)
*   **Description:** Self-Attentive Sequential Recommendation (Kang & McAuley, 2018).
*   **Method:** A causal (left-to-right) Transformer over each user's chronological history; the hidden state at each position predicts the next item via a shared item-embedding output layer. Trained with full-softmax next-item cross-entropy; evaluated leave-one-out.
*   **Educational Takeaway:** How self-attention models *sequence dynamics* — a user is represented by what they just did, not a static embedding — with a causal mask making every position a next-item predictor.

### 07. LightGCN (`07.lightgcn.py`)
*   **Description:** Light Graph Convolutional Network (He et al., 2020).
*   **Method:** Propagates user/item embeddings over the symmetrically-normalized user-item bipartite graph with **no** feature transforms or non-linearities, averaging embeddings across layers. Trained with the BPR pairwise ranking loss.
*   **Educational Takeaway:** That the useful part of a GCN for CF is neighborhood smoothing alone; stripping out the learned weights and activations both simplifies the model and improves ranking.

### 08. Mult-VAE (`08.mult-vae.py`)
*   **Description:** Variational Autoencoder with a multinomial likelihood (Liang et al., 2018).
*   **Method:** Encodes a user's full interaction vector into a latent Gaussian, samples, and decodes to a catalog-wide distribution under a multinomial reconstruction loss with KL annealing and input (denoising) dropout.
*   **Educational Takeaway:** Treating a *whole user history* as the modeling unit, and why the multinomial likelihood suits ranking implicit feedback far better than Gaussian/logistic losses.

---

## Expected Performance & Comparisons

*   **Matrix Factorization (Baseline MSE ~0.90):** Performs reliably and converges quickly. However, it is restricted to linear interactions, missing complex relationship patterns.
*   **NCF (MSE ~0.82-0.85):** MLP layers successfully extract non-linear interactions, lowering prediction error.
*   **DeepFM (MSE ~0.80-0.83):** Integrating low-order feature crossings directly with deep MLP networks yields the highest precision on sparse collaborative datasets.
*   **FM (MSE ~0.86-0.89):** Matches MF while framing it as a general pairwise-interaction model; the shallow foundation the deep models extend.
*   **Wide & Deep (MSE ~0.85-0.88):** Memorization (wide crosses) plus generalization (deep MLP) lands near DeepFM, which automates the wide half with an FM component.

### Ranking models (regime B — Recall@10 / NDCG@10, higher is better)

| Model | Protocol | Recall@10 | NDCG@10 | Notes |
|---|---|---|---|---|
| **SASRec** | leave-one-out | ~0.14 | ~0.07 | Single held-out item; not comparable to the two below. |
| **LightGCN** | 20% holdout | ~0.30 | ~0.29 | Strong, parameter-light graph baseline. |
| **Mult-VAE** | 20% holdout | ~0.30 | ~0.28 | Matches LightGCN; excels when histories are dense. |

*   **LightGCN vs. Mult-VAE:** two very different inductive biases (graph smoothing vs. generative reconstruction) reach comparable ranking quality on MovieLens-100k — a good illustration that no single paradigm dominates.
*   Numbers are from full runs on the default settings and will shift with `--epochs`, `--layers`, `--beta-cap`, etc.
