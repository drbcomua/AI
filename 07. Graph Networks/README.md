# 07. Graph Networks — Message Passing on Non-Euclidean Spaces

This directory explores Graph Neural Networks (GNNs), demonstrating how convolutions generalize to non-Euclidean data structured as nodes and edges.

It features the three core graph tasks, plus self-supervised and generative paradigms:
1.  **Node Classification (Semi-Supervised):** Evaluated on the **Cora Citation Network** — classify documents into subject categories, using citation edges for context.
2.  **Graph Classification (Supervised):** Evaluated on the **MUTAG** molecular dataset — predict whether a molecule is mutagenic by aggregating the whole graph.
3.  **Link Prediction (Unsupervised):** Also on Cora — hold out edges and score whether a node pair should be connected (GAE/VGAE).

---

## Utility Module: `gnn_common.py`

Every script imports `gnn_common.py` as `mc`. It handles:
*   Downloading and parsing Cora and MUTAG datasets.
*   Formatting adjacency matrices and computing normalized graph Laplacians.
*   Splitting node indices into train, validation, and test indices.
*   Training loops: Backpropagating over node-level or graph-level losses.
*   Graph pooling functions (Global Mean Pooling, Global Sum Pooling) for graph-level classification.
*   Metrics reporting (Accuracy, F1 score).

---

## The Catalog of Scripts

The scripts trace GNN convolutions, aggregations, and architectures:

### 01. Baselines (`01.baselines.py`)
*   **Description:** Non-relational vs. simple structural baselines.
*   **Models:**
    *   *MLP:* Predicts node label using only individual node features (word bags), ignoring citation edges entirely.
    *   *Label Propagation:* A classical algorithm that spreads node labels across neighboring edges without learning features.
*   **Educational Takeaway:** Setting limits. Seeing how much features alone help vs. how much structure alone helps.

### 02. Graph Convolutional Network (`02.gcn.py`)
*   **Description:** Neighborhood localized spectral graph convolutions (Kipf & Welling, 2016).
*   **Method:** Aggregates a node's features with its neighbors' features weighted by their degree using a normalized adjacency matrix with self-loops:
    $$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$
*   **Educational Takeaway:** Understanding how spatial convolution extends to irregular grid structures.

### 03. Graph Attention Network (`03.gat.py`)
*   **Description:** Anisotropic message passing with self-attention (Veličković et al., 2017).
*   **Method:** Neighborhood weights are not determined statically by degrees; they are calculated dynamically using learnable self-attention coefficients over node feature pairs.
*   **Educational Takeaway:** The benefits of letting the model dynamically weight which neighbors' information is most relevant.

### 04. GraphSAGE (`04.graphsage.py`)
*   **Description:** Inductive representation learning via neighborhood sampling (Hamilton et al., 2017).
*   **Method:** Instead of computing over the entire graph, it samples a fixed-size local neighborhood and aggregates them using pooling functions (Mean, LSTM, or Max Pooling).
*   **Educational Takeaway:** Solving the scaling limits of spectral GNNs to enable processing massive, dynamic graphs.

### 05. Graph Isomorphism Network (`05.gin.py`)
*   **Description:** Highly expressive GNN (Xu et al., 2018).
*   **Method:** Multi-layer message passing designed to match the power of the Weisfeiler-Lehman (WL) graph isomorphism test. Uses global pooling to generate graph-level embeddings for MUTAG.
*   **Educational Takeaway:** Learning graph classification tasks. Understanding why sum aggregation makes models more expressive at distinguishing graph topologies than mean or max pooling.

### 06. Graph Autoencoders — GAE / VGAE (`06.gae-vgae.py`)
*   **Description:** Link prediction via autoencoding (Kipf & Welling, 2016). `--variant gae|vgae`.
*   **Method:** A GCN encodes nodes to embeddings; the decoder reconstructs the adjacency as `sigmoid(Z Z^T)`. 10% of edges are held out and scored vs sampled non-edges (ROC-AUC / Average Precision). VGAE adds a variational latent + KL term.
*   **Educational Takeaway:** The third core graph task (link prediction) and the unsupervised/generative paradigm.

### 07. Graph Transformer (`07.graph-transformer.py`)
*   **Description:** Transformer with attention masked to graph neighborhoods + Laplacian positional encoding (Dwivedi & Bresson, 2020).
*   **Method:** Dot-product multi-head attention over each node's edges, with the lowest Laplacian eigenvectors as structural "coordinates."
*   **Educational Takeaway:** Generalizes GAT's attention; masking to edges injects the graph prior (unrestricted global attention overfits Cora's 140 labels).

### 08. Node2vec / DeepWalk (`08.node2vec.py`)
*   **Description:** Random-walk + skip-gram node embeddings (Perozzi 2014; Grover & Leskovec 2016). `--variant "p,q"` for biased walks.
*   **Method:** Generate random walks, train skip-gram with negative sampling, then a linear probe. Uses **no node features** — all signal from connectivity.
*   **Educational Takeaway:** The pre-GNN, structure-only embedding approach; homophily alone already clusters classes.

### 09. SGC & APPNP (`09.sgc-appnp.py`)
*   **Description:** Decoupling propagation from transformation. `--variant sgc|appnp`.
*   **Method:** SGC removes all nonlinearities (`A^K·X` + linear); APPNP predicts per-node with an MLP then propagates via personalized PageRank.
*   **Educational Takeaway:** A GCN is largely a low-pass filter + linear classifier; propagation can be decoupled to go many hops without oversmoothing.

### 10. Oversmoothing — GCN vs GCNII (`10.oversmoothing.py`)
*   **Description:** The signature GNN pathology and its fix (GCNII, Chen et al., 2020).
*   **Method:** Trains both at depths {2,4,8,16,32} and plots accuracy vs depth. Plain GCN collapses (≈0.78 → 0.15); GCNII's initial-residual + identity mapping stays ≈0.79.
*   **Educational Takeaway:** Why naively deep GNNs fail, and the residual ideas that resolve it.

### 11. ChebNet (`11.chebnet.py`)
*   **Description:** Spectral graph convolution via Chebyshev polynomials of the scaled Laplacian (Defferrard et al., 2016).
*   **Educational Takeaway:** The spectral filter GCN descends from; K-localized and eigendecomposition-free.

### 12. DiffPool (`12.diffpool.py`)
*   **Description:** Differentiable hierarchical graph pooling for MUTAG (Ying et al., 2018).
*   **Method:** A GNN learns a soft assignment of nodes to clusters; the graph is coarsened (`X'=S^T Z`, `A'=S^T A S`) with auxiliary link-prediction + entropy losses.
*   **Educational Takeaway:** Learns a graph hierarchy (like CNN downsampling) instead of GIN's single flat readout.

### 13. Deep Graph Infomax (`13.dgi.py`)
*   **Description:** Self-supervised node representations by mutual-information maximization (Velickovic et al., 2019).
*   **Method:** Train the encoder so real-graph node embeddings agree with a global summary while a feature-shuffled corruption does not; then a linear probe on frozen embeddings.
*   **Educational Takeaway:** Contrastive, label-free pretraining that rivals supervised GCN.

---

## Expected Performance & Comparisons

Running these models on **Cora** (Node Classification) and **MUTAG** (Graph Classification) highlights the progress in GNN architectures:

### 1. Isolated Baselines vs. Graph Neural Networks (Cora)
*   **MLP Feature Baseline (~55-60% Test Accuracy):** Classifies documents using only the local word attributes, completely ignoring citation structures. It performs moderately because text is indicative, but lacks network context.
*   **Label Propagation (~68-72% Test Accuracy):** Propagates class labels strictly across citation links, completely ignoring document text. It performs well because citations are highly homophilic, but fails when links are sparse or noisy.
*   **GCN, GAT, GraphSAGE (>80% Test Accuracy):** By combining both localized document features and graph adjacency structures, GNNs outperform isolated baselines, proving that relational features are far more expressive.

### 2. Isotropic vs. Anisotropic Message Passing
*   **Isotropic (GCN):** Neighbor weights are determined statically by degrees (high-degree hubs get washed out). All neighbors contribute equally to a node's update.
*   **Anisotropic (GAT):** Computes learnable self-attention coefficients dynamically. A node can learn to ignore noisy citation links and focus only on highly relevant neighboring papers.

### 3. Transductive vs. Inductive Scaling
*   **Transductive (GCN/GAT):** Requires computing normalization values over the entire adjacency matrix ($N \times N$) in memory. If a new node is added, the entire graph must be reprocessed.
*   **Inductive (GraphSAGE):** Learns aggregator functions over localized neighborhood samples. This allows generalizing to completely unseen nodes or graphs without retraining.

### 4. Node Classification vs. Graph Classification (MUTAG)
*   For **Cora**, we classify individual nodes inside a single massive citation network.
*   For **MUTAG**, we classify entire graph topologies (molecules). We use GIN (`05.gin.py`) to aggregate node embeddings using **Global Sum Pooling** (Jumping Knowledge) and perform prediction. GIN uses sum aggregation because sums preserve node cardinality, matching the power of the Weisfeiler-Lehman (WL) graph isomorphism test.

### 5. Visualizing GNN Latent Space (t-SNE)
Running GCN, GAT, or GraphSAGE automatically generates a high-resolution 2D t-SNE projection of the hidden representations:
*   **GCN:** `gcn_cora_tsne.png`
*   **GAT:** `gat_cora_tsne.png`
*   **GraphSAGE:** `graphsage_cora_tsne.png`
Open these scatter plots to observe how GNN message-passing convolutions successfully cluster similar academic papers (color-coded by category) into distinct, well-separated topological neighborhoods in the latent space!


