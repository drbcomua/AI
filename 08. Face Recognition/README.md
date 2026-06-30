# 08. Face Recognition — Metric Learning & Embeddings

This directory explores Metric Learning, demonstrating how neural networks are trained to map high-dimensional images into low-dimensional vector embeddings where spatial distance represents semantic identity.

Instead of classifying an image into a closed set of classes, these models learn to match and verify images of people they have never seen during training.

All models are evaluated on the **LFW (Labeled Faces in the Wild)** dataset.

---

## The Core Tasks: Verification & Identification

Both operate on embedding distance $d = \|f(x_1) - f(x_2)\|_2$ between unseen test identities:
*   **Verification (1:1):** "Are these two the same person?" Predict "Same" if $d < \tau$. Metrics: ROC curve, AUC, optimal accuracy. *(scripts 01–08, 11)*
*   **Identification (1:N):** "Who, among a gallery, is this probe?" Rank the gallery by distance. Metrics: rank-1 accuracy and the CMC curve. *(script 09)*

---

## Utility Module: `face_common.py`

Every script imports `face_common.py` as `mc`. It is responsible for:
*   Downloading, cropping, and normalizing the LFW face images.
*   Generating pair lists: matching pairs (same person) and non-matching pairs (different people).
*   Batching pairs (or triplets for triplet loss) and mapping them to devices.
*   Training loops: Backpropagating over contrastive, triplet, or angular loss heads.
*   Plotting verification ROC curves and embedding projections (using t-SNE or PCA to visualize face clusters).

---

## The Catalog of Scripts

The scripts trace the development of face representation and metric learning:

### 01. Baselines (`01.baselines.py`)
*   **Description:** Classical, non-deep face representation.
*   **Models:**
    *   *Eigenfaces (PCA):* Projects face pixel vectors into a lower-dimensional principal component subspace, using Cosine or Euclidean distance for comparison.
    *   *Fisherfaces (LDA):* Optimizes projections to maximize between-class variance and minimize within-class variance.
*   **Educational Takeaway:** Understanding how linear projection methods represent faces, and seeing their sensitivity to alignment, lighting, and pose.

### 02. Siamese & Contrastive Loss (`02.siamese-contrastive.py`)
*   **Description:** Twin weight-sharing networks (Chopra et al., 2005).
*   **Method:** Pairs of images $(x_1, x_2)$ are passed through identical CNN backbones. The model is trained using **Contrastive Loss**, which penalizes large distances between matching pairs and small distances (less than a margin $m$) between non-matching pairs:
    $$\mathcal{L} = (1-y) \frac{1}{2} d^2 + y \frac{1}{2} \max(0, m - d)^2$$
*   **Educational Takeaway:** The foundation of deep metric learning: teaching networks to pull similar vectors together and push dissimilar vectors apart.

### 03. Triplet Networks (`03.triplet-net.py`)
*   **Description:** Three-way comparison learning (Schroff et al., FaceNet, 2015).
*   **Method:** Feeds three inputs simultaneously: an Anchor ($A$), a Positive ($P$, same person), and a Negative ($N$, different person). Optimizes **Triplet Loss** to ensure the positive is closer to the anchor than the negative by at least a margin $m$:
    $$\mathcal{L} = \max(0, \|f(A) - f(P)\|^2 - \|f(A) - f(N)\|^2 + m)$$
*   **Educational Takeaway:** Understanding anchor-comparison dynamics and experiencing the necessity of **Hard Negative Mining** (selecting informative triplets to keep gradients active).

### 04. ArcFace (`04.arcface.py`)
*   **Description:** Additive Angular Margin Loss (Deng et al., 2019).
*   **Method:** Instead of optimizing pairs/triplets (which suffer from combinatorial scaling issues during batching), ArcFace trains a classification head where the weight vectors and feature embeddings are $L_2$-normalized. An angular margin $m$ is added directly to the target angle $\theta_y$ in the cosine space:
    $$\mathcal{L} = -\log \frac{e^{s \cdot \cos(\theta_{y} + m)}}{e^{s \cdot \cos(\theta_{y} + m)} + \sum_{j \neq y} e^{s \cdot \cos\theta_j}}$$
*   **Educational Takeaway:** The modern state-of-the-art methodology for face recognition: optimizing hyperspherical classification margins to automatically yield discriminative embeddings.

### 05. SphereFace & CosFace (`05.margin-softmax.py`)
*   **Description:** The angular-margin losses ArcFace descends from. `--variant sphereface|cosface`.
*   **Method:** SphereFace = multiplicative margin `cos(m·theta)`; CosFace = additive cosine margin `cos(theta) - m`. Same normalized-classification setup as ArcFace.
*   **Educational Takeaway:** The progression SphereFace -> CosFace -> ArcFace (multiplicative -> additive-cosine -> additive-angular margins).

### 06. Center Loss (`06.center-loss.py`)
*   **Description:** Softmax + a term pulling embeddings toward learned per-class centers (Wen et al., 2016).
*   **Educational Takeaway:** The bridge from plain classification to the margin losses — it adds intra-class compactness, which the angular-margin losses later enforce directly.

### 07. Supervised Contrastive (`07.supcon.py`)
*   **Description:** In-batch contrastive loss with *multiple* positives per anchor (Khosla et al., 2020). Uses "P identities x K images" batches.
*   **Educational Takeaway:** The modern, mining-free successor to pairwise/triplet learning — richer gradients per step.
*   *Note:* the in-batch Gram loss hits an Apple-MPS BatchNorm-backward bug, so it auto-falls-back to CPU on Macs (fine on CUDA).

### 08. Circle Loss (`08.circle-loss.py`)
*   **Description:** A unified pair/class loss that re-weights each similarity by its distance from the optimum (Sun et al., 2020).
*   **Educational Takeaway:** Generalizes triplet and softmax-margin views; the decision boundary becomes a circle. *(Same MPS->CPU note as SupCon.)*

### 09. Face Identification (`09.identification.py`)
*   **Description:** The 1:N identification task — rank-1 accuracy and the CMC curve over a gallery/probe split.
*   **Method:** Trains a CosFace embedding, enrolls one image per identity, ranks the gallery for each probe (`face_common.evaluate_identification`).
*   **Educational Takeaway:** Verification answers "same?", identification answers "who?" — the other core face-recognition evaluation.

### 10. Local Binary Patterns (`10.lbp.py`)
*   **Description:** A classical, non-deep *texture* descriptor (spatial histograms of 8-bit neighbor codes), verified by chi-square distance.
*   **Educational Takeaway:** A hand-crafted baseline complementing Eigen/Fisherfaces; shows how much learned embeddings (02–09) improved on hand-crafted features.

### 11. Proxy-Anchor Loss (`11.proxy-anchor.py`)
*   **Description:** Proxy-based metric learning — one learnable proxy per class, compared to embeddings (Kim et al., 2020).
*   **Educational Takeaway:** Triplet-like gradients at softmax-like speed/stability, with no pair/triplet mining; converges fast with random batches.
