# 09. 3D Classification — Convolution & Attention on Volumetric Grids

This directory explores 3D deep learning across its major **representations** — dense voxel grids (Conv3D / ResNet3D / VoxNet / ViT3D / sparse conv / DenseNet3D), unordered **point clouds** (PointNet, PointNet++, DGCNN, Point Transformer), and **multi-view** 2D projections (MVCNN) — demonstrating how convolution, graphs, and self-attention each adapt to 3D data.

To avoid the file size and preprocessing overhead of massive raw medical scans (MRI/CT), all networks are evaluated on **3D MNIST**. This is a synthetic dataset representing handwritten digits as $28 \times 28 \times 28$ binary voxel grids, focusing learning purely on volumetric spatial relationships.

---

## The Core Task: Volumetric Classification

Given a 3D tensor of shape $1 \times D \times H \times W$ (representing a single-channel voxel volume of depth $D$, height $H$, and width $W$), predict the class probability distribution across 10 digit classes.

---

## Utility Module: `voxel_common.py`

Every script imports `voxel_common.py` as `mc`. It is responsible for:
*   Downloading and caching the 3D MNIST dataset (typically represented as HDF5 or numpy array files).
*   Performing spatial augmentations in 3D (e.g., random 3D rotations, translations, and scaling).
*   Batching voxel grids and device mapping.
*   Standardizing training loops and monitoring metrics (accuracy, confidence intervals).
*   Saving classification confusion matrices and outputting voxel visualization graphics (using 3D plotting libraries like `matplotlib` or raw voxel grid export).

---

## The Catalog of Scripts

The scripts demonstrate volumetric architectures and projection methods:

### 01. Volumetric CNN Baseline (`01.conv3d-baseline.py`)
*   **Description:** Basic 3D Convolutional Neural Network.
*   **Architecture:** Uses standard volumetric convolutional layers (`nn.Conv3d`) and 3D max pooling (`nn.MaxPool3d`) to extract features across all three spatial dimensions.
*   **Educational Takeaway:** Setting the baseline for volumetric classification and observing the computation/parameter scaling when adding a third spatial dimension.

### 02. 3D ResNet (`02.resnet3d.py`)
*   **Description:** Residual learning scaled to 3D (Hara et al., 2017).
*   **Architecture:** Adapts the ResNet bottleneck and residual skip connection blocks to use 3D convolutions.
*   **Educational Takeaway:** Implementing deeper volumetric networks using skip connections to overcome vanishing gradients during 3D backpropagation.

### 03. VoxNet (`03.voxnet.py`)
*   **Description:** Real-time 3D convolutional network for point clouds and voxels (Maturana & Scherer, 2015).
*   **Architecture:** Specifically configured volumetric layers optimized for sparse binary voxel grids (e.g., handling LiDAR scans or occupancy grids), utilizing small kernel sizes and high strides.
*   **Educational Takeaway:** Designing efficient volumetric nets to process sparse data where most voxels are empty.

### 04. 3D Vision Transformer (`04.vit3d.py`)
*   **Description:** Attention over volumetric patches (Tubelets).
*   **Architecture:** Segments the $28 \times 28 \times 28$ volume into 3D voxel patches (e.g., $4 \times 4 \times 4$ tubelets). Flat patches are projected into linear embeddings, combined with 3D positional encodings, and processed by standard Transformer encoder blocks.
*   **Educational Takeaway:** Extending patch-based attention to 3D, and understanding how self-attention weights capture cross-volume spatial relationships.

### 05. PointNet (`05.pointnet.py`)
*   **Description:** Deep learning on raw point clouds (Qi et al., 2017) — a different 3D *representation* (occupied voxels sampled as a point set).
*   **Architecture:** Per-point shared MLP -> permutation-invariant global max-pool -> classifier, plus a T-Net input alignment.
*   **Educational Takeaway:** The symmetric max-pool gives order-invariance and selects the shape's "critical points."

### 06. PointNet++ (`06.pointnet2.py`)
*   **Description:** Hierarchical point-set learning (Qi et al., 2017).
*   **Architecture:** Stacked Set Abstraction layers (sample centroids -> group k-NN -> local PointNet), growing the receptive field like a CNN.
*   **Educational Takeaway:** Captures local geometry and how it composes — what flat PointNet misses.

### 07. DGCNN / EdgeConv (`07.dgcnn.py`)
*   **Description:** Dynamic Graph CNN on point clouds (Wang et al., 2019).
*   **Architecture:** EdgeConv on a k-NN graph rebuilt in *feature* space each layer (edge feature [x_i, x_j - x_i] -> MLP -> max).
*   **Educational Takeaway:** Point clouds as graphs; ties 3D learning to the message-passing GNNs of folder 07.

### 08. MVCNN (`08.mvcnn.py`)
*   **Description:** Multi-View CNN (Su et al., 2015) — the projection-to-2D *representation*.
*   **Architecture:** Render the volume to N silhouette views, shared 2D CNN per view, view-pool (max), classify.
*   **Educational Takeaway:** Reusing mature 2D backbones on rendered views often beats native voxel CNNs at lower cost.

### 09. Point Transformer (`09.point-transformer.py`)
*   **Description:** Self-attention for point clouds (Zhao et al., 2021).
*   **Architecture:** Vector attention within each point's k-NN neighbourhood, modulated by a learned relative-position encoding.
*   **Educational Takeaway:** Per-channel (vector) attention with positional encoding — more expressive than scalar dot-product attention.

### 10. Submanifold Sparse Conv3D (`10.sparse-conv3d.py`)
*   **Description:** Sparse convolution that keeps the active-site set fixed (Graham & van der Maaten, 2017).
*   **Architecture:** Conv3d masked by the (fixed) occupancy grid, so features never dilate into empty space (an educational dense-tensor emulation).
*   **Educational Takeaway:** Most voxels are empty; sparse conv preserves sparsity and (in the real version) skips empty-voxel compute.

### 11. DenseNet3D (`11.densenet3d.py`)
*   **Description:** Densely-connected volumetric CNN (Huang et al., 2017, adapted to 3D).
*   **Architecture:** Dense blocks (every layer concatenates all previous feature maps) + transition (compress + pool).
*   **Educational Takeaway:** Feature reuse and gradient flow make deep 3D nets more parameter-efficient.

---

## Expected Performance & Comparisons

Running these models on the synthetic **3D MNIST** voxel grids highlights key differences in volumetric processing:

### 1. Conv3D Baseline vs. VoxNet
*   **VoxNet (~85-90% Test Accuracy):** Rapidly downsamples the sparse binary occupancy grid using strided convolutions without padding. This design makes it highly parameter-efficient and fast to train, achieving high accuracy in just 2 epochs.
*   **Conv3D Baseline (~88-93% Test Accuracy):** Outperforms VoxNet slightly by utilizing standard dense convolutional padding, which helps preserve structural border information in the voxel grids, but consumes more parameters and memory.

### 2. 3D ResNet
*   **ResNet3D (~75-80% Test Accuracy in 2 epochs):** Leverages residual skip connections and Global Average Pooling (GAP). While GAP keeps the classification head extremely small (saving parameters), it takes slightly more epochs to match the localized spatial features of standard fully-connected flatten layers.

### 3. Volumetric CNNs vs. 3D Vision Transformers
*   **Inductive Bias:** Volumetric CNNs (Conv3D, ResNet, VoxNet) have a strong built-in spatial inductive bias (pixels/voxels next to each other are related). This allows them to converge extremely quickly, reaching ~90% accuracy in 2 epochs.
*   **No Inductive Bias (ViT3D):** Transformers have no built-in spatial priors. They treat voxel patches as a sequence of tokens. Consequently, ViT3D takes longer to align 3D positional encodings and map patch projections, starting around ~30% accuracy in early epochs and requiring more steps/epochs to reach parity.

---

## 3D Voxel Grid Visualizations

Running the Volumetric CNN Baseline (`01.conv3d-baseline.py`) automatically renders the 3D voxel grids:
*   **Occupancy Rendering:** Saves to `conv3d_digit_sample.png`. This plots actual 3D cubes for active voxels, showing a beautiful, volumetric representation of the handwritten digit in 3D coordinate space!
*   **Confusion Matrices:** Every script generates a confusion matrix (e.g., `voxnet_confusion_matrix.png`) showing how digits are classified.
