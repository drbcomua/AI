# 12. Image Segmentation — Pixel-Level Dense Prediction

This directory explores Image Segmentation, demonstrating how convolutional neural networks generalize from single-label image classification to dense, pixel-level predictions.

Instead of outputting a single class probability for the entire image, segmentation models output a class label for every individual pixel, outlining precise spatial boundaries.

All models are evaluated on a synthetic shape segmentation dataset or the **Oxford-IIIT Pet segmentation dataset** (segmenting cats/dogs from background).

---

## The Core Task: Semantic Segmentation

Given an image $x \in \mathbb{R}^{3 \times H \times W}$, predict a classification grid $\hat{y} \in \mathbb{R}^{C \times H \times W}$ where $C$ is the number of target classes.
*   **Loss Functions:** Cross-Entropy loss combined with **Soft Dice Loss** to handle class imbalance (background vs. foreground boundary):
    $$\mathcal{L}_{Dice} = 1 - \frac{2 \sum p_i g_i}{\sum p_i^2 + \sum g_i^2}$$
*   **Evaluation Metrics:** Mean Intersection over Union (mIoU) and pixel accuracy.

---

## Utility Module: `segmentation_common.py`

Every script imports `segmentation_common.py` as `mc`. It is responsible for:
*   Generating or downloading image-mask pairs.
*   Applying pixel-aligned data augmentations (e.g., matching random crops, flips, and rotations on both image and target mask simultaneously).
*   Standardizing training loops and monitoring mIoU and Dice scores.
*   Plotting test prediction grids showing the Input Image, Ground Truth Mask, and Predicted Segmentation Mask side-by-side.

---

## The Catalog of Scripts

The scripts trace the development of convolutional segmentation models:

### 01. Fully Convolutional Network (`01.fcn.py`)
*   **Description:** Classic classification network modified for dense prediction (Long et al., 2015).
*   **Architecture:** Replaces fully connected classifier layers in a standard CNN with $1 \times 1$ convolutions, using transposed convolutions to upsample intermediate feature maps back to the original image dimensions.
*   **Educational Takeaway:** Transitioning from global image classification to dense spatial prediction.

### 02. U-Net (`02.unet.py`)
*   **Description:** Symmetric encoder-decoder architecture with skip-connections (Ronneberger et al., 2015).
*   **Architecture:**
    *   *Contracting path (Encoder):* Standard CNN extraction layers downsampling spatial size while expanding feature channels.
    *   *Expanding path (Decoder):* Transposed convolutions upsampling spatial size.
    *   *Skip Connections:* Concatenates high-resolution encoder features directly with decoding features to preserve fine structural borders.
*   **Educational Takeaway:** The power of skip connections to preserve high-resolution spatial information, forming the basis for medical imaging and modern diffusion model stems.

### 03. Pyramid Scene Parsing Network (`03.pspnet.py`)
*   **Description:** Multi-scale context aggregation network using a Pyramid Pooling Module (PPM) (Zhao et al., 2017).
*   **Architecture:** Downsamples inputs to a feature map, applies PPM to pool features at different scales (e.g., $1\times1$, $2\times2$, $3\times3$, $6\times6$), concatenates the upsampled pooled representations, and projects back to the class score grid.
*   **Educational Takeaway:** Enhancing the receptive field via parallel sub-region average pooling at different grid resolutions.

### 04. DeepLabV3+ (`04.deeplabv3plus.py`)
*   **Description:** State-of-the-art encoder-decoder model combining dilated convolutions (ASPP) and early feature fusion (Chen et al., 2018).
*   **Architecture:**
    *   *Encoder (ASPP):* Employs parallel atrous convolutions with varying dilation rates (`[2, 4, 6]`) and global pooling to capture multi-scale context.
    *   *Decoder:* Fuses the upsampled encoder output with a $1 \times 1$-conv projected low-level feature map from an earlier block, improving boundary recovery.
*   **Educational Takeaway:** The synergy of Atrous Spatial Pyramid Pooling (multi-scale context) and refined boundary skip-connections.

### 05. SegNet (`05.segnet.py`)
*   **Description:** Encoder-decoder model highlighting memory-efficient pooling index upsampling (Badrinarayanan et al., 2015).
*   **Architecture:**
    *   *Encoder:* Stores max-pooling indices during downsampling.
    *   *Decoder:* Performs spatial unpooling using those saved indices to position activations back in their original coordinates, followed by convolutions.
*   **Educational Takeaway:** Memory-efficient spatial upsampling that avoids both learning transpose parameters (FCN) and storing high-resolution skip feature maps (U-Net).

### 06. SegFormer (`06.segformer.py`)
*   **Description:** High-efficiency transformer-based semantic segmentation model (Xie et al., 2021).
*   **Architecture:**
    *   *Encoder (MiT):* Hierarchical transformer blocks generating multi-resolution feature maps without absolute position embeddings (implicit via depth-wise FFN convs), and applying efficient key-value reduction.
    *   *Decoder:* All-MLP structure that projects and bilinear-upsamples stage features to a common resolution, concatenates them, and outputs predictions.
*   **Educational Takeaway:** Introducing the transformer attention paradigm to dense prediction using hierarchical representations and lightweight MLP heads.

---

## Expected Performance & Comparisons

*   **FCN (mIoU ~0.55-0.60):** Struggles to predict sharp edge boundaries because upsampling high-level latent features directly loses fine spatial details.
*   **U-Net (mIoU ~0.75-0.85):** Skip connections successfully restore spatial precision, producing highly refined outlines of pets and shapes.
*   **PSPNet (mIoU ~0.94-0.96):** PPM collects global and regional context effectively, leading to highly robust shape segmentation.
*   **DeepLabV3+ (mIoU ~0.98-0.99):** The fusion of multi-scale atrous contexts and sharp low-level feature boundaries achieves almost perfect segmentation alignment.
*   **SegNet (mIoU ~0.98-0.99):** Sparing use of pooling indices helps maintain extreme spatial alignment, generating highly precise shapes.
*   **SegFormer (mIoU ~0.98-0.99):** Multi-scale self-attention captures rich context without hand-crafted convolutions, offering supreme shape reconstruction.
