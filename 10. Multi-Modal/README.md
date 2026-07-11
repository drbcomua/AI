# 10. Multi-Modal — Aligning Vision & Language Representation Space

This directory explores Multi-Modal Learning, demonstrating how neural networks are trained to map completely different input modalities (images and text) into a unified, shared embedding space.

Instead of analyzing images or text in isolation, multi-modal systems align these representations so that semantic identity matches across visual and textual data.

All models are evaluated on a synthetic caption-to-shape/image dataset or the Oxford-IIIT Pet multimodal captions.

---

## The Core Task: Contrastive Alignment, Causal Generation, Question Answering & Fused Matching

1.  **Contrastive Alignment (CLIP, SigLIP):** Given a batch of $N$ image-text pairs $(x_i^I, x_i^T)$, maximize the similarity of matching pairs $(x_i^I, x_i^T)$ while minimizing the similarity of all non-matching pairs $(x_i^I, x_j^T)$ for $i \neq j$ -- either via a batch-wide softmax (CLIP) or an independent pairwise sigmoid (SigLIP).
2.  **Causal Generation (Captioning):** Given an image $x^I$, autoregressively predict a textual description sequence $y = (y_1, y_2, \dots, y_T)$ by conditioning a causal decoder on the visual prefix features.
3.  **Closed-Set Question Answering (VQA):** Given an image $x^I$ and a question $x^Q$, classify a fused joint representation into one of a fixed set of answers.
4.  **Fused Image-Text Matching (VisualBERT, ViLBERT, ViLT):** Given an image-text pair, fuse both modalities *inside* shared or cross-attention layers (rather than after independent encoding) and predict whether they match -- trading the efficiency of dual encoders for deeper, earlier cross-modal interaction.
5.  **Unified Multi-Task Pretraining (BLIP, CoCa):** Combine several of the above objectives (contrastive, matching, and/or generative) in a single model, either via separate forward-pass "modes" over a shared backbone (BLIP) or a single forward pass split into unimodal and multimodal stages (CoCa).
6.  **Frozen-Backbone Bridging (BLIP-2):** Freeze large vision and language backbones entirely and train only a small "connector" module (a Querying Transformer) that translates between them -- trading end-to-end fine-tuning for drastic parameter efficiency.

---

## Utility Module: `multimodal_common.py`

Every script imports `multimodal_common.py` as `mc`. It is responsible for:
*   Generating or downloading a synthetic captioning/matching dataset (e.g., shapes of different colors with captions like "a blue circle", "a red square").
*   Tokenizing captions using character/subword vocabularies and preparing image transformations.
*   Generating synthetic VQA question/answer pairs (`generate_vqa_pairs`) from the same captions, e.g. "what color is this" -> "red".
*   Building positive/hard-negative Image-Text Matching pairs (`build_itm_pairs`) and evaluating fused encoders via joint-forward-pass retrieval (`evaluate_itm_retrieval`), since fused models cannot precompute independent embeddings.
*   Standardizing the training loops for contrastive losses (CLIP, SigLIP) and language modeling cross-entropy (Captioner).
*   Plotting contrastive/matching similarity matrices, VQA result grids, and generating test image caption grids.

---

## The Catalog of Scripts

The scripts demonstrate the core architectures of vision-language pairing and generation:

### 01. Contrastive Language-Image Pretraining (`01.clip.py`)
*   **Description:** Aligning separate image and text backbones (Radford et al., OpenAI, 2021).
*   **Architecture:**
    *   *Image Encoder:* Simple Conv2D CNN or Vision Transformer (ViT).
    *   *Text Encoder:* Simple causal MLP or tiny causal GRU/Transformer.
    *   *Alignment Head:* Normalizes both embeddings to the unit hypersphere and computes a symmetric cross-entropy loss over a similarity matrix.
*   **Educational Takeaway:** Understanding how contrastive loss binds vision and language representations, and observing zero-shot classification capabilities.

### 02. Image Captioning Transformer (`02.image-captioner.py`)
*   **Description:** Generating text conditioned on visual inputs.
*   **Architecture:**
    *   *Vision Stem:* Conv2D CNN feature extractor.
    *   *Decoder:* Causal Transformer Decoder utilizing multi-head cross-attention over the visual feature grid to predict character tokens sequentially.
*   **Educational Takeaway:** Combining convolutional encoders with causal sequence decoders via cross-attention, laying the foundation for modern Vision-Language Models (VLMs).

### 03. Visual Question Answering Baseline (`03.vqa-baseline.py`)
*   **Description:** The original VQA fusion baseline (Antol et al., 2015), framing vision-language reasoning as closed-set classification rather than retrieval or generation.
*   **Architecture:**
    *   *Image Encoder:* Conv2D CNN producing a pooled visual embedding.
    *   *Question Encoder:* GRU producing a pooled question embedding.
    *   *Fusion:* Element-wise (Hadamard) product of the two embeddings, followed by an MLP classifier over a fixed answer vocabulary.
*   **Educational Takeaway:** The simplest possible multi-modal fusion mechanism, and the third major multi-modal task family (classification) alongside retrieval (CLIP) and generation (Captioner).

### 04. Sigmoid Loss for Language-Image Pretraining (`04.siglip.py`)
*   **Description:** A near-identical dual-encoder architecture to CLIP, replacing the batch-wide softmax InfoNCE loss with an independent pairwise sigmoid loss (Zhai et al., Google DeepMind, 2023).
*   **Architecture:** Identical CNN/GRU dual towers to CLIP, but every cell of the similarity matrix is trained as an independent binary match/non-match decision via `-log(sigmoid(label * logit))`, with a learnable temperature *and* bias term.
*   **Educational Takeaway:** Removing the global softmax normalizer makes the loss trivially parallelizable and stable even at very small batch sizes -- contrast `04.siglip.py --batch-size 8` against `01.clip.py --batch-size 8`.

### 05. VisualBERT: Single-Stream Fusion Transformer (`05.visualbert.py`)
*   **Description:** A single shared Transformer encoder fuses image and text from the very first layer (Li et al., 2019).
*   **Architecture:** `[CLS] + 16 visual grid tokens + text tokens` are concatenated with segment/position embeddings and passed through one shared self-attention stack, jointly trained with Image-Text Matching (ITM) and Masked Language Modeling (MLM) losses.
*   **Educational Takeaway:** "Single-stream" fusion happens as early as possible inside ordinary self-attention, at the cost of losing CLIP's ability to precompute independent embeddings -- retrieval requires a fresh joint forward pass per candidate caption (`mc.evaluate_itm_retrieval`).

### 06. ViLBERT: Dual-Stream Co-Attention Transformer (`06.vilbert.py`)
*   **Description:** Two parallel Transformer streams exchange information through dedicated cross-attention (co-attention) layers instead of one fully fused stack (Lu et al., 2019).
*   **Architecture:** Independent visual and text self-attention streams, bridged by `CoAttentionLayer`s where each stream's queries attend to the other stream's keys/values; pooled CLS embeddings are matched via a projected cosine similarity.
*   **Educational Takeaway:** A middle ground between CLIP's "fusion only at the very end" and VisualBERT's "fusion from layer one" -- each modality keeps specialized self-attention weights while still achieving deep bidirectional fusion through explicit cross-attention.

### 07. ViLT: Convolution-Free Fusion Transformer (`07.vilt.py`)
*   **Description:** The single-stream fusion recipe of VisualBERT, but with the CNN visual backbone removed entirely (Kim et al., 2021).
*   **Architecture:** Raw image patches are flattened and linearly projected (ViT-style, no convolution) into visual tokens, then fused with text via the same `[CLS] + visual + text` single self-attention stack as VisualBERT, trained with ITM + MLM.
*   **Educational Takeaway:** Removing the CNN's spatial inductive bias makes the model noticeably harder and noisier to train on fine-grained (shape) distinctions than VisualBERT at this tiny data scale -- a small-scale echo of why the original paper needed large-scale pretraining data.

### 08. BLIP: Bootstrapping Language-Image Pretraining (`08.blip.py`)
*   **Description:** One shared text Transformer, reused in three modes against one image encoder, unifying retrieval, matching, and captioning into a single model (Li et al., 2022).
*   **Architecture:** A `BLIPTextTransformer` toggles between a unimodal encoder (ITC, no image), an image-grounded encoder (ITM, bidirectional + cross-attention), and an image-grounded decoder (LM, causal + cross-attention) -- same weights, different attention pattern and head per mode.
*   **Educational Takeaway:** A natural synthesis of `01.clip.py` (retrieval), `02.image-captioner.py` (generation), and `05.visualbert.py` (matching) into one architecture, showing how little structural change separates "understanding" from "generation" in a Transformer.

### 09. BLIP-2: Frozen Querying Transformer (`09.blip2-qformer.py`)
*   **Description:** Freeze large vision and language backbones entirely; train only a small Q-Former that bridges them via a handful of learnable query tokens (Li et al., 2023).
*   **Architecture:** Learnable queries cross-attend into a frozen (randomly-initialized, `requires_grad=False`) CNN's visual tokens, and the resulting compact summary is projected into a frozen causal decoder as a soft-prompt prefix.
*   **Educational Takeaway:** Demonstrates the parameter-efficiency pattern behind most modern VLM "connector" modules -- print the trainable-vs-frozen parameter split and compare against `02.image-captioner.py`'s fully end-to-end training. Since this repo has no real pretrained weights, "frozen" means randomly initialized here, which makes convergence genuinely slower/noisier than with a real pretrained LM -- an honest limitation, not a bug.

### 10. CoCa: Contrastive Captioner (`10.coca.py`)
*   **Description:** Gets contrastive alignment *and* captioning from a single forward pass by splitting the text tower into unimodal (contrastive) and multimodal (captioning) stages (Yu et al., Google Research, 2022).
*   **Architecture:** Causal self-attention-only "unimodal" layers pool a text embedding before ever seeing the image; causal self-attention + cross-attention "multimodal" layers continue from there to produce captioning logits.
*   **Educational Takeaway:** Simpler and more efficient than BLIP (no separate ITM forward pass, no ITM loss), but the shared trunk needs enough of its own capacity to serve both objectives -- one unimodal layer lets the fast-converging captioning loss dominate and retrieval plateaus around ~46%, while two layers reach 100%.

---

## Expected Performance & Comparisons

*   **CLIP Contrastive Alignment (~80-90% Zero-shot Accuracy):** In early epochs, the similarity matrix is chaotic. As the symmetric InfoNCE loss optimizes, diagonal alignment dominates, allowing the model to accurately pair unseen text descriptions to target shapes.
*   **Image Captioner (Perplexity & BLEU score):** Conditioning the causal decoder on structural image grids results in highly coherent description sequences compared to unconditioned generation.
*   **VQA Baseline (~100% Answer Accuracy):** The Hadamard fusion baseline saturates quickly on this synthetic closed-set task, since color/shape are each linearly separable from the pooled CNN/GRU embeddings.
*   **SigLIP (~80-100% Zero-shot Accuracy):** Converges to similar accuracy as CLIP but remains stable even at very small batch sizes, since its loss does not depend on a large pool of in-batch negatives to normalize against.
*   **VisualBERT & ViLBERT (~100% Zero-shot Matching Accuracy, given enough epochs):** Both fused encoders need noticeably more epochs than CLIP/SigLIP to separate fine-grained (same-color, different-shape) hard negatives, since gradient signal for those cases is sparser under an Image-Text-Matching objective than under CLIP's full in-batch contrastive loss.
*   **ViLT (~30-90% Zero-shot Matching Accuracy, highly variable):** Without a CNN's spatial inductive bias, fine-grained shape discrimination is a much harder optimization problem for pure patch self-attention at this data scale; MLM cloze-filling, by contrast, converges reliably and near-perfectly in every run.
*   **BLIP (~100% Zero-shot ITC Retrieval, near-perfect captions):** Reusing the same CNN backbone and training recipe ingredients proven by CLIP (contrastive), VisualBERT (matching), and the image captioner (generation) lets all three of BLIP's objectives converge cleanly together.
*   **BLIP-2 Q-Former (~15-50% Exact-Match Caption Accuracy, still improving with more epochs):** With only randomly-initialized (not genuinely pretrained) frozen backbones available in this repo, the Q-Former must learn to steer a fixed random decoder from scratch through a tight 4-token bottleneck -- slower and noisier than any fully-trained script here, but the *trainable parameter fraction* it achieves (a small fraction of the full model) is the real point.
*   **CoCa (~100% Zero-shot Retrieval, near-perfect captions, given enough unimodal capacity):** A single forward pass yields both a fast CLIP-style retrieval embedding and full captioning logits, but the shared trunk needs at least two unimodal layers before the pooling point or the fast-converging captioning loss starves the contrastive objective of useful gradient.
