# 04. Language Modeling — Text Generation

This directory explores character-level Autoregressive Language Modeling across the major sequence-modeling families: classical n-gram statistics, recurrent networks, causal convolutions (WaveNet), self-attention (Transformers) and its sparse (Mixture-of-Experts) and linear/attention-free (RWKV) variants, and selective state-space scans (Mamba).

All models are trained on **Shakespeare's Plays** to predict the next character given a sliding history window.

---

## Utility Module: `lm_common.py`

Every script imports `lm_common.py` as `mc`. It is responsible for:
*   Downloading and caching the Tiny Shakespeare dataset (falling back to a synthetic string if offline).
*   Character tokenization, vocabulary mapping, and sequence slicing into windows of length $W+1$.
*   Standardizing autoregressive cross-entropy training loops and computing evaluation **Perplexity** (exp of loss).
*   Text generation loops featuring temperature-based sampling from logits.

---

## The Catalog of Scripts

The scripts trace text generation models:

### 01. Char-RNN (`01.char-rnn.py`)
*   **Description:** Recurrent neural network language model.
*   **Architecture:** Character Embedding followed by stacked LSTM layers and a fully connected language modeling head.
*   **Educational Takeaway:** The sequential generation paradigm. Understand how recurrence builds character-by-character contextual memory.

### 02. Nano-GPT (`02.gpt-nano.py`)
*   **Description:** A scaled-down, decoder-only Transformer (Karpathy's nanoGPT style).
*   **Architecture:** Causal self-attention, learnable positional embeddings, LayerNorm, and residual additions.
*   **Educational Takeaway:** Self-attention. Learn how causal masks block future leakage and enable parallel context calculation during training.

### 03. Nano-Mamba (`03.mamba-nano.py`)
*   **Description:** Selective State Space Model sequence generator.
*   **Architecture:** Multi-layer selective scan (Mamba) with local causal 1D convolutions and gated residual paths.
*   **Educational Takeaway:** Selective state retention. Learn how selective parameters dynamically decide what to keep or discard based on sequence tokens.

### 04. N-gram (`04.ngram.py`)
*   **Description:** Classical, count-based language model (no neural network).
*   **Architecture:** k-gram frequency tables for k = 1..n with add-k smoothing and back-off to shorter contexts.
*   **Educational Takeaway:** The pre-neural baseline. Local character statistics alone already produce Shakespeare-flavored text; raising the order lowers perplexity but explodes the table — the curse of dimensionality neural models avoid. (`--order N`)

### 05. WaveNet (`05.wavenet.py`)
*   **Description:** Gated causal dilated 1D CNN (van den Oord 2016 / Gated CNN, Dauphin 2017).
*   **Architecture:** Stacked causal convolutions with exponentially growing dilations, gated (tanh*sigmoid) activations, residual and skip connections.
*   **Educational Takeaway:** The convolutional sequence paradigm — a large receptive field via dilation, fully parallel training, with a fixed (not learned) mixing pattern.

### 06. Mixture-of-Experts GPT (`06.moe-gpt.py`)
*   **Description:** Decoder-only Transformer whose feed-forward layer is a routed mixture of experts (Shazeer 2017; Switch Transformer, Fedus 2021).
*   **Architecture:** Causal self-attention + per-token top-k expert routing (noisy gating); capacity scales with experts while per-token FLOPs stay fixed.
*   **Educational Takeaway:** Conditional computation — the sparse-scaling technique behind modern frontier LLMs.

### 07. RWKV (`07.rwkv.py`)
*   **Description:** Attention-free "linear Transformer" that trains in parallel but runs as an O(1)/step RNN (Peng et al., 2023).
*   **Architecture:** Time-mixing via a numerically stable WKV recurrence (learned per-channel decay) + channel-mixing, with token-shift and no positional embeddings.
*   **Educational Takeaway:** A fixed-size recurrent state replaces the O(T²) attention matrix — the bridge between Char-RNN and nano-GPT.
