# 03. Text Classification — Sentiment Analysis

This directory explores NLP classification, tracing the full arc from classical bag-of-words and frequency-based models through dense embeddings, recurrent and convolutional networks, attention and self-attention (Transformers), character-level and hierarchical models, and finally fine-tuning a pretrained Transformer (DistilBERT).

All models are trained on the **IMDB Movie Reviews** dataset to classify review text as either positive (1) or negative (0).

---

## Utility Module: `tc_common.py`

Every script imports `tc_common.py` as `mc`. It is responsible for:
*   Downloading and parsing the IMDB dataset (falling back to a synthetic reviews generator if offline).
*   Space/alphanumeric tokenization and vocabulary building (defaulting to the top 5,000 words).
*   Encoding and padding sequences to a standard length (default 150 words).
*   Standardizing training loops under Binary Cross-Entropy with Logits loss (`BCEWithLogitsLoss`).
*   Printing evaluation classification reports (Precision, Recall, F1, Accuracy) and saving confusion matrix figures.

---

## The Catalog of Scripts

The scripts trace text classification methodologies:

### 01. Classical NLP Baselines (`01.classical-classification.py`)
*   **Description:** Frequency-based text representation benchmarks.
*   **Models:** TF-IDF representation paired with Multinomial Naive Bayes or Logistic Regression.
*   **Educational Takeaway:** Establish a non-deep-learning baseline. Shows how far frequency metrics can go before word order or semantic embeddings are introduced.

### 02. Embedding MLP (`02.embedding-mlp.py`)
*   **Description:** Continuous vector space embeddings mapped to a flat feed-forward network.
*   **Architecture:** Learnable Word Embeddings followed by Global Average Pooling (which averages vectors across the sequence length) and a multi-layer perceptron.
*   **Educational Takeaway:** Introduce dense vector representations and understand the limitations of bag-of-words when word order is discarded.

### 03. Recurrent Classifiers (`03.recurrent-classifier.py`)
*   **Description:** Recurrent architectures mapping sequences to a single sentiment logit.
*   **Models:** Bidirectional LSTM and GRU networks.
*   **Educational Takeaway:** Modeling sequential dependency. Learn how the final recurrent hidden states summarize sequence context.

### 04. Kim CNN (`04.kim-cnn.py`)
*   **Description:** 1D Convolutional Neural Network for sentence classification (Kim, 2014).
*   **Architecture:** Multiple parallel 1D convolution filters of varying sizes (e.g., 3, 4, 5 words) mapping over embedding sequences, followed by max-pooling-over-time.
*   **Educational Takeaway:** Understand how convolutions extract local n-gram features efficiently in parallel.

### 05. Transformer Encoder (`05.transformer-classifier.py`)
*   **Description:** Self-attention encoder (Vaswani et al., 2017) trained from scratch, with a learnable `[CLS]` token and padding-masked attention.
*   **Educational Takeaway:** The dominant modern paradigm — every word attends to every other in one parallel step. From scratch it rivals (not crushes) CNN/RNN; its power needs pretraining (see 10).

### 06. Attention-Pooled BiLSTM (`06.attention-bilstm.py`)
*   **Description:** BiLSTM with additive (Bahdanau-style) attention pooling instead of last-hidden-state.
*   **Educational Takeaway:** The RNN→attention bridge; the learned weights are interpretable (which words drove the sentiment).

### 07. RCNN (`07.rcnn.py`)
*   **Description:** Recurrent Convolutional Network (Lai et al., 2015) — BiLSTM context concatenated with word embeddings, then max-pool-over-time.
*   **Educational Takeaway:** RNN's unbounded context window + CNN's position-invariant pooling are complementary.

### 08. Character-level CNN (`08.char-cnn.py`)
*   **Description:** Convolutions over the raw character stream (Zhang et al., 2015); no word vocabulary at all.
*   **Educational Takeaway:** Language can be modeled from characters up — robust to typos/rare words, at the cost of longer sequences. (Uses the char-level loader in `tc_common`.)

### 09. Hierarchical Attention Network (`09.han.py`)
*   **Description:** Two-level attention (words→sentence, sentences→document) over BiGRU encoders (Yang et al., 2016).
*   **Educational Takeaway:** Structuring the model like the document improves accuracy and gives interpretable word- and sentence-level attention. (Uses the hierarchical loader in `tc_common`.)

### 10. DistilBERT Fine-tuning (`10.bert.py`)
*   **Description:** Fine-tunes a *pretrained* DistilBERT (Sanh et al., 2019) rather than training from scratch.
*   **Educational Takeaway:** Pretraining + fine-tuning is why modern NLP works — high accuracy with little data and few epochs. **Requires `pip install transformers`** and a ~270 MB weight download on first run (the deliberate "batteries-not-included" demo). Use `--limit` on CPU.
