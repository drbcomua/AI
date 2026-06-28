# 02. Time Series & Forecasting — A Tour of Sequence Predictors

This directory contains comparative implementations of time-series forecasting models, comparing classical baselines, recurrent neural networks, temporal convolutional networks, and transformer architectures.

All models are evaluated on the following datasets to highlight the role of signal-to-noise ratio (and, now, *scale*) in model performance:
1.  **`jena`** — Jena Climate, hourly, 3 features (temperature, pressure, density). Physical, highly predictable with clear seasonal/diurnal patterns. (~70k rows)
2.  **`spy`** — Synthetic S&P 500 ETF close price (Geometric Brownian Motion). Chaotic, noisy, close to a mathematical random walk. (5k rows, offline)
3.  **`jena_full`** — **the "show scale" weather dataset.** The *same* Jena CSV at full 10-minute resolution with **all 14 variables** (~420k rows, multivariate). No extra download — it just stops throwing the raw data away. Use this to see whether high-capacity models (iTransformer, TimesNet, Autoformer, Mamba) can convert more data + more variables into lower error, where small models plateau.
4.  **`spy_full`** — **real, multivariate market data.** Full daily **OHLCV** history of the SPY ETF since 1993 (~8.4k rows), downloaded free from the Yahoo Finance chart API and cached to `data/spy.csv` (falls back to a synthetic OHLCV series if offline). Unlike synthetic `spy`, this exhibits genuine **non-stationarity / distribution shift** (price grew ~16x), which makes it a sharp test of normalization tricks (try `nlinear`) and a reminder of how hard real markets are.

> **On dataset size & scale.** The small datasets are great for fast comparison but too easy/small to reward big models. `jena_full` (more data + variables) and `spy_full` (real, non-stationary) are the levers: `python3 11.itransformer.py --dataset jena_full --epochs 5`. For quick iteration use `--limit N` to cap the row count. (Want even more? The natural next additions are the standard long-horizon benchmarks — ETT, Electricity, Traffic, Weather — which `ts_common.py` could grow downloaders for.)

---

## The Core Task

Predict the next step (or sequence of steps) $y_{t+1:t+H}$ given a historical window $y_{t-W:t}$.
*   **Window Size ($W$):** Default is 24 timesteps.
*   **Forecast Horizon ($H$):** Default is 1 timestep (single-step forecasting).
*   **Features:** Univariate (predict close price from close price for SPY) or Multivariate (predict temperature from temperature + pressure + humidity for Jena).

---

## Utility Module: `ts_common.py`

Every script imports `ts_common.py` as `mc`. It is responsible for:
*   Downloading and caching the Jena Climate CSV and fetching SPY stock pricing via a public financial API or synthetic fallback.
*   Processing, scaling (MinMax or Standard scaling), and splitting data into training, validation, and test datasets.
*   Generating windowed datasets using PyTorch `TensorDataset`.
*   Standardizing training loops and tracking metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE).
*   Plotting prediction overlays against ground truth.

### Plot controls

The forecast plot shows a slice of the test set (the metrics always cover the *full* test set). Two shared flags control which slice, and the plotted date range is written into the chart title (e.g. `recent 150: 2025-11-19 .. 2026-06-26`):

*   `--plot-window {recent,first}` — `recent` (default) shows the most recent points; `first` shows the start of the test period.
*   `--plot-points N` — number of points to plot (default 150).

```bash
python3 11.itransformer.py --dataset spy_full --plot-window recent --plot-points 200
```

---

## The Catalog of Scripts

The following scripts should be implemented sequentially:

### 01. Baselines (`01.baselines.py`)
*   **Description:** Non-deep learning statistical baselines.
*   **Models:**
    *   *Naive / Persistence:* Predict $y_{t+1} = y_t$. (Critical benchmark for financial data).
    *   *Moving Average:* Simple average of the last $N$ window steps.
    *   *ARIMA:* Classical autoregressive integrated moving average model (implemented using `statsmodels`).
*   **Educational Takeaway:** Setting the benchmark. Showing how hard it is for deep neural networks to beat the naive baseline on stock prices.

### 02. Linear & MLP (`02.linear-mlp.py`)
*   **Description:** Autoregressive linear networks and feedforward MLPs.
*   **Models:**
    *   *Linear AR:* A single linear layer mapping the history window $W \times F \to 1$.
    *   *Multi-Layer Perceptron (MLP):* Stacked fully-connected layers mapping flat history window to forecast.
*   **Educational Takeaway:** Testing if non-sequential models with simple mappings are sufficient for cyclical data.

### 03. Recurrent Networks (`03.rnn-lstm-gru.py`)
*   **Description:** Sequence-aware recurrent neural networks.
*   **Variants:**
    *   `rnn` - Standard vanilla Recurrent Neural Network (demonstrating vanishing gradients).
    *   `lstm` - Long Short-Term Memory network (gated state preservation).
    *   `gru` - Gated Recurrent Unit (simplified gating mechanism).
*   **Educational Takeaway:** How gating mechanisms prevent gradients from vanishing or exploding over long sequence windows.

### 04. Temporal Convolutional Networks (`04.tcn.py`)
*   **Description:** 1D Dilated Causal Convolutional networks (Bai et al., 2018).
*   **Architecture:** Stacks of 1D convolutions with dilated receptive fields and causal masking (to prevent looking into the future).
*   **Educational Takeaway:** Demonstrating that parallel 1D convolutions can outperform recurrent models in speed and long-term memory capacity.

### 05. Time-Series Transformer (`05.transformer.py`)
*   **Description:** Attention-based sequence forecasting.
*   **Variants:**
    *   `vanilla` - Standard encoder-decoder transformer applied to sequence steps.
    *   `patch` - PatchTST style (segmenting the time series into sub-series patches to reduce attention complexity).
*   **Educational Takeaway:** The pros and cons of using self-attention on continuous numerical data, and why direct tokenization can fail without patch representations.

### 06. DLinear / NLinear (`06.dlinear.py`)
*   **Description:** One-layer linear forecasters (Zeng et al., 2023).
*   **Variants:** `dlinear` (trend/seasonal decomposition + linear), `nlinear` (last-value normalization + linear).
*   **Educational Takeaway:** A single linear layer rivals transformers on many benchmarks — a sanity check against over-engineering.

### 07. N-BEATS / N-HiTS (`07.nbeats-nhits.py`)
*   **Description:** Pure-MLP doubly-residual stacks (Oreshkin 2020 / Challu 2023).
*   **Variants:** `nbeats` (generic blocks), `nhits` (multi-rate pooling for hierarchical frequencies).
*   **Educational Takeaway:** Backcast/forecast residual stacking peels the signal apart with no recurrence or attention.

### 08. TSMixer (`08.tsmixer.py`)
*   **Description:** All-MLP model alternating time-mixing and feature-mixing (Chen et al., 2023).
*   **Educational Takeaway:** "Mixing," not specifically attention, is the operation that matters — the MLP-Mixer lesson for time series.

### 09. DeepAR (`09.deepar.py`)
*   **Description:** Probabilistic autoregressive LSTM trained by maximum likelihood (Salinas et al., 2020).
*   **Educational Takeaway:** Predicts a distribution (mean + variance), not a point — uncertainty widens on noisy SPY and narrows on smooth Jena. Trained with Gaussian NLL.

### 10. Autoformer (`10.autoformer.py`)
*   **Description:** Decomposition transformer with FFT-based Auto-Correlation attention (Wu et al., 2021).
*   **Educational Takeaway:** Attention over *lags/periods* rather than positions; series decomposition is built into every block. Excellent fit for strongly periodic data.

### 11. iTransformer (`11.itransformer.py`)
*   **Description:** Inverted transformer — variables (not timesteps) are the tokens (Liu et al., 2024).
*   **Educational Takeaway:** Attention models correlations *between variables*; shines on multivariate data — run it on `--dataset jena_full` (14 variables).

### 12. TimesNet (`12.timesnet.py`)
*   **Description:** Folds the 1D series into 2D by FFT-detected periods, then applies 2D inception convs (Wu et al., 2023).
*   **Educational Takeaway:** Recasting temporal modeling as a vision problem captures multiple overlapping periodicities at once.

### 13. Mamba (`13.mamba.py`)
*   **Description:** Selective state-space model — the SSM alternative to transformers (Gu & Dao, 2023).
*   **Educational Takeaway:** Input-dependent state-space parameters give attention-like selectivity at O(W) cost; implemented here with an explicit (clear, not fast) selective scan.

---

## Recommended run for the "scale" story

```bash
# small + easy: most models cluster together, linear baselines are competitive
python3 06.dlinear.py   --dataset jena --epochs 5
# large + multivariate: capacity-hungry models pull ahead
python3 11.itransformer.py --dataset jena_full --epochs 5
python3 12.timesnet.py     --dataset jena_full --epochs 5
python3 13.mamba.py        --dataset jena_full --epochs 5
```
