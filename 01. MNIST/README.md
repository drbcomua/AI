# 01. MNIST — a tour of image-classification architectures

A single, consistent playground that runs **34 scripts** — from a k-nearest-neighbour
baseline through LeNet-5, the whole CNN lineage, efficient/mobile nets, attention and
re-parameterization tricks, rotation-equivariant convolutions, and the transformer/MLP
families up to Swin — all on the same MNIST handwritten-digit data.

Every script trains a model, prints a research-grade test report, and (by default) saves a
confusion-matrix PNG. They all share one tiny utility module so each file can focus purely
on the *architecture*.

---

## Quick start

```bash
cd "01. MNIST"

# train any model with sensible defaults (5 epochs)
python3 07.simple-cnn.py

# a fast smoke test on a small subset
python3 10.resnet.py --variant resnet50 --limit 2000 --epochs 2

# the classical (non-deep) baselines
python3 25.classical-baselines.py --variant svm
```

> Use `python3` (not `python`) on macOS. The MNIST data downloads automatically on first
> run into `./data/` (no `torchvision` required) and is cached afterwards.

### Requirements

```bash
pip install torch numpy scikit-learn scipy matplotlib
```

`torch` runs the models; `scikit-learn`/`scipy` power the metrics report and the classical
baselines; `matplotlib` saves the confusion-matrix figures (skip it with `--no-figure`).
A CUDA or Apple-Silicon (MPS) GPU is auto-detected but **not** required — everything runs on
CPU, just slower.

---

## Common command-line flags

Every neural-network script shares the same CLI:

| Flag | Default | Meaning |
|---|---|---|
| `--epochs N` | 5 (10 for ViT, DeiT, Swin, MLP-Mixer) | training epochs |
| `--batch-size N` | 128 | mini-batch size |
| `--lr F` | 1e-3 | Adam learning rate |
| `--limit N` | none | use only the first N train images (fast smoke test) |
| `--device` | auto | `auto` / `cpu` / `cuda` / `mps` |
| `--no-figure` | off | don't save the confusion-matrix PNG |
| `--variant V` | (per file) | pick an architecture variant where applicable |

A few scripts add their own flags:

- **`25.classical-baselines.py`** — `--variant {knn,svm,rf,logreg}`, `--train-size N`
  (no `--epochs`/`--device`; it's scikit-learn, not PyTorch).
- **`30.deit.py`** — `--no-distill` (supervise with labels instead of the CNN teacher),
  `--teacher-epochs N`.

Pass `-h` to any script for its exact options, e.g. `python3 13.efficientnet.py -h`.

---

## What to expect

- **Accuracy.** MNIST is *easy*. A plain CNN hits ~99% in a few epochs, and almost every
  model here lands between **98% and 99.5%**. The point of this folder is to see the *ideas*
  side by side, **not** to chase a leaderboard — the dataset is far too simple to reward the
  giant architectures, which is itself the lesson.
- **The report.** After training, each script prints accuracy with a 95% confidence
  interval, macro/weighted F1, Cohen's kappa, Matthews correlation, log-loss, top-2
  accuracy, a per-class breakdown, and the confusion matrix. The matrix is also written to
  `confusion_<ModelName>.png`.
- **Speed.** Small models (LeNet, Simple CNN, NiN, MobileNet) train in seconds–minutes on
  CPU. The deep/wide ones (ResNet-152, EfficientNet-B7, ResNeXt-101, VGG-19, Wide-ResNet)
  are slow on CPU — use `--limit` for a quick look, or a GPU/MPS for full runs.
- **Honesty about scale.** These are **faithful but MNIST-scaled** adaptations. The native
  ImageNet versions downsample 224×224 by 32× and would annihilate a 28×28 digit, so stems
  and channel widths are shrunk while each architecture's signature block is kept intact.
  Where a paper's exact configuration isn't reproducible at this scale (e.g. NASNet-B/C),
  the script says so in its docstring.

---

## The catalog

Scripts are numbered roughly chronologically / by theme. Families expose their members via
`--variant`.

### Classical & foundational
| # | Script | Architecture | Variants |
|---|---|---|---|
| 25 | `classical-baselines.py` | k-NN · RBF-SVM · Random Forest · logistic regression | `knn` `svm` `rf` `logreg` |
| 06 | `mlp.py` | Multi-layer perceptron — *"do more layers help?"* | `linear` `h1` `h2` `h4` `h8` |
| 29 | `highway.py` | Highway Networks (gated, the precursor to ResNet) | `d10` `d20` `d50` |

### Classic CNNs
| # | Script | Architecture | Variants |
|---|---|---|---|
| 01 | `lenet-5.py` | LeNet-5 (1998) | — |
| 08 | `alexnet.py` | AlexNet (2012) | — |
| 17 | `nin.py` | Network-in-Network (1×1 convs + global average pooling) | — |
| 09 | `vgg.py` | VGG | `vgg16` `vgg19` |
| 07 | `simple-cnn.py` | the modern "hello world" baseline | — |

### Inception / Xception
| # | Script | Architecture | Variants |
|---|---|---|---|
| 02 | `googlenet-inception.py` | GoogLeNet / Inception | — |
| 31 | `inception-v3.py` | Inception-v3 (factorized convs + label smoothing) | — |
| 15 | `xception.py` | Xception (depthwise-separable Inception) | — |

### Residual family
| # | Script | Architecture | Variants |
|---|---|---|---|
| 28 | `resnet-basic.py` | ResNet-18 / 34 (BasicBlock) | `18` `34` |
| 10 | `resnet.py` | ResNet-50 / 101 / 152 (Bottleneck) | `resnet50` `resnet101` `resnet152` |
| 23 | `wide-resnet.py` | Wide ResNet | `16-8` `28-10` `40-4` |
| 22 | `resnext.py` | ResNeXt (grouped-conv cardinality) | `50` `101` |
| 11 | `densenet.py` | DenseNet-BC | `121` `169` `201` |

### Efficient / mobile
| # | Script | Architecture | Variants |
|---|---|---|---|
| 18 | `squeezenet.py` | SqueezeNet (Fire modules) | `1.0` `1.1` |
| 12 | `mobilenet.py` | MobileNet | `v1` `v2` `v3` |
| 19 | `shufflenet.py` | ShuffleNet V2 | `0.5x` `1.0x` `1.5x` `2.0x` |
| 13 | `efficientnet.py` | EfficientNet (compound scaling) | `b0` … `b7` |
| 14 | `nasnet.py` | NASNet (searched cells) | `a` `b` `c` |

### Attention & re-parameterization
| # | Script | Architecture | Variants |
|---|---|---|---|
| 21 | `se-resnet.py` | SE-ResNet (channel attention) | `50` `101` `152` |
| 32 | `cbam.py` | CBAM (channel **+ spatial** attention) | `18` `34` |
| 33 | `repvgg.py` | RepVGG (multi-branch → fused 3×3 at inference) | `a` `b` |

### Modern conv & equivariance
| # | Script | Architecture | Variants |
|---|---|---|---|
| 05 | `convnext.py` | ConvNeXt ("a ConvNet for the 2020s") | — |
| 26 | `convmixer.py` | ConvMixer ("patches are all you need?") | `s` `b` |
| 34 | `gcnn.py` | Group-equivariant CNN (p4 rotations) | — |
| 04 | `spatial-transformer-network.py` | Spatial Transformer Network | — |
| 03 | `capsnet.py` | Capsule Network (dynamic routing) | — |

### Transformers & MLP-only
| # | Script | Architecture | Variants |
|---|---|---|---|
| 16 | `vit.py` | Vision Transformer | — |
| 30 | `deit.py` | DeiT (distillation token + CNN teacher) | `--no-distill` |
| 24 | `swin.py` | Swin Transformer (shifted windows) | `t` `s` |
| 20 | `mlp-mixer.py` | MLP-Mixer (token- vs channel-mixing) | `s` `b` |
| 27 | `poolformer.py` | PoolFormer / MetaFormer (attention → pooling) | `s` `m` |

---

## A few demos worth running deliberately

- **`06.mlp.py` vs `29.highway.py`** — watch a plain deep MLP plateau (`--variant h8`) while
  a gated Highway net of the same depth keeps training (`--variant d50`). The motivation for
  residual learning, in two commands.
- **`33.repvgg.py`** — after training it re-parameterizes the multi-branch blocks into a
  single 3×3 conv stack and prints the train-vs-deploy output difference (≈0) to prove the
  fusion is exact.
- **`34.gcnn.py`** — a rotation-equivariant network; a nice (and slightly imperfect) fit for
  digits, since 6 and 9 are 180° rotations of each other.
- **`25.classical-baselines.py`** — a reminder that MNIST was nearly solved *before* deep
  learning: an RBF-SVM reaches ~98–99%.

---

## How it's wired

- **`mnist_common.py`** — shared utilities: raw IDX download/parse (no `torchvision`),
  DataLoaders with standard normalization, device selection, the training loop, evaluation,
  and the metrics/figure report. Every script imports it as `mc`.
- **`data/`** — cached MNIST IDX files (created on first run).
- **`confusion_*.png`** — confusion-matrix figures saved per model.

To add your own architecture, copy the smallest script (`01.lenet-5.py`), swap in your
`nn.Module`, and keep the `main()` pattern — `mc.train` / `mc.evaluate` / `mc.report` do the
rest.
