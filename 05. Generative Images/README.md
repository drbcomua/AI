# 05. Generative Images — Synthesizing Latent Distributions

This directory demonstrates the major mathematical paradigms of generative modeling: reconstruction bottlenecking, explicit density modeling, adversarial game theory, autoregressive factorization, exact-likelihood normalizing flows, discrete-latent quantization, and iterative noise-denoising (diffusion, with fast/guided sampling).

All models train on the **FashionMNIST** dataset (or the standard **MNIST** dataset) to synthesize completely novel, synthetic clothing items or digit configurations from a latent Gaussian distribution.

---

## Utility Module: `gen_common.py`

Every script imports `gen_common.py` as `mc`. It is responsible for:
*   Loading and normalizing FashionMNIST/MNIST datasets.
*   Generating random noise vectors $z \sim \mathcal{N}(0, I)$ for evaluation sampling.
*   Plotting and saving grid visualizers (e.g., $8 \times 8$ grids of generated samples).
*   Latent space interpolation utilities (e.g., performing a spherical linear interpolation (slerp) between two latent points to demonstrate latent space continuity).
*   Calculating evaluation metrics: Reconstruction Loss, and **Frechet Inception Distance (FID)** for sample quality. `compute_fid(real, fake, train_loader)` is self-contained — it briefly trains a small CNN classifier on the dataset and measures the Frechet distance in its feature space (the standard lightweight "MNIST-FID"), so no torchvision/InceptionV3 download is needed. Lower = closer to the real distribution.

---

## The Catalog of Scripts

The scripts are organized chronologically and by generative paradigm:

### 01. Autoencoder (`01.autoencoder.py`)
*   **Description:** Non-probabilistic reconstruction bottleneck.
*   **Architecture:** Encoder (reduces $28 \times 28 \to$ code vector size $D$) and Decoder (maps $D \to 28 \times 28$).
*   **Educational Takeaway:** Understanding reconstruction losses (MSE) and recognizing why a standard AE's latent space is discontinuous (i.e., sampling a random $z$ does not guarantee a realistic image).

### 02. Variational Autoencoder (`02.vae.py`)
*   **Description:** Probabilistic latent space mapping (Kingma & Welling, 2013).
*   **Architecture:** Encoder maps inputs to mean $\mu$ and variance $\sigma$. A reparameterization trick ($z = \mu + \epsilon \cdot \sigma$) allows backpropagation. Trained with Reconstruction Loss + KL Divergence.
*   **Educational Takeaway:** Force the latent space to follow a standard normal distribution, enabling stable sampling of new, synthetic fashion designs.

### 03. GAN & DCGAN (`03.gan-dcgan.py`)
*   **Description:** Adversarial mini-max game between a Generator and a Discriminator.
*   **Variants:**
    *   `vanilla` - Basic linear/MLP GAN (Goodfellow et al., 2014).
    *   `dcgan` - Deep Convolutional GAN using transposed convolutions and batch norm (Radford et al., 2015).
*   **Educational Takeaway:** Understanding the game-theoretic training process and experiencing mode collapse (where the generator outputs the same design repeatedly).

### 04. Wasserstein GAN with GP (`04.wgan-gp.py`)
*   **Description:** Stable adversarial training using Earth Mover's distance (Gulrajani et al., 2017).
*   **Architecture:** Replaces the classification Discriminator with a regression Critic, enforcing a Lipschitz constraint via Gradient Penalty (GP).
*   **Educational Takeaway:** Seeing how Wasserstein distance prevents vanishing gradients and solves the instability issues of vanilla GAN training.

### 05. Conditional GAN (`05.cgan.py`)
*   **Description:** Class-conditioned generation (Mirza & Osindero, 2014).
*   **Architecture:** Embeds class labels (e.g., "Handbag", "Sneaker") and concatenates them with both the latent noise vector $z$ and the convolutional feature maps.
*   **Educational Takeaway:** Moving from random synthesis to targeted class-conditional synthesis.

### 06. Diffusion DDPM (`06.diffusion-ddpm.py`)
*   **Description:** Iterative denoising model (Ho et al., 2020).
*   **Architecture:** A forward process adds Gaussian noise to images over $T$ steps. The network (a small convolutional U-Net with residual connections) is trained to predict the noise added at any given step $t$.
*   **Educational Takeaway:** The modern state-of-the-art paradigm in generative modeling: breaking generation down into a sequence of small, stable denoising steps.

### 07. PixelCNN (`07.pixelcnn.py`)
*   **Description:** Autoregressive image model (van den Oord et al., 2016) — `p(x) = prod_i p(x_i | x_<i)`.
*   **Architecture:** Masked convolutions (type A then B) enforce raster-scan causality; a 1x1 conv outputs a categorical distribution over pixel intensities.
*   **Educational Takeaway:** Exact, tractable likelihood trained with plain cross-entropy. The price is sequential, one-pass-per-pixel sampling.

### 08. Normalizing Flow — RealNVP (`08.normalizing-flow.py`)
*   **Description:** Exact-likelihood generative model via invertible affine coupling layers (Dinh et al., 2017).
*   **Architecture:** Alternating-mask coupling layers; the triangular Jacobian makes log-det (and inversion for sampling) cheap. Trained by pure maximum likelihood.
*   **Educational Takeaway:** The only paradigm here offering both exact likelihood and exact latent inference; invertibility constrains the architecture.

### 09. VQ-VAE (`09.vqvae.py`)
*   **Description:** Autoencoder with a discrete codebook latent (van den Oord et al., 2017).
*   **Architecture:** Encoder → nearest-codebook quantization (straight-through gradients) → decoder; codebook + commitment losses. Sampling uses a per-position code marginal (a weak prior).
*   **Educational Takeaway:** Discrete latents + straight-through estimation; the basis of token-based image generation. A strong prior (e.g. PixelCNN over codes) would complete it.

### 10. DDIM (`10.ddim.py`)
*   **Description:** Deterministic, few-step sampling for a DDPM-trained noise predictor (Song et al., 2021).
*   **Architecture:** Same U-Net/training as DDPM; a non-Markovian deterministic reverse process (eta=0) with x0-thresholding skips most timesteps (e.g. 25 vs 200).
*   **Educational Takeaway:** Decouples sampling steps from training steps — the practical speed-up that made diffusion usable. (`--ddim-steps N`)

### 11. Classifier-Free Guided Diffusion (`11.guided-diffusion.py`)
*   **Description:** Conditional diffusion steered by classifier-free guidance (Ho & Salimans, 2022).
*   **Architecture:** One U-Net trained both conditionally and unconditionally (label dropout to a null token); sampling combines the two predictions, `eps = eps_uncond + w·(eps_cond − eps_uncond)`, with x0-thresholding for stability.
*   **Educational Takeaway:** Controllable, class-conditioned synthesis (the diffusion analogue of CGAN); higher `--guidance` = more class-typical but less diverse.

---

## Expected Output Graphics & Comparisons

When you execute these scripts, they will write visual artifacts to this folder. Comparing them provides deep insights into each model's internal representations:

### 1. Reconstructions Comparison: AE vs. VAE
*   **Takeaway:** If you open `ae_reconstructions.png` and `vae_reconstructions.png`, you will notice that the Autoencoder reconstructions are slightly sharper and contain more pixel-level details than the Variational Autoencoder.
*   **Why?** The standard AE only optimizes reconstruction error (MSE). The VAE has a regularizing KL divergence penalty that forces the latent distribution to behave like a standard normal distribution. This regularization introduces a trade-off: it makes the latent space smooth, but limits the capacity to store extreme pixel-level exceptions, resulting in slightly blurrier images.

### 2. Latent Walk Comparison: AE vs. VAE
*   **Takeaway:** Compare `ae_latent_walk.png` and `vae_latent_walk.png` (which perform spherical linear interpolation `slerp` between two random points).
*   **Why?** The Autoencoder's walk will often show sharp, abrupt transitions or intermediate frames of meaningless noise. This is because the AE's latent space has "gaps" (unmapped regions). In contrast, the VAE's walk is smooth, showing a gradual morph from one category (e.g., shoe) to another (e.g., shirt) without passing through non-fashion noise.

### 3. Adversarial Quality & Diversity: DCGAN vs. MLP-GAN
*   **Takeaway:** Compare `dcgan_generated_samples.png` and `mlp_generated_samples.png`.
*   **Why?** The MLP-based GAN generates noisy, distorted pixel blobs, and often suffers from *mode collapse* (generating the same item repeatedly for different noise seeds). The Deep Convolutional GAN (DCGAN) leverages spatial convolutions to produce sharp, realistic item silhouettes with much higher fidelity.

### 4. Stability: WGAN-GP vs. DCGAN
*   **Takeaway:** WGAN-GP (`wgan_generated_samples.png`) exhibits highly diverse and stable samples.
*   **Why?** Standard GAN training can diverge, causing the generator loss to fluctuate wildly. WGAN-GP uses Wasserstein distance and gradient penalty constraint to stabilize training, preventing vanishing gradients and guaranteeing smooth optimization curves.

### 5. Directed Synthesis: CGAN
*   **Takeaway:** Check `cgan_generated_samples.png`.
*   **Why?** Unlike standard GANs which generate items at random, CGAN outputs a structured grid where each row represents a specific class (0 through 9). This demonstrates targeted, label-guided synthesis.

### 6. Modern State-of-the-Art: DDPM Diffusion
*   **Takeaway:** Check `diffusion_generated_samples.png` and `diffusion_denoising_walk.png`.
*   **Why?** Instead of generating the image in one step (which makes GAN training unstable), DDPM learns to iteratively denoise a random Gaussian vector. The walk shows the progressive birth of a crisp silhouette out of pure static noise across the denoising timeline.

