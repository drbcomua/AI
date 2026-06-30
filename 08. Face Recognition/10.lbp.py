"""
10. Local Binary Patterns (LBP) — a classical texture baseline
==============================================================

A non-deep face descriptor that complements the Eigen/Fisherfaces baselines (01).
Where PCA/LDA model *global* appearance from raw pixels, LBP captures *local
texture* and is robust to monotonic lighting changes. For each pixel it thresholds
its 8 neighbors against the center to form an 8-bit code; the image is divided into
a grid of cells, a histogram of codes is taken per cell, and the concatenated,
normalized histograms form the descriptor. Faces are then verified by the
chi-square distance between descriptors — no learning at all.

Key insights / educational takeaways:
    * Hand-crafted local texture statistics already give non-trivial verification,
      and show how far the field came once embeddings were *learned* (02-09).
    * Spatial cell histograms add coarse position information to the texture code.

Run:
    python "10.lbp.py"
"""

import os
import numpy as np
import face_common as mc


def lbp_descriptor(images_np, grid=8):
    """images_np: [N, 3, H, W] in [0,1] -> [N, grid*grid*256] spatial LBP histograms."""
    gray = images_np.mean(axis=1)                              # [N, H, W]
    n, h, w = gray.shape
    padded = np.pad(gray, ((0, 0), (1, 1), (1, 1)), mode="edge")
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    code = np.zeros((n, h, w), dtype=np.int32)
    for k, (dy, dx) in enumerate(offsets):
        neighbor = padded[:, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        code += ((neighbor >= gray).astype(np.int32) << k)

    ch, cw = max(1, h // grid), max(1, w // grid)
    descs = []
    for img_code in code:
        hist = []
        for i in range(grid):
            for j in range(grid):
                cell = img_code[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]
                hist.append(np.histogram(cell, bins=256, range=(0, 256))[0])
        d = np.concatenate(hist).astype(np.float32)
        descs.append(d / (d.sum() + 1e-8))
    return np.array(descs)


def main():
    mc.build_argparser("Local Binary Patterns (LBP) baseline").parse_args()

    images, labels = mc.load_lfw()
    _, _, test_img, test_lbl = mc.split_lfw_identities(images, labels)

    print("Generating verification pairs and computing LBP descriptors...")
    p1, p2, pair_lbls = mc.generate_verification_pairs(test_img, test_lbl, num_pairs=1000)
    d1 = lbp_descriptor(p1.numpy())
    d2 = lbp_descriptor(p2.numpy())

    # chi-square distance between the two histogram descriptors
    distances = 0.5 * np.sum((d1 - d2) ** 2 / (d1 + d2 + 1e-8), axis=1)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    mc.plot_verification_roc(pair_lbls.numpy(), distances,
                             os.path.join(save_dir, "lbp_baseline_roc.png"), "LBP")


if __name__ == "__main__":
    main()
