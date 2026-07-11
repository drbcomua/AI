"""
03. Visual Question Answering Baseline (Antol et al., 2015)
=============================================================

The original VQA baseline: a CNN image encoder and a recurrent question encoder
are fused via an element-wise (Hadamard) product into a single joint vector, which
an MLP classifies into one of a fixed set of answers.

Architecture Diagram / Layout:
    Image [B x 3 x 64 x 64] -> ImageEncoder (CNN)    -> Image Embed [B x 64]
                                                                 \\
                                                                  * (Hadamard product) -> MLP Classifier -> Answer Logits [B x 8]
                                                                 /
    Question [B x SeqLen]   -> QuestionEncoder (GRU) -> Question Embed [B x 64]

Key insights / educational takeaways:
    * The simplest possible multi-modal fusion: combine independently encoded modalities
      with a single element-wise product before classification.
    * Frames vision-language reasoning as closed-set classification rather than
      retrieval (CLIP) or generation (Image Captioner), the third major multi-modal task family.
    * The Hadamard fusion lets each embedding dimension "gate" the other modality, but cannot
      model fine-grained spatial reasoning -- motivating the attention-based fusions in later scripts.

Run:
    python "03.vqa-baseline.py" --epochs 10
    python "03.vqa-baseline.py" --limit 2000        # fast smoke test
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
import multimodal_common as mc


class ImageEncoder(nn.Module):
    """Simple 2D CNN extracting visual features."""
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 32x32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # -> 8x8
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        return self.fc(h)


class QuestionEncoder(nn.Module):
    """Simple GRU question encoder projecting tokenized questions into shared dimensions."""
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64, embedding_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        emb = self.embedding(x)
        _, hn = self.gru(emb) # hn shape: [1, B, hidden_dim]
        return self.fc(hn[0])


class VQABaseline(nn.Module):
    """Joint-fusion classifier predicting an answer class from an (image, question) pair."""
    def __init__(self, vocab_size: int, n_answers: int = 8, embedding_dim: int = 64):
        super().__init__()
        self.image_encoder = ImageEncoder(embedding_dim)
        self.question_encoder = QuestionEncoder(vocab_size, embedding_dim=embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_answers),
        )

    def forward(self, image, question):
        img_feat = self.image_encoder(image)
        q_feat = self.question_encoder(question)
        fused = img_feat * q_feat # Hadamard product fusion (Antol et al. baseline)
        return self.classifier(fused)


def main():
    p = mc.build_argparser("VQA Baseline (Image-Question Fusion Classifier)", epochs=10, batch_size=64)
    args = p.parse_args()

    device = mc.get_device(args.device)

    # Generate multi-modal shapes dataset + synthetic VQA question/answer pairs
    images, captions = mc.generate_shapes_dataset(num_samples=args.limit or 1200)
    questions, answers, answer_idx = mc.generate_vqa_pairs(captions)
    q_tokens = mc.tokenize_captions(questions)

    # Train / test split (80/20)
    split_idx = int(len(images) * 0.8)
    train_img, train_q, train_y = images[:split_idx], q_tokens[:split_idx], answer_idx[:split_idx]
    test_img, test_q, test_y = images[split_idx:], q_tokens[split_idx:], answer_idx[split_idx:]
    test_questions = questions[split_idx:]
    test_answers = answers[split_idx:]

    model = VQABaseline(vocab_size=len(mc.VOCAB), n_answers=len(mc.ANSWER_LIST)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("Training VQA Baseline Model...")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | trainable params: {n_params:,}")
    print("-" * 64)

    train_dataset = TensorDataset(train_img, train_q, train_y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        total = 0

        for img, q, y in train_loader:
            img, q, y = img.to(device), q.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(img, q)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * img.size(0)
            total += img.size(0)

        epoch_loss_avg = epoch_loss / total
        print(f"Epoch {epoch:2d}/{args.epochs} | loss: {epoch_loss_avg:.4f}")

    print("-" * 64)

    # Evaluate on the held-out test set
    model.eval()
    with torch.no_grad():
        test_img_d, test_q_d = test_img.to(device), test_q.to(device)
        logits = model(test_img_d, test_q_d)
        preds = logits.argmax(dim=1).cpu().numpy()

    y_true = test_y.numpy()
    acc = float(np.mean(preds == y_true))
    print(f"Test Answer Accuracy: {acc * 100:.2f}%")
    print()
    print(classification_report(y_true, preds, labels=list(range(len(mc.ANSWER_LIST))),
                                 target_names=mc.ANSWER_LIST, zero_division=0))

    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        grid_path = os.path.join(save_dir, "vqa_baseline_results.png")

        val_images = test_img[:6]
        val_questions = test_questions[:6]
        val_gt = test_answers[:6]
        val_pred = [mc.ANSWER_LIST[i] for i in preds[:6]]

        mc.plot_vqa_grid(val_images, val_questions, val_gt, val_pred, grid_path)


if __name__ == "__main__":
    main()
