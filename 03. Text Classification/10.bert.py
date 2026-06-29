"""
10. Fine-tuning a Pretrained Transformer — DistilBERT (Sanh et al., 2019; Devlin et al., 2018)
=============================================================================================

Every other model here learns from the IMDB labels alone. This one starts from
**DistilBERT**, a Transformer already pretrained on billions of words of English,
and merely *fine-tunes* it on sentiment. That transfer of general language
knowledge is why pretrained models dominate modern NLP — they reach high accuracy
with little task data and few epochs.

Architecture Diagram / Layout:
    Input text -> WordPiece tokenizer -> [CLS] tokens ... [SEP]
       -> Pretrained DistilBERT encoder (6 Transformer layers)
       -> [CLS] representation -> classification head -> 2 logits

Key insights / educational takeaways:
    * Pretraining + fine-tuning beats training from scratch, especially with
      limited labels — contrast the accuracy/epochs here with 05.transformer.
    * This script adds a real dependency (`transformers`) and downloads weights
      (~270 MB) on first run; it is intentionally the "batteries-not-included"
      showcase of the folder. Use --limit and few epochs on CPU.

Requires:
    pip install transformers

Run:
    python "10.bert.py" --limit 2000 --epochs 2
    python "10.bert.py" --limit 500 --epochs 1 --device cpu     # quick smoke test
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import tc_common as mc

MODEL_NAME = "distilbert-base-uncased"


def main():
    p = mc.build_argparser("DistilBERT Fine-tuned Sentiment Classifier",
                           epochs=2, batch_size=16, lr=2e-5)
    p.add_argument("--max-len", type=int, default=256, help="WordPiece tokens per review")
    args = p.parse_args()
    device = mc.get_device(args.device)

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        print("This demo needs the 'transformers' library. Install it with:\n"
              "    pip install transformers\n")
        raise SystemExit(0)

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    except Exception as e:
        print(f"Could not load pretrained '{MODEL_NAME}' ({e}).\n"
              "A network connection is required on first run to download the weights.")
        raise SystemExit(0)

    # Data: raw text -> WordPiece ids + attention masks
    reviews = mc.load_imdb(args.limit)
    texts = [r[0] for r in reviews]
    labels = np.array([r[1] for r in reviews], dtype=np.int64)
    enc = tokenizer(texts, truncation=True, padding="max_length",
                    max_length=args.max_len, return_tensors="np")
    ids = torch.from_numpy(enc["input_ids"])
    mask = torch.from_numpy(enc["attention_mask"])
    y = torch.from_numpy(labels)

    split = int(len(ids) * 0.8)
    train_ds = TensorDataset(ids[:split], mask[:split], y[:split])
    test_ds = TensorDataset(ids[split:], mask[split:], y[split:])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_params = sum(pm.numel() for pm in model.parameters() if pm.requires_grad)
    print(f"Device: {device} | fine-tuning {MODEL_NAME} | params: {n_params:,}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = total = 0.0
        for input_ids, attn, yb in train_loader:
            input_ids, attn, yb = input_ids.to(device), attn.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attn, labels=yb)
            out.loss.backward()
            optimizer.step()
            running += out.loss.item() * yb.size(0)
            total += yb.size(0)
        print(f"Epoch {epoch:2d}/{args.epochs} | train_loss {running / total:.4f}")
    print("-" * 64)

    # Evaluate
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for input_ids, attn, yb in test_loader:
            logits = model(input_ids=input_ids.to(device), attention_mask=attn.to(device)).logits
            y_pred.append(logits.argmax(-1).cpu().numpy())
            y_true.append(yb.numpy())
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

    mc.report_classification(y_true, y_pred, model_name="DistilBERT",
                             save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
