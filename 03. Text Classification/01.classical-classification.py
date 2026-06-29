"""
01. Classical Text Classification Baselines
===========================================

Frequency-based text classification benchmarks.

Models:
    * Naive Bayes: Multinomial Naive Bayes using TF-IDF feature frequencies.
    * Logistic Regression: Logistic Regression classifier over TF-IDF matrices.

Run:
    python "01.classical-classification.py" --variant naive_bayes
    python "01.classical-classification.py" --variant logistic_regression
"""

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import tc_common as mc


def main():
    # Build standard argparser
    p = mc.build_argparser("Classical Text classification models")
    p.add_argument("--classical-variant", type=str, default="naive_bayes",
                   choices=["naive_bayes", "logistic_regression"])
    args = p.parse_args()

    variant = args.classical_variant if args.classical_variant != "naive_bayes" else (args.variant or "naive_bayes")
    if variant not in ["naive_bayes", "logistic_regression"]:
        variant = "naive_bayes"

    print(f"Loading IMDB classification reviews (limit={args.limit})...")
    reviews = mc.load_imdb(limit=args.limit)

    texts = [r[0] for r in reviews]
    labels = np.array([r[1] for r in reviews], dtype=np.int64)

    # Train/Test Split (80% / 20%)
    split = int(len(texts) * 0.8)
    train_texts, test_texts = texts[:split], texts[split:]
    y_train, y_test = labels[:split], labels[split:]

    print(f"Vectorizing texts using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    if variant == "naive_bayes":
        model = MultinomialNB()
        model_name = "Naive-Bayes"
    elif variant == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
        model_name = "Logistic-Regression"
    else:
        raise ValueError(f"Unknown variant: {variant}")

    print(f"Training {model_name}...")
    model.fit(X_train, y_train)

    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)

    # Report
    mc.report_classification(
        y_test, y_pred, model_name=model_name,
        save_dir=None if args.no_figure else os.path.dirname(os.path.abspath(__file__))
    )


if __name__ == "__main__":
    main()
