"""
06. Gaussian Naive Bayes (Maron, 1961; "naive" independence assumption)
=======================================================================

The simplest *generative* classifier. Instead of learning a decision boundary
directly (discriminative), it models how each class *generates* its features and
applies Bayes' rule. "Naive" = it assumes every feature is conditionally
independent given the class, so each class-conditional density factorizes into
one 1-D Gaussian per feature.

Architecture Diagram / Layout:
    For each class c, fit per-feature Gaussians:  p(x_j | c) = N(mu_{c,j}, sig_{c,j})
    Prediction (Bayes' rule, log space):
        log P(c | x)  ~  log P(c)  +  sum_j log N(x_j; mu_{c,j}, sig_{c,j})
        y_hat = argmax_c  log P(c | x)

Key insights / educational takeaways:
    * Generative vs. discriminative: NB estimates class-conditional densities
      p(x|c) and priors p(c), then inverts with Bayes' rule — a different
      philosophy from logistic regression's direct p(c|x).
    * Training is essentially free: one pass computing per-class means/variances,
      no iterative optimization.
    * The independence assumption is almost always false (features correlate),
      yet NB is a solid, well-calibrated baseline — and it degrades gracefully
      as a probabilistic model even when the assumption is violated.

Run:
    python "06.naive-bayes.py" --dataset covtype
    python "06.naive-bayes.py" --limit 2000        # fast smoke test
"""

import os
from sklearn.naive_bayes import GaussianNB
import tabular_common as mc


def main():
    p = mc.build_argparser("Gaussian Naive Bayes Tabular Classifier")
    args = p.parse_args()
    mc.set_seed(args.seed)

    X_train, X_test, y_train, y_test, feature_names, class_names = \
        mc.load_classification_dataset(args.dataset)
    if args.limit is not None:
        X_train, y_train = X_train[:args.limit], y_train[:args.limit]

    print("Training Gaussian Naive Bayes (single closed-form pass)...")
    model = GaussianNB()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    mc.report_classification(y_test, preds, proba, class_names=class_names,
                             model_name="GaussianNB",
                             save_dir=None if args.no_figure else
                             os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
