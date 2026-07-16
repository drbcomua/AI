"""
03. Gradient Boosting Classifier
================================

Sequential ensemble adding decision trees iteratively to minimize the residuals of preceding estimators (Friedman, 2001).

Architecture Diagram / Layout:
    Input Features -> Base Prediction F_0 (Majority Prior)
                   -> Step 1 -> Fit Tree 1 to Loss Gradients -> Update F_1 = F_0 + lr * Tree 1
                   -> Step N -> Fit Tree N to Loss Gradients -> Update F_N = F_{N-1} + lr * Tree N

Key insights / educational takeaways:
    * Boosting constructs weak trees sequentially (each tree correcting the remaining residuals of the current ensemble).
    * Reduces model bias iteratively by descending gradient paths on the classification loss function.

Run:
    python "03.gradient-boosting.py" --n-estimators 100 --lr 0.1 --max-depth 3
"""

import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import tabular_common as mc


def main():
    p = mc.build_argparser("Gradient Boosting Tabular Classifier", max_depth=3, n_estimators=100, lr=0.1)
    args = p.parse_args()

    # Load wine dataset
    X_train, X_test, y_train, y_test, feature_names, class_names = mc.load_wine_dataset()

    if args.limit is not None:
        X_train, y_train = X_train[:args.limit], y_train[:args.limit]

    # Create and train Gradient Boosting model
    print(f"Training Gradient Boosting (n_estimators={args.n_estimators}, lr={args.lr}, max_depth={args.max_depth})...")
    model = GradientBoostingClassifier(n_estimators=args.n_estimators, learning_rate=args.lr,
                                       max_depth=args.max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average="macro")

    print("-" * 64)
    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Macro F1 Score: {test_f1:.4f}")

    # Plot feature importances
    if not args.no_figure:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        chart_path = os.path.join(save_dir, "gb_feature_importances.png")
        mc.plot_feature_importances(model.feature_importances_, feature_names,
                                    chart_path, "Gradient Boosting Feature Importances")


if __name__ == "__main__":
    main()
