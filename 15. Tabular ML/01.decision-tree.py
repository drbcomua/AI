"""
01. Decision Tree Classifier
============================

A single greedy decision tree recursively splitting features to minimize Gini impurity (Breiman et al., 1984).

Architecture Diagram / Layout:
    Input Features -> Feature Split Threshold check (e.g., alcohol <= 12.8)
                          |----> [True]  -> Left Child Split / Leaf Node
                          |----> [False] -> Right Child Split / Leaf Node

Key insights / educational takeaways:
    * Demonstrates recursive partitioning strategies that divide feature spaces into orthogonal decision hyperplanes.
    * Feature importance rankings reveal which indicators reduce node impurity the most across split thresholds.

Run:
    python "01.decision-tree.py" --max-depth 4
"""

import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import tabular_common as mc


def main():
    p = mc.build_argparser("Decision Tree Tabular Classifier", max_depth=4)
    args = p.parse_args()

    # Load wine dataset
    X_train, X_test, y_train, y_test, feature_names, class_names = mc.load_wine_dataset()

    if args.limit is not None:
        X_train, y_train = X_train[:args.limit], y_train[:args.limit]

    # Create and train Decision Tree model
    print(f"Training Decision Tree (max_depth={args.max_depth})...")
    model = DecisionTreeClassifier(max_depth=args.max_depth, random_state=42)
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
        chart_path = os.path.join(save_dir, "dt_feature_importances.png")
        mc.plot_feature_importances(model.feature_importances_, feature_names,
                                    chart_path, "Decision Tree Feature Importances")


if __name__ == "__main__":
    main()
