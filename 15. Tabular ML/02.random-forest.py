"""
02. Random Forest Classifier
============================

Bootstrap aggregated (bagged) ensemble combining multiple independent decision trees (Breiman, 2001).

Architecture Diagram / Layout:
    Input Features -> Bootstrap Sample 1 -> Tree 1 -> Prediction -\
                   -> Bootstrap Sample 2 -> Tree 2 -> Prediction ---> Majority Voting Prediction
                   -> Bootstrap Sample N -> Tree N -> Prediction -/

Key insights / educational takeaways:
    * Bagging trains individual trees on random subsets of samples and features to decorrelate individual estimators.
    * Aggregating votes reduces ensemble variance, shielding the model from overfitting compared to single trees.

Run:
    python "02.random-forest.py" --n-estimators 100 --max-depth 5
"""

import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import tabular_common as mc


def main():
    p = mc.build_argparser("Random Forest Tabular Classifier", max_depth=5, n_estimators=100)
    args = p.parse_args()

    # Load wine dataset
    X_train, X_test, y_train, y_test, feature_names, class_names = mc.load_wine_dataset()

    if args.limit is not None:
        X_train, y_train = X_train[:args.limit], y_train[:args.limit]

    # Create and train Random Forest model
    print(f"Training Random Forest (n_estimators={args.n_estimators}, max_depth={args.max_depth})...")
    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
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
        chart_path = os.path.join(save_dir, "rf_feature_importances.png")
        mc.plot_feature_importances(model.feature_importances_, feature_names,
                                    chart_path, "Random Forest Feature Importances")


if __name__ == "__main__":
    main()
